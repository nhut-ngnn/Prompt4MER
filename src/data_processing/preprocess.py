import argparse
import glob
import logging
import os
import pickle
import random
import re
from collections import Counter

import numpy as np
import soundfile as sf
import torch
import tqdm
from sklearn.model_selection import train_test_split

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

LABEL_MAP_4 = {
    "ang": 0,
    "hap": 1,
    "sad": 2,
    "neu": 3,
    "exc": 1,
}
MSP_IMPROV_LABEL_MAP = {
    "A": 0,
    "H": 1,
    "S": 2,
    "N": 3,
}

VIDEO_FILE_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".flv")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _first_existing_path(candidates):
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def _build_video_candidates(video_root, keys):
    candidates = []
    for key in keys:
        if not key:
            continue
        norm_key = os.path.normpath(str(key))
        candidates.append(os.path.join(video_root, norm_key))
        for ext in VIDEO_FILE_EXTS:
            candidates.append(os.path.join(video_root, f"{norm_key}{ext}"))
    return candidates


def _infer_video_from_audio_path(audio_path):
    if not audio_path:
        return None

    audio_str = str(audio_path)
    stem = os.path.splitext(audio_str)[0]
    candidates = [stem]
    for ext in VIDEO_FILE_EXTS:
        candidates.append(f"{stem}{ext}")
    return _first_existing_path(candidates)


def _infer_iemocap_utterance_video_path(video_root, utt_id, dialog_id, wav_path):
    nearby_video = _infer_video_from_audio_path(wav_path)
    if nearby_video is not None:
        return nearby_video

    if not video_root:
        return None

    wav_stem = os.path.splitext(os.path.basename(wav_path))[0]
    keys = [
        os.path.join(dialog_id, utt_id),
        os.path.join(dialog_id, wav_stem),
        utt_id,
        wav_stem,
    ]
    return _first_existing_path(_build_video_candidates(video_root, keys))


def _infer_iemocap_dialog_video_path(data_root, sess_id, dialog_id, video_root=None):
    roots = []
    if video_root:
        roots.extend(
            [
                os.path.join(video_root, "Session{0}".format(sess_id), "dialog", "avi", "DivX"),
                os.path.join(video_root, "Session{0}".format(sess_id), "dialog", "avi"),
                os.path.join(video_root, "dialog", "avi", "DivX"),
                os.path.join(video_root, "dialog", "avi"),
                video_root,
            ]
        )

    roots.extend(
        [
            os.path.join(data_root, f"Session{sess_id}", "dialog", "avi", "DivX"),
            os.path.join(data_root, f"Session{sess_id}", "dialog", "avi"),
        ]
    )

    candidates = []
    for root in roots:
        if not root:
            continue
        for ext in VIDEO_FILE_EXTS:
            candidates.append(os.path.join(root, f"{dialog_id}{ext}"))
        candidates.append(os.path.join(root, dialog_id))

    return _first_existing_path(candidates)


def _build_sample(
    sample_id,
    audio_path,
    text,
    label,
    video_path=None,
    start_time=None,
    end_time=None,
    dialog_id=None,
    session_id=None,
):
    if isinstance(label, np.generic):
        label = label.item()
    if isinstance(label, np.ndarray):
        if label.size == 1:
            label = label.item()
        else:
            label = label.tolist()

    sample = {
        "sample_id": sample_id,
        "audio_path": audio_path,
        "video_path": video_path,
        "text": "" if text is None else str(text),
        "emotion": label,
    }
    if start_time is not None:
        sample["start_time"] = float(start_time)
    if end_time is not None:
        sample["end_time"] = float(end_time)
    if dialog_id is not None:
        sample["dialog_id"] = dialog_id
    if session_id is not None:
        sample["session_id"] = int(session_id)
    return sample


def _stratify_labels_or_none(labels):
    if not labels:
        return None
    valid_labels = []
    for value in labels:
        if isinstance(value, (float, np.floating)):
            return None
        valid_labels.append(value)

    counts = Counter(valid_labels)
    if len(counts) < 2:
        return None
    if min(counts.values()) < 2:
        return None
    return valid_labels


def _split_train_val_test(samples, labels, seed, test_size=0.1, val_size=0.1):
    if len(samples) < 3:
        raise ValueError("At least 3 samples are required to split train/val/test.")

    stratify = _stratify_labels_or_none(labels)
    train_val, test_samples = train_test_split(
        samples,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )

    train_val_labels = [sample["emotion"] for sample in train_val]
    stratify_train_val = _stratify_labels_or_none(train_val_labels)
    val_ratio = val_size / (1.0 - test_size)

    train_samples, val_samples = train_test_split(
        train_val,
        test_size=val_ratio,
        random_state=seed,
        stratify=stratify_train_val,
    )

    return train_samples, val_samples, test_samples


def _save_splits(output_dir, train_samples, val_samples, test_samples):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "train.pkl"), "wb") as f:
        pickle.dump(train_samples, f)
    with open(os.path.join(output_dir, "val.pkl"), "wb") as f:
        pickle.dump(val_samples, f)
    with open(os.path.join(output_dir, "test.pkl"), "wb") as f:
        pickle.dump(test_samples, f)


def preprocess_iemocap(args):
    if not args.data_root:
        raise ValueError("--data_root is required for IEMOCAP preprocessing.")

    session_ids = list(range(1, 6))
    ignore_length = args.ignore_length
    seed = args.seed
    data_root = args.data_root

    valid_emotions = {"ang", "hap", "sad", "neu", "exc"}
    samples = []

    for sess_id in tqdm.tqdm(session_ids, desc="Processing IEMOCAP"):
        sess_path = os.path.join(data_root, f"Session{sess_id}")
        audio_root = os.path.join(sess_path, "sentences/wav")
        text_root = os.path.join(sess_path, "dialog/transcriptions")
        label_root = os.path.join(sess_path, "dialog/EmoEvaluation")

        label_files = glob.glob(os.path.join(label_root, "*.txt"))

        for label_file in label_files:
            base_name = os.path.basename(label_file)
            transcript_file = os.path.join(text_root, base_name)
            if not os.path.isfile(transcript_file):
                logging.warning("Transcript file not found: %s", transcript_file)
                continue

            transcript_lines = {}
            with open(transcript_file, "r") as f:
                for line in f:
                    if ":" not in line:
                        continue
                    left, right = line.split(":", 1)
                    transcript_lines[left.strip()] = right.strip()

            with open(label_file, "r") as f:
                for line in f:
                    if not line.startswith("["):
                        continue

                    data = line[1:].split()
                    if len(data) < 5:
                        continue

                    start_time = float(data[0])
                    end_time = float(data[2][:-1])
                    utt_id = data[3]
                    emotion = data[4]

                    if emotion not in valid_emotions:
                        continue

                    dialog_id = utt_id[:-5]
                    wav_name = f"{utt_id}.wav"
                    wav_path = os.path.join(audio_root, dialog_id, wav_name)

                    try:
                        wav_data, _ = sf.read(wav_path, dtype="int16")
                    except Exception:
                        logging.warning("Cannot read %s", wav_path)
                        continue

                    if len(wav_data) < ignore_length:
                        logging.warning("Ignored short sample: %s", wav_path)
                        continue

                    text_key = f"{utt_id} [{start_time:08.4f}-{end_time:08.4f}]"
                    text = transcript_lines.get(text_key)

                    if text is None:
                        text_key_alt1 = f"{utt_id} [{start_time:08.4f}-{end_time + 0.0001:08.4f}]"
                        text_key_alt2 = f"{utt_id} [{start_time + 0.0001:08.4f}-{end_time:08.4f}]"
                        text = transcript_lines.get(text_key_alt1) or transcript_lines.get(text_key_alt2)

                    if text is None:
                        logging.warning("Transcript not found: %s", text_key)
                        continue

                    label = LABEL_MAP_4.get(emotion)
                    if label is None:
                        continue

                    utt_video_path = _infer_iemocap_utterance_video_path(
                        video_root=args.video_root,
                        utt_id=utt_id,
                        dialog_id=dialog_id,
                        wav_path=wav_path,
                    )
                    dialog_video_path = _infer_iemocap_dialog_video_path(
                        data_root=data_root,
                        sess_id=sess_id,
                        dialog_id=dialog_id,
                        video_root=args.video_root,
                    )

                    if utt_video_path is not None:
                        video_path = utt_video_path
                        clip_start = None
                        clip_end = None
                    else:
                        video_path = dialog_video_path
                        clip_start = start_time if dialog_video_path is not None else None
                        clip_end = end_time if dialog_video_path is not None else None

                    sample = _build_sample(
                        sample_id=utt_id,
                        audio_path=wav_path,
                        text=text,
                        label=label,
                        video_path=video_path,
                        start_time=clip_start,
                        end_time=clip_end,
                        dialog_id=dialog_id,
                        session_id=sess_id,
                    )
                    samples.append(sample)

    if not samples:
        raise ValueError("No IEMOCAP samples found. Check --data_root and dataset structure.")

    random.Random(seed).shuffle(samples)
    labels = [sample["emotion"] for sample in samples]
    train_samples, val_samples, test_samples = _split_train_val_test(samples, labels, seed)

    output_dir = os.path.join(args.output_root, "IEMOCAP_preprocessed")
    _save_splits(output_dir, train_samples, val_samples, test_samples)

    logging.info(
        "IEMOCAP - Train: %d | Val: %d | Test: %d",
        len(train_samples),
        len(val_samples),
        len(test_samples),
    )
    logging.info("IEMOCAP - Saved preprocessed data to %s", output_dir)


def _read_msp_improv_transcript(transcript_path):
    if not os.path.isfile(transcript_path):
        return ""
    with open(transcript_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.strip() for line in f if line.strip()]
    if not lines:
        return ""
    if len(lines) == 1:
        return lines[0]
    return " ".join(lines)


def _msp_improv_label_from_sample_id(sample_id):
    match = re.match(r"MSP-IMPROV-S\d+([AHNS])-", sample_id)
    if match is None:
        return None
    return MSP_IMPROV_LABEL_MAP.get(match.group(1))


def preprocess_msp_improv(args):
    if not args.data_root:
        raise ValueError("--data_root is required for MSP-IMPROV preprocessing.")

    audio_root = args.audio_root or os.path.join(args.data_root, "Audio")
    transcript_root = os.path.join(
        args.data_root,
        "Human_transcriptions",
        "All_human_transcriptions",
    )
    wav_files = sorted(glob.glob(os.path.join(audio_root, "**", "*.wav"), recursive=True))
    samples = []

    for wav_path in tqdm.tqdm(wav_files, desc="Processing MSP-IMPROV"):
        sample_id = os.path.splitext(os.path.basename(wav_path))[0]
        label = _msp_improv_label_from_sample_id(sample_id)
        if label is None:
            logging.warning("Skipped MSP-IMPROV sample with unknown label: %s", sample_id)
            continue

        try:
            wav_data, _ = sf.read(wav_path, dtype="int16")
        except Exception:
            logging.warning("Cannot read %s", wav_path)
            continue

        if len(wav_data) < args.ignore_length:
            logging.warning("Ignored short sample: %s", wav_path)
            continue

        transcript_path = os.path.join(transcript_root, f"{sample_id}.txt")
        text = _read_msp_improv_transcript(transcript_path)
        samples.append(
            _build_sample(
                sample_id=sample_id,
                audio_path=wav_path,
                text=text,
                label=label,
                video_path=None,
            )
        )

    if not samples:
        raise ValueError("No MSP-IMPROV samples found. Check --data_root and dataset structure.")

    random.Random(args.seed).shuffle(samples)
    labels = [sample["emotion"] for sample in samples]
    train_samples, val_samples, test_samples = _split_train_val_test(samples, labels, args.seed)

    output_dir = os.path.join(args.output_root, "MSP_IMPROV_preprocessed")
    _save_splits(output_dir, train_samples, val_samples, test_samples)

    logging.info(
        "MSP-IMPROV - Train: %d | Val: %d | Test: %d",
        len(train_samples),
        len(val_samples),
        len(test_samples),
    )
    logging.info("MSP-IMPROV - Saved preprocessed data to %s", output_dir)


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["iemocap", "msp-improv"],
        required=True,
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Root path of raw data (required for IEMOCAP/MSP-IMPROV preprocessing)",
    )
    parser.add_argument(
        "--audio_root",
        type=str,
        default=None,
        help="Optional root to resolve relative audio paths in metadata",
    )
    parser.add_argument(
        "--video_root",
        type=str,
        default=None,
        help="Optional root to resolve relative video/frame paths",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="metadata",
        help="Output root directory to store <DATASET>_preprocessed/train|val|test.pkl",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ignore_length", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    if args.dataset == "iemocap":
        preprocess_iemocap(args)
    elif args.dataset == "msp-improv":
        preprocess_msp_improv(args)
