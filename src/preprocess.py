import argparse
import csv
import glob
import json
import logging
import os
import pickle
import random
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

VIDEO_FILE_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".flv")
AUDIO_FILE_EXTS = (".wav", ".flac", ".mp3", ".m4a")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _first_non_empty(*values):
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and value.strip() == "":
            continue
        return value
    return None


def _item_get(item, *keys):
    if not isinstance(item, dict):
        return None
    lowered = {str(key).lower(): value for key, value in item.items()}
    values = []
    for key in keys:
        values.append(item.get(key))
        values.append(lowered.get(str(key).lower()))
    return _first_non_empty(*values)


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


def _resolve_path(raw_path, base_root=None):
    if raw_path is None:
        return None

    path = str(raw_path)
    if os.path.isabs(path):
        return path

    if base_root:
        return os.path.join(base_root, path)

    return path


def _normalize_split_name(name):
    if name is None:
        return None
    value = str(name).strip().lower()
    if value in {"train", "trn", "training"}:
        return "train"
    if value in {"valid", "val", "dev", "validation"}:
        return "val"
    if value in {"test", "tst", "testing"}:
        return "test"
    return None


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


def _infer_meld_media_path(root, stem, split_name=None, exts=VIDEO_FILE_EXTS):
    if not root or not stem:
        return None

    split_aliases = []
    if split_name is not None:
        split_aliases.append(split_name)
        if split_name == "val":
            split_aliases.append("dev")

    candidates = []
    for ext in exts:
        candidates.append(os.path.join(root, f"{stem}{ext}"))
    for split_token in split_aliases:
        for ext in exts:
            candidates.append(os.path.join(root, split_token, f"{stem}{ext}"))
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


def _load_metadata_records(metadata_path):
    if os.path.isdir(metadata_path):
        records = []
        patterns = ["*.csv", "*.jsonl", "*.json", "*.pkl", "*.pickle"]
        all_files = []
        for pattern in patterns:
            all_files.extend(glob.glob(os.path.join(metadata_path, pattern)))
        for file_path in sorted(set(all_files)):
            inferred_split = _normalize_split_name(os.path.basename(file_path))
            file_records = _load_metadata_records(file_path)
            for item in file_records:
                if not isinstance(item, dict):
                    records.append(item)
                    continue
                row = dict(item)
                split_name = _normalize_split_name(_item_get(row, "split"))
                if split_name is None and inferred_split is not None:
                    row["split"] = inferred_split
                records.append(row)
        return records

    ext = os.path.splitext(metadata_path)[1].lower()

    if ext == ".csv":
        records = []
        with open(metadata_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(dict(row))
        return records

    if ext == ".jsonl":
        records = []
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records

    if ext == ".json":
        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            records = []
            for split_name, split_items in data.items():
                split_norm = _normalize_split_name(split_name)
                if split_norm is None:
                    continue
                for item in split_items:
                    row = dict(item)
                    row.setdefault("split", split_norm)
                    records.append(row)
            return records
        raise ValueError("Unsupported JSON schema. Expected list or split->list dict.")

    if ext in {".pkl", ".pickle"}:
        with open(metadata_path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            records = []
            for split_name, split_items in data.items():
                split_norm = _normalize_split_name(split_name)
                if split_norm is None:
                    continue
                for item in split_items:
                    row = dict(item)
                    row.setdefault("split", split_norm)
                    records.append(row)
            return records
        raise ValueError("Unsupported pickle schema. Expected list or split->list dict.")

    raise ValueError("Unsupported metadata extension. Use .csv, .jsonl, .json, or .pkl")


def _preprocess_from_metadata(args, dataset_name, output_subdir):
    if not args.metadata_path:
        raise ValueError(f"--metadata_path is required for {dataset_name} preprocessing.")

    records = _load_metadata_records(args.metadata_path)
    if not records:
        raise ValueError(f"No metadata records found in {args.metadata_path}")

    grouped = {"train": [], "val": [], "test": []}

    for idx, item in enumerate(records):
        if not isinstance(item, dict):
            logging.warning("Skipped metadata record %d because it is not a dict", idx)
            continue

        split_name = _normalize_split_name(_item_get(item, "split"))
        if split_name is None:
            split_name = _normalize_split_name(os.path.basename(args.metadata_path))
        if split_name is None:
            logging.warning("Skipped metadata record %d because split is missing/invalid", idx)
            continue

        sample_id = _first_non_empty(
            _item_get(item, "sample_id", "id", "utt_id", "utterance_id"),
            f"{dataset_name}_{idx}",
        )

        audio_path = _first_non_empty(
            _item_get(item, "audio_path", "wav_path", "path", "filename", "audio", "wav"),
        )

        video_path = _first_non_empty(
            _item_get(item, "video_path", "video_frame_dir", "frame_dir", "video", "video_file"),
        )
        text = _first_non_empty(
            _item_get(item, "text", "transcript", "utterance", "Utterance"),
            "",
        )
        label = _first_non_empty(
            _item_get(item, "emotion", "label", "sentiment", "regression_label", "Emotion"),
        )

        audio_path = _resolve_path(audio_path, base_root=args.audio_root)
        video_path = _resolve_path(video_path, base_root=args.video_root)

        if dataset_name == "MELD":
            dialog_id = _item_get(item, "dialogue_id", "dialog_id", "Dialogue_ID")
            utterance_id = _item_get(item, "utterance_id", "utt_id", "Utterance_ID")
            if dialog_id is not None and utterance_id is not None:
                meld_stem = f"dia{dialog_id}_utt{utterance_id}"
                if audio_path is None:
                    audio_path = _infer_meld_media_path(
                        args.audio_root,
                        meld_stem,
                        split_name=split_name,
                        exts=AUDIO_FILE_EXTS,
                    )
                if video_path is None:
                    video_path = _infer_meld_media_path(
                        args.video_root,
                        meld_stem,
                        split_name=split_name,
                        exts=VIDEO_FILE_EXTS,
                    )
                if sample_id == f"{dataset_name}_{idx}":
                    sample_id = meld_stem

        if audio_path is None:
            logging.warning("Skipped metadata record %d because audio path is missing", idx)
            continue

        if video_path is None:
            video_path = _infer_video_from_audio_path(audio_path)

        sample = _build_sample(
            sample_id=sample_id,
            audio_path=audio_path,
            text=text,
            label=label,
            video_path=video_path,
        )
        grouped[split_name].append(sample)

    if not grouped["train"] and not grouped["val"] and not grouped["test"]:
        raise ValueError("No valid metadata records after normalization.")

    if not grouped["val"] or not grouped["test"]:
        merged = grouped["train"] + grouped["val"] + grouped["test"]
        labels = [sample["emotion"] for sample in merged]
        train_samples, val_samples, test_samples = _split_train_val_test(merged, labels, args.seed)
    else:
        train_samples = grouped["train"]
        val_samples = grouped["val"]
        test_samples = grouped["test"]

    output_dir = os.path.join(args.output_root, output_subdir)
    _save_splits(output_dir, train_samples, val_samples, test_samples)

    logging.info(
        "%s - Train: %d | Val: %d | Test: %d",
        dataset_name,
        len(train_samples),
        len(val_samples),
        len(test_samples),
    )
    logging.info("%s - Saved preprocessed data to %s", dataset_name, output_dir)


def preprocess_mosi(args):
    _preprocess_from_metadata(args, dataset_name="CMU-MOSI", output_subdir="MOSI_preprocessed")


def preprocess_sims(args):
    _preprocess_from_metadata(args, dataset_name="CH-SIMS", output_subdir="SIMS_preprocessed")


def preprocess_meld(args):
    _preprocess_from_metadata(args, dataset_name="MELD", output_subdir="MELD_preprocessed")


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["iemocap", "mosi", "sims", "meld"],
        required=True,
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Root path of raw IEMOCAP data (required when --dataset iemocap)",
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        default=None,
        help="Path to metadata file or directory (.csv/.jsonl/.json/.pkl) for MOSI/SIMS/MELD preprocessing",
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
    elif args.dataset == "mosi":
        preprocess_mosi(args)
    elif args.dataset == "sims":
        preprocess_sims(args)
    elif args.dataset == "meld":
        preprocess_meld(args)
