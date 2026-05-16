import argparse
import os
import pickle
import sys
import warnings

import librosa
import numpy as np
import soundfile as sf
import torch
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.feature_extract.config import (
    PKL_DIR,
    OUTPUT_DIR,
    device,
    TOKENIZER,
    AUDIO_PROCESSOR,
    VIDEO_PROCESSOR,
    TEXT_MODEL,
    AUDIO_MODEL,
    VIDEO_MODEL,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".flv"}

DATASET_ALIASES = {
    "iemocap": "iemocap",
    "IEMOCAP": "iemocap",
    "mosi": "mosi",
    "MOSI": "mosi",
    "cmu-mosi": "mosi",
    "CMU-MOSI": "mosi",
    "sims": "sims",
    "SIMS": "sims",
    "ch-sims": "sims",
    "CH-SIMS": "sims",
    "meld": "meld",
    "MELD": "meld",
    "msp-improv": "msp-improv",
    "MSP-IMPROV": "msp-improv",
    "msp_improv": "msp-improv",
    "MSP_IMPROV": "msp-improv",
}


def _first_non_empty(*values):
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and value.strip() == "":
            continue
        return value
    return None


def _resolve_path(raw_path, base_dir=None):
    if raw_path is None:
        return None

    path = str(raw_path)
    if os.path.isabs(path):
        return path

    if base_dir:
        joined = os.path.join(base_dir, path)
        if os.path.exists(joined):
            return joined

        joined_basename = os.path.join(base_dir, os.path.basename(path))
        if os.path.exists(joined_basename):
            return joined_basename

        return joined

    return path


def _to_float_or_none(value):
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _infer_video_from_audio_path(audio_path):
    if not audio_path:
        return None

    stem = os.path.splitext(str(audio_path))[0]
    candidates = [stem]
    for ext in VIDEO_EXTENSIONS:
        candidates.append(f"{stem}{ext}")
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def _infer_iemocap_dialog_video_from_audio(audio_path, sample_id=None):
    if not audio_path:
        return None

    basename = os.path.splitext(os.path.basename(str(audio_path)))[0]
    utt_id = sample_id if sample_id else basename
    if "_" not in utt_id:
        return None

    dialog_id = utt_id.rsplit("_", 1)[0]
    audio_str = str(audio_path)
    parts = audio_str.split(os.sep)
    session_name = None
    for token in parts:
        if token.startswith("Session") and token[7:].isdigit():
            session_name = token
            break
    if session_name is None:
        return None

    base = audio_str.split(session_name + os.sep)[0]
    if base.endswith(os.sep):
        root_prefix = base[:-1]
    else:
        root_prefix = base

    candidates = []
    roots = [
        os.path.join(root_prefix, session_name, "dialog", "avi", "DivX"),
        os.path.join(root_prefix, session_name, "dialog", "avi"),
    ]
    for root in roots:
        for ext in VIDEO_EXTENSIONS:
            candidates.append(os.path.join(root, f"{dialog_id}{ext}"))
        candidates.append(os.path.join(root, dialog_id))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def _infer_meld_video_from_sample(video_base, sample_id=None, audio_path=None):
    if not video_base:
        return None

    stems = []
    if sample_id:
        stems.append(str(sample_id))
    if audio_path:
        stems.append(os.path.splitext(os.path.basename(str(audio_path)))[0])
    stems = list(dict.fromkeys(stems))

    split_tokens = ["train", "dev", "val", "test"]
    candidates = []
    for stem in stems:
        for ext in VIDEO_EXTENSIONS:
            candidates.append(os.path.join(video_base, f"{stem}{ext}"))
            for split_name in split_tokens:
                candidates.append(os.path.join(video_base, split_name, f"{stem}{ext}"))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def _load_audio(audio_path):
    array, sr = sf.read(audio_path)

    if isinstance(array, np.ndarray):
        waveform = array.astype(np.float32)
    else:
        waveform = np.array(array, dtype=np.float32)

    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    if sr != 16000:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
        sr = 16000

    return np.ascontiguousarray(waveform), sr


def _extract_audio_embedding(waveform, sr, processor, model, device, source=None):
    try:
        inputs = processor(waveform, sampling_rate=sr, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            pooled, _ = model(inputs["input_values"])
        return pooled.squeeze().cpu()
    except Exception as exc:
        prefix = f"{source}: " if source else ""
        print(f"[ERROR] Audio failed: {prefix}({exc})")
        return None


def extract_audio_features(audio_path, processor, model, device):
    try:
        waveform, sr = _load_audio(audio_path)
    except Exception as exc:
        print(f"[ERROR] Audio failed: {audio_path} ({exc})")
        return None

    return _extract_audio_embedding(waveform, sr, processor, model, device, source=audio_path)


def extract_text_features(text, tokenizer, model, device):
    try:
        content = "" if text is None else str(text)
        inputs = tokenizer(
            content,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        ).to(device)
        with torch.no_grad():
            pooled, _ = model(inputs["input_ids"], inputs["attention_mask"])
        return pooled.squeeze().cpu()
    except Exception as exc:
        preview = "" if text is None else str(text)[:30]
        print(f"[ERROR] Text failed: {preview}... ({exc})")
        return None


def _sample_indices(total, max_frames):
    if total <= 0:
        return []
    if total <= max_frames:
        return list(range(total))
    return np.linspace(0, total - 1, num=max_frames, dtype=int).tolist()


def _load_image(path):
    image = Image.open(path).convert("RGB")
    return image


def _load_frames_from_video_file(video_path, max_frames, start_time=None, end_time=None):
    try:
        import cv2
    except Exception:
        print("[WARN] OpenCV is not available. Install opencv-python to read video files directly.")
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        return []

    use_segment = start_time is not None and end_time is not None and end_time > start_time
    indices = None

    if use_segment:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is not None and fps > 0:
            start_idx = int(max(0, np.floor(start_time * fps)))
            end_idx = int(min(frame_count, np.ceil(end_time * fps)))
            if end_idx > start_idx:
                seg_len = end_idx - start_idx
                n_take = min(max_frames, seg_len)
                if n_take > 0:
                    indices = np.linspace(start_idx, end_idx - 1, num=n_take, dtype=int).tolist()

    if indices is None:
        indices = _sample_indices(frame_count, max_frames)
    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        success, frame = cap.read()
        if not success:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(rgb))

    cap.release()
    return frames


def _load_video_frames(video_source, max_frames=8, start_time=None, end_time=None):
    if video_source is None:
        return []

    if isinstance(video_source, (list, tuple)):
        frame_paths = [str(path) for path in video_source if path]
        if not frame_paths:
            return []
        sampled = _sample_indices(len(frame_paths), max_frames)
        return [_load_image(frame_paths[idx]) for idx in sampled]

    source = str(video_source)
    if os.path.isdir(source):
        frame_paths = []
        for name in sorted(os.listdir(source)):
            ext = os.path.splitext(name)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                frame_paths.append(os.path.join(source, name))

        if not frame_paths:
            return []

        sampled = _sample_indices(len(frame_paths), max_frames)
        return [_load_image(frame_paths[idx]) for idx in sampled]

    if os.path.isfile(source):
        ext = os.path.splitext(source)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            return [_load_image(source)]
        if ext in VIDEO_EXTENSIONS:
            return _load_frames_from_video_file(
                source,
                max_frames=max_frames,
                start_time=start_time,
                end_time=end_time,
            )

    return []


def extract_video_features(
    video_source,
    processor,
    model,
    device,
    max_frames=8,
    start_time=None,
    end_time=None,
):
    try:
        frames = _load_video_frames(
            video_source,
            max_frames=max_frames,
            start_time=start_time,
            end_time=end_time,
        )
    except Exception as exc:
        print(f"[ERROR] Video loading failed: {video_source} ({exc})")
        return None

    if not frames:
        return None

    try:
        inputs = processor(images=frames, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        with torch.no_grad():
            pooled, _ = model(pixel_values)
        return pooled.mean(dim=0).cpu()
    except Exception as exc:
        print(f"[ERROR] Video encoding failed: {video_source} ({exc})")
        return None


def _parse_sample_item(item, pseudo=False):
    if isinstance(item, dict):
        sample_id = _first_non_empty(item.get("sample_id"), item.get("id"))
        audio_ref = _first_non_empty(item.get("filename"), item.get("audio_path"), item.get("path"))
        video_ref = _first_non_empty(
            item.get("video_path"),
            item.get("dialog_video_path"),
            item.get("video_frame_dir"),
            item.get("frame_dir"),
            item.get("video"),
        )
        text = _first_non_empty(item.get("text"), item.get("transcript"), item.get("utterance"), "")
        start_time = _to_float_or_none(
            _first_non_empty(item.get("start_time"), item.get("video_start"), item.get("clip_start"))
        )
        end_time = _to_float_or_none(
            _first_non_empty(item.get("end_time"), item.get("video_end"), item.get("clip_end"))
        )

        if pseudo:
            label = _first_non_empty(item.get("pseudo_label"), item.get("emotion"), item.get("label"))
        else:
            label = _first_non_empty(item.get("emotion"), item.get("label"), item.get("pseudo_label"))

        confidence = item.get("confidence")
        return sample_id, audio_ref, video_ref, text, label, confidence, start_time, end_time

    if isinstance(item, (tuple, list)):
        audio_ref = item[0] if len(item) > 0 else None
        text = item[1] if len(item) > 1 else ""
        label = item[2] if len(item) > 2 else None
        video_ref = item[3] if len(item) > 3 else None
        confidence = item[4] if len(item) > 4 else None
        start_time = _to_float_or_none(item[5]) if len(item) > 5 else None
        end_time = _to_float_or_none(item[6]) if len(item) > 6 else None
        sample_id = os.path.splitext(os.path.basename(str(audio_ref)))[0] if audio_ref else None
        return sample_id, audio_ref, video_ref, text, label, confidence, start_time, end_time

    raise TypeError(f"Unexpected item type: {type(item)}")


def _default_video_embedding():
    hidden_size = int(getattr(VIDEO_MODEL.clip_vision.config, "hidden_size", 768))
    return torch.zeros(hidden_size, dtype=torch.float32)


def process_single_sample(
    audio_path,
    text,
    label,
    video_path=None,
    video_start=None,
    video_end=None,
    sample_id=None,
    is_pseudo=False,
    confidence=None,
    skip_text=False,
    allow_missing_video=True,
    max_video_frames=8,
):
    audio_embed = extract_audio_features(audio_path, AUDIO_PROCESSOR, AUDIO_MODEL, device)
    if audio_embed is None:
        return None

    if skip_text:
        text_embed = torch.zeros_like(audio_embed)
    else:
        text_embed = extract_text_features(text, TOKENIZER, TEXT_MODEL, device)
        if text_embed is None:
            return None

    video_embed = extract_video_features(
        video_path,
        VIDEO_PROCESSOR,
        VIDEO_MODEL,
        device,
        max_frames=max_video_frames,
        start_time=video_start,
        end_time=video_end,
    )
    video_missing = video_embed is None

    if video_missing:
        if not allow_missing_video:
            return None
        video_embed = _default_video_embedding()

    resolved_sample_id = sample_id if sample_id else os.path.splitext(os.path.basename(audio_path))[0]

    return {
        "sample_id": resolved_sample_id,
        "text_embed": text_embed,
        "audio_embed": audio_embed,
        "video_embed": video_embed,
        "label": label,
        "is_pseudo": is_pseudo,
        "confidence": confidence,
        "missing_text": bool(skip_text),
        "missing_video": bool(video_missing),
        "raw_text": "" if text is None else str(text),
        "audio_path": audio_path,
        "video_path": video_path,
        "video_start": video_start,
        "video_end": video_end,
    }


def process_dataset(
    pkl_path,
    wav_base,
    video_base,
    dataset_key,
    output_path,
    pseudo=False,
    skip_text=False,
    allow_missing_video=True,
    max_video_frames=8,
):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    processed_samples = []
    print(f"Processing {len(data)} samples from {pkl_path}")

    for item in tqdm(data, desc=f"Processing {os.path.basename(pkl_path)}"):
        try:
            (
                sample_id,
                audio_ref,
                video_ref,
                text,
                label,
                confidence,
                video_start,
                video_end,
            ) = _parse_sample_item(item, pseudo=pseudo)
        except Exception as exc:
            print(f"[SKIP] Invalid sample schema: {exc}")
            continue

        if audio_ref is None:
            print("[SKIP] Missing audio reference in one sample")
            continue

        audio_path = _resolve_path(audio_ref, base_dir=wav_base)
        video_path = _resolve_path(video_ref, base_dir=video_base)
        if video_path is None and dataset_key == "meld":
            video_path = _infer_meld_video_from_sample(
                video_base=video_base,
                sample_id=sample_id,
                audio_path=audio_path,
            )
        if video_path is None:
            video_path = _infer_video_from_audio_path(audio_path)
        if video_path is None and dataset_key == "iemocap":
            video_path = _infer_iemocap_dialog_video_from_audio(audio_path, sample_id=sample_id)
        if video_path is not None and (video_start is None or video_end is None):
            # Only dialog videos need time segmenting. Utterance-level clips keep full range.
            sample_utt = sample_id if sample_id else os.path.splitext(os.path.basename(str(audio_path)))[0]
            if sample_utt and "_" in sample_utt:
                dialog_id = sample_utt.rsplit("_", 1)[0]
                video_name = os.path.splitext(os.path.basename(str(video_path)))[0]
                if video_name == dialog_id:
                    if isinstance(item, dict):
                        video_start = _to_float_or_none(item.get("start_time"))
                        video_end = _to_float_or_none(item.get("end_time"))

        sample = process_single_sample(
            audio_path=audio_path,
            text=text,
            label=label,
            video_path=video_path,
            video_start=video_start,
            video_end=video_end,
            sample_id=sample_id,
            is_pseudo=pseudo,
            confidence=confidence,
            skip_text=skip_text,
            allow_missing_video=allow_missing_video,
            max_video_frames=max_video_frames,
        )
        if sample is not None:
            processed_samples.append(sample)
        else:
            print(f"[SKIP] Failed to process: {audio_path}")

    with open(output_path, "wb") as f:
        pickle.dump(processed_samples, f)

    print(f"Saved processed data to: {output_path}")
    print(f"Total processed samples: {len(processed_samples)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=sorted(DATASET_ALIASES.keys()),
        help="Dataset to process",
    )
    parser.add_argument("--pseudo", action="store_true", help="Process train split as pseudo-labeled")
    parser.add_argument(
        "--wav_base",
        type=str,
        default=None,
        help="Optional root directory containing waveform files",
    )
    parser.add_argument(
        "--video_base",
        type=str,
        default=None,
        help="Optional root directory containing video files or frame directories",
    )
    parser.add_argument(
        "--max_video_frames",
        type=int,
        default=8,
        help="Maximum number of frames sampled for CLIP video encoding",
    )
    parser.add_argument(
        "--strict_video",
        action="store_true",
        help="Drop samples without available video input",
    )
    parser.add_argument(
        "--force_text",
        action="store_true",
        help="Force running text encoder even if dataset default is to skip text",
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Starting feature extraction...")
    print(f"Using device: {device}")

    dataset_key = DATASET_ALIASES[args.dataset]
    dataset_cfg = {
        "iemocap": {
            "preprocessed_dir": "IEMOCAP_preprocessed",
            "output_prefix": "IEMOCAP_BERT_LARGE_WavLM_CLIP",
            "skip_text": False,
        },
        "mosi": {
            "preprocessed_dir": "MOSI_preprocessed",
            "output_prefix": "CMU_MOSI_BERT_LARGE_WavLM_CLIP",
            "skip_text": False,
        },
        "sims": {
            "preprocessed_dir": "SIMS_preprocessed",
            "output_prefix": "CH_SIMS_BERT_LARGE_WavLM_CLIP",
            "skip_text": False,
        },
        "meld": {
            "preprocessed_dir": "MELD_preprocessed",
            "output_prefix": "MELD_BERT_LARGE_WavLM_CLIP",
            "skip_text": False,
        },
        "msp-improv": {
            "preprocessed_dir": "MSP_IMPROV_preprocessed",
            "output_prefix": "MSP_IMPROV_BERT_LARGE_WavLM_CLIP",
            "skip_text": False,
        },
    }

    cfg = dataset_cfg[dataset_key]
    skip_text = cfg["skip_text"] and (not args.force_text)

    splits = ["train", "val", "test"]
    for split_name in splits:
        pkl_file = os.path.join(PKL_DIR, cfg["preprocessed_dir"], f"{split_name}.pkl")
        output_file = os.path.join(OUTPUT_DIR, f"{cfg['output_prefix']}_{split_name}.pkl")

        print(f"\n{'=' * 50}")
        print(f"Processing {split_name} split: {pkl_file}")
        print(f"{'=' * 50}")

        process_dataset(
            pkl_path=pkl_file,
            wav_base=args.wav_base,
            video_base=args.video_base,
            dataset_key=dataset_key,
            output_path=output_file,
            pseudo=args.pseudo and split_name == "train",
            skip_text=skip_text,
            allow_missing_video=not args.strict_video,
            max_video_frames=args.max_video_frames,
        )


if __name__ == "__main__":
    main()
