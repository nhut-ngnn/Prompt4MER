import glob
import os
import pickle
import random

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


MELD_LABEL_MAP = {
    "neutral": 0,
    "joy": 1,
    "surprise": 2,
    "anger": 3,
    "sadness": 4,
    "disgust": 5,
    "fear": 6,
}


class MELDFeatureData(Dataset):
    SPLIT_MAP = {
        "train": "train",
        "valid": "val",
        "test": "test",
    }

    def __init__(
        self,
        data_path,
        split_type="train",
        drop_rate=0.6,
        full_data=False,
        l_type=None,
        a_type=None,
        v_type=None,
    ):
        super(MELDFeatureData, self).__init__()
        split_name = self.SPLIT_MAP.get(split_type, split_type)
        self.drop_rate = drop_rate
        self.full_data = full_data
        self.fixed_missing_mode = None

        pkl_path = self._resolve_pkl_path(
            data_path=data_path,
            split_name=split_name,
            l_type=l_type,
            a_type=a_type,
            v_type=v_type,
        )
        with open(pkl_path, "rb") as f:
            raw_samples = pickle.load(f)

        self.samples = []
        for item in raw_samples:
            if not isinstance(item, dict):
                continue
            l_feat = self._to_feature_tensor(item.get("text_embed"))
            a_feat = self._to_feature_tensor(item.get("audio_embed"))
            v_feat = self._to_feature_tensor(item.get("video_embed"))
            label = self._to_label(item.get("label", item.get("emotion")))
            if l_feat is None or a_feat is None or v_feat is None or label is None:
                continue
            self.samples.append((l_feat, a_feat, v_feat, label))

        if not self.samples:
            raise ValueError(f"No valid samples loaded from {pkl_path}")

        self.orig_dims = (
            self.samples[0][0].shape[1],
            self.samples[0][1].shape[1],
            self.samples[0][2].shape[1],
        )
        self.seq_len = self._infer_seq_len()

    def _resolve_pkl_path(self, data_path, split_name, l_type=None, a_type=None, v_type=None):
        preferred_tokens = []
        for token in [l_type, a_type, v_type]:
            if token is not None:
                preferred_tokens.append(str(token).lower())

        primary_pattern = os.path.join(data_path, f"MELD_*_{split_name}.pkl")
        fallback_pattern = os.path.join(data_path, f"*_{split_name}.pkl")
        candidates = sorted(set(glob.glob(primary_pattern) + glob.glob(fallback_pattern)))

        if not candidates:
            raise FileNotFoundError(
                f"Cannot find extracted MELD feature file for split '{split_name}' under {data_path}"
            )
        if len(candidates) == 1:
            return candidates[0]

        if preferred_tokens:
            scored = []
            for path in candidates:
                name = os.path.basename(path).lower()
                score = sum(int(token in name) for token in preferred_tokens)
                scored.append((score, path))
            scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
            if scored[0][0] > 0:
                return scored[0][1]

        return candidates[0]

    def _to_feature_tensor(self, value):
        if value is None:
            return None

        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().float()
        else:
            tensor = torch.tensor(np.asarray(value), dtype=torch.float32)

        if tensor.ndim == 0:
            tensor = tensor.view(1, 1)
        elif tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim > 2:
            first_dim = tensor.shape[0]
            tensor = tensor.view(first_dim, -1)
        return tensor

    def _to_label(self, value):
        if value is None:
            return None

        if isinstance(value, str):
            label = value.strip().lower()
            if label in MELD_LABEL_MAP:
                return MELD_LABEL_MAP[label]
            try:
                return int(label)
            except Exception:
                return None

        if isinstance(value, torch.Tensor):
            arr = value.detach().cpu().numpy()
        else:
            arr = np.asarray(value)

        if arr.ndim == 0:
            scalar = arr.item()
            if isinstance(scalar, str):
                label = scalar.strip().lower()
                if label in MELD_LABEL_MAP:
                    return MELD_LABEL_MAP[label]
                try:
                    return int(label)
                except Exception:
                    return None
            return int(scalar)

        flat = arr.reshape(-1)
        if flat.size == 1:
            scalar = flat.item()
            if isinstance(scalar, str):
                label = scalar.strip().lower()
                if label in MELD_LABEL_MAP:
                    return MELD_LABEL_MAP[label]
                try:
                    return int(label)
                except Exception:
                    return None
            return int(scalar)

        return int(np.argmax(flat))

    def _infer_seq_len(self):
        l_len = max(sample[0].shape[0] for sample in self.samples)
        a_len = max(sample[1].shape[0] for sample in self.samples)
        v_len = max(sample[2].shape[0] for sample in self.samples)
        return (l_len, a_len, v_len)

    def get_dim(self):
        return self.orig_dims

    def get_seq_len(self):
        return self.seq_len

    def get_missing_mode(self):
        if self.fixed_missing_mode is not None:
            return self.fixed_missing_mode
        if self.full_data:
            return 6
        if random.random() < self.drop_rate:
            return random.randint(0, 5)
        return 6

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        l_feat, a_feat, v_feat, label = self.samples[index]
        x = (l_feat, a_feat, v_feat)
        missing_code = self.get_missing_mode()
        return x, int(label), missing_code

    def collate_fn(self, batch):
        l_batch = [sample[0][0] for sample in batch]
        a_batch = [sample[0][1] for sample in batch]
        v_batch = [sample[0][2] for sample in batch]

        l_batch = pad_sequence(l_batch, batch_first=True, padding_value=0.0)
        a_batch = pad_sequence(a_batch, batch_first=True, padding_value=0.0)
        v_batch = pad_sequence(v_batch, batch_first=True, padding_value=0.0)

        labels = torch.tensor([int(sample[1]) for sample in batch], dtype=torch.long)
        missing_code = torch.tensor([sample[2] for sample in batch])
        x = (l_batch, a_batch, v_batch)
        return x, labels, missing_code
