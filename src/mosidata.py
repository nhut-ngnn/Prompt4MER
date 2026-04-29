import random
import os
import numpy as np
from torch.utils.data.dataset import Dataset
import pickle
import torch


class MOSIData(Dataset):
    DEFAULT_FILENAMES = ("mosi_data.pkl", "mosi_raw.pkl")

    def __init__(
        self, dataset_path, split_type="train", drop_rate=0.6, full_data=False
    ):
        super(MOSIData, self).__init__()
        dataset_path = self._resolve_dataset_path(dataset_path)
        with open(dataset_path, "rb") as f:
            dataset = pickle.load(f)
        if split_type not in dataset:
            raise KeyError(
                f"Split '{split_type}' was not found in {dataset_path}. "
                f"Available splits: {sorted(dataset.keys())}"
            )
        for key in ["text", "audio", "vision", "labels"]:
            if key not in dataset[split_type]:
                raise KeyError(
                    f"Required key '{key}' was not found in split '{split_type}' "
                    f"from {dataset_path}"
                )

        self.vision = (
            torch.tensor(dataset[split_type]["vision"].astype(np.float32))
            .cpu()
            .detach()
        )
        self.text = (
            torch.tensor(dataset[split_type]["text"].astype(np.float32)).cpu().detach()
        )
        self.audio = dataset[split_type]["audio"].astype(np.float32)
        self.audio[self.audio == -np.inf] = 0
        self.audio = torch.tensor(self.audio).cpu().detach()
        self.labels = (
            torch.tensor(dataset[split_type]["labels"].astype(np.float32))
            .cpu()
            .detach()
        )

        self.drop_rate = drop_rate
        self.full_data = full_data
        self.fixed_missing_mode = None
        self.n_modalities = 3  # vision/ text/ audio

    @classmethod
    def _resolve_dataset_path(cls, dataset_path):
        if dataset_path:
            dataset_path = os.path.expanduser(dataset_path)
            if os.path.isdir(dataset_path):
                for filename in cls.DEFAULT_FILENAMES:
                    candidate = os.path.join(dataset_path, filename)
                    if os.path.isfile(candidate):
                        return candidate
                pkl_candidates = sorted(
                    os.path.join(dataset_path, name)
                    for name in os.listdir(dataset_path)
                    if name.endswith(".pkl")
                )
                if pkl_candidates:
                    return pkl_candidates[0]
                raise FileNotFoundError(
                    f"No MOSI pickle file found under directory: {dataset_path}"
                )
            if os.path.isfile(dataset_path):
                return dataset_path
            raise FileNotFoundError(f"MOSI data path does not exist: {dataset_path}")

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        workspace_root = os.path.dirname(project_root)
        search_dirs = [
            os.path.join(workspace_root, "mosi"),
            os.path.join(project_root, "mosi"),
            os.path.join(os.getcwd(), "mosi"),
            os.getcwd(),
        ]
        for directory in search_dirs:
            for filename in cls.DEFAULT_FILENAMES:
                candidate = os.path.join(directory, filename)
                if os.path.isfile(candidate):
                    return candidate

        searched = ", ".join(search_dirs)
        raise FileNotFoundError(
            "MOSI data path was not provided and no default pickle was found. "
            f"Searched these directories: {searched}"
        )

    def get_n_modalities(self):
        return self.n_modalities

    def get_seq_len(self):
        return self.text.shape[1], self.audio.shape[1], self.vision.shape[1]

    def get_dim(self):
        return self.text.shape[2], self.audio.shape[2], self.vision.shape[2]

    def get_lbl_info(self):
        return self.labels.shape[1], self.labels.shape[2]

    def get_missing_mode(self):
        if self.fixed_missing_mode is not None:
            return self.fixed_missing_mode
        if self.full_data:
            return 6
        if random.random() < self.drop_rate:
            return random.randint(0, 5)
        else:
            return 6

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        X = (self.text[index], self.audio[index], self.vision[index])
        Y = self.labels[index]
        return X, Y, self.get_missing_mode()
