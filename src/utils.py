import torch
from torch.utils.data import DataLoader

from src.iemocap_feature_data import IEMOCAPFeatureData
from src.mosidata import MOSIData
from src.simsdata import SIMSData


def get_data(args, split="train", full_data=False):
    if args.dataset == "iemocap":
        data = IEMOCAPFeatureData(
            data_path=args.data_path,
            split_type=split,
            drop_rate=args.drop_rate,
            full_data=full_data,
            l_type=getattr(args, "l_type", None),
            a_type=getattr(args, "a_type", None),
            v_type=getattr(args, "v_type", None),
        )
    elif args.dataset == "meld":
        try:
            from src.meld_feature_data import MELDFeatureData
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "MELD dataset support requires src/meld_feature_data.py. "
                "That file is not present in the current worktree."
            ) from exc
        data = MELDFeatureData(
            data_path=args.data_path,
            split_type=split,
            drop_rate=args.drop_rate,
            full_data=full_data,
            l_type=getattr(args, "l_type", None),
            a_type=getattr(args, "a_type", None),
            v_type=getattr(args, "v_type", None),
        )
    elif args.dataset == "mosi" or args.dataset == "mosei":
        data = MOSIData(
            args.data_path, split, drop_rate=args.drop_rate, full_data=full_data
        )
    elif args.dataset == "sims":
        data = SIMSData(
            args.data_path, split, drop_rate=args.drop_rate, full_data=full_data
        )
    return data


def get_loader(args):
    dataloaders = {}
    n_nums = []
    orig_dims = None
    seq_len = None
    if args.dataset in {"iemocap", "meld"}:
        for split in ["train", "valid", "test"]:
            dataset = get_data(args, split, full_data=(split != "train"))
            dataloaders[split] = DataLoader(
                dataset,
                batch_size=args.batch_size,
                drop_last=False,
                collate_fn=dataset.collate_fn,
            )
            current_dims = dataset.get_dim()
            current_seq_len = dataset.get_seq_len()
            orig_dims = current_dims if orig_dims is None else orig_dims
            n_nums.append(len(dataset))
            seq_len = (
                current_seq_len
                if seq_len is None
                else tuple(max(a, b) for a, b in zip(seq_len, current_seq_len))
            )
    else:
        for split in ["train", "valid", "test"]:
            dataset = get_data(args, split, full_data=(split != "train"))
            dataloaders[split] = DataLoader(dataset, batch_size=args.batch_size)
            current_dims = dataset.get_dim()
            current_seq_len = dataset.get_seq_len()
            orig_dims = current_dims if orig_dims is None else orig_dims
            n_nums.append(len(dataset))
            seq_len = (
                current_seq_len
                if seq_len is None
                else tuple(max(a, b) for a, b in zip(seq_len, current_seq_len))
            )
    return dataloaders, orig_dims, n_nums, seq_len


def transfer_model(new_model, pretrained):
    def _load_checkpoint(path):
        # The pretrained checkpoint is a local artifact under user control.
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")

    loaded = _load_checkpoint(pretrained)
    if isinstance(loaded, dict):
        if "model_state_dict" in loaded and isinstance(loaded["model_state_dict"], dict):
            pretrain_dict = loaded["model_state_dict"]
        elif "state_dict" in loaded and isinstance(loaded["state_dict"], dict):
            pretrain_dict = loaded["state_dict"]
        else:
            pretrain_dict = loaded
    else:
        pretrain_dict = loaded.state_dict()

    skip_keys = {
        "proj_l.weight",
        "proj_a.weight",
        "proj_v.weight",
        "out_layer.weight",
        "out_layer.bias",
        "classifer.weight",
        "classifer.bias",
    }
    obsolete_prefixes = (
        "text_gate.1.",
        "audio_gate.1.",
        "vision_gate.1.",
    )

    new_dict = new_model.state_dict()
    transferred = {}
    for k, v in pretrain_dict.items():
        if k.startswith(obsolete_prefixes):
            print("Skipped obsolete pretrained key :{}".format(k))
            continue
        if k not in new_dict:
            print("Unexpected pretrained key :{}".format(k))
            continue
        if k in skip_keys:
            print("Skipped pretrained key :{}".format(k))
            continue
        if new_dict[k].shape != v.shape:
            print(
                "Skipped pretrained key due to shape mismatch :{} {} -> {}".format(
                    k, tuple(v.shape), tuple(new_dict[k].shape)
                )
            )
            continue
        transferred[k] = v

    new_dict.update(transferred)
    new_model.load_state_dict(new_dict)
    for _, param in new_model.named_parameters():
        param.requires_grad = True
    return new_model
