import os

import torch
import torch.nn.functional as F
from torch import nn

from src import model as mm
from src.eval_metrics import eval_iemocap, eval_mosei_senti, eval_mosi, eval_sims


EVAL_MODALITY_TO_MISSING_MODE = {
    "av": 0,
    "tv": 1,
    "at": 2,
    "v": 3,
    "a": 4,
    "t": 5,
}


def get_underlying_model(model):
    if isinstance(model, nn.DataParallel):
        return model.module
    return model


def extract_prompted_sample(model, batch_X, missing_mod, hyp_params, sample_index=0):
    base_model = get_underlying_model(model)
    if not (
        hasattr(base_model, "get_complete_data")
        and hasattr(base_model, "missing_type_prompt")
        and hasattr(base_model, "get_proj_matrix")
    ):
        return None

    text, audio, vision = batch_X
    if hyp_params.use_cuda:
        with torch.cuda.device(0):
            text = text.cuda()
            audio = audio.cuda()
            vision = vision.cuda()
            missing_mod = missing_mod.cuda()

    idx = int(max(0, min(sample_index, text.size(0) - 1)))
    mode = int(missing_mod[idx].detach().cpu().item())

    with torch.no_grad():
        x_l = F.dropout(
            text.transpose(1, 2), p=base_model.embed_dropout, training=base_model.training
        )
        x_a = audio.transpose(1, 2)
        x_v = vision.transpose(1, 2)

        prompted_l, prompted_a, prompted_v = base_model.get_complete_data(
            x_l[idx], x_a[idx], x_v[idx], mode
        )
        base_model.get_proj_matrix()
        prompted_type = torch.matmul(base_model.missing_type_prompt, base_model.mp[mode])

    return {
        "missing_mode": mode,
        "prompted_text": prompted_l[0].transpose(0, 1).detach().cpu(),
        "prompted_audio": prompted_a[0].transpose(0, 1).detach().cpu(),
        "prompted_vision": prompted_v[0].transpose(0, 1).detach().cpu(),
        "prompted_missing_type": prompted_type.detach().cpu(),
    }


def print_prompted_sample(model, loader, hyp_params, title):
    if not getattr(hyp_params, "print_prompt_sample", False):
        return
    try:
        batch_X, _, missing_mod = next(iter(loader))
    except StopIteration:
        print(f"{title}: no sample available.")
        return

    sample = extract_prompted_sample(model, batch_X, missing_mod, hyp_params)
    if sample is None:
        print(f"{title}: current model does not expose prompt internals.")
        return

    print(f"{title}: missing_mode={sample['missing_mode']}")
    for key in ["prompted_text", "prompted_audio", "prompted_vision", "prompted_missing_type"]:
        tensor = sample[key]
        flat = tensor.reshape(-1)
        preview = flat[:8].tolist()
        print(f"  {key} shape={tuple(tensor.shape)} preview={preview}")
    if sample["prompted_missing_type"].abs().sum().item() == 0:
        print(
            "  WARNING: prompted_missing_type is all zeros. "
            "This usually means the checkpoint was trained with zero-initialized missing-type prompts."
        )

    output_path = getattr(hyp_params, "prompt_sample_out", None)
    if output_path is None:
        base_path = (
            getattr(hyp_params, "name", None)
            or getattr(hyp_params, "checkpoint", None)
            or "prompted_sample"
        )
        safe_title = "".join(
            ch.lower() if ch.isalnum() else "_" for ch in title
        ).strip("_")
        output_path = f"{base_path}.{safe_title}.pt"

    output_path = os.path.abspath(output_path)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save({"title": title, **sample}, output_path)
    print(f"Saved prompted sample to {output_path}")


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in [
            "model_state_dict",
            "state_dict",
            "model",
            "net",
            "network",
            "model_dict",
        ]:
            if key in checkpoint:
                return checkpoint[key]
        for value in checkpoint.values():
            if isinstance(value, dict) and any(
                torch.is_tensor(inner_value) for inner_value in value.values()
            ):
                return value
    return checkpoint


def _normalize_state_dict_keys(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    if not any(key.startswith("module.") for key in state_dict.keys()):
        return state_dict
    return {
        key[len("module."):] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def _build_model_from_state_dict(state_dict, hyp_params):
    state_keys = state_dict.keys()
    is_prompt_model = any(
        key.startswith("generative_prompt") or key.startswith("missing_type_prompt")
        for key in state_keys
    )
    model_cls = getattr(mm, "PromptModel" if is_prompt_model else "MULTModel")
    model = model_cls(hyp_params)
    overlap = set(model.state_dict().keys()) & set(state_keys)
    if not overlap:
        raise ValueError(
            "Could not find any model weights in the checkpoint. "
            "Expected a full model, a raw state_dict, or a checkpoint dict containing model_state_dict/state_dict."
        )
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print("Missing key(s) when loading checkpoint:", missing_keys)
    if unexpected_keys:
        print("Unexpected key(s) when loading checkpoint:", unexpected_keys)
    return model


def load_model(checkpoint_path, hyp_params):
    device = torch.device("cuda" if hyp_params.use_cuda else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint = _extract_state_dict(checkpoint)
    checkpoint = _normalize_state_dict_keys(checkpoint)

    if isinstance(checkpoint, nn.Module):
        model = checkpoint
    elif isinstance(checkpoint, dict):
        model = _build_model_from_state_dict(checkpoint, hyp_params)
    else:
        raise TypeError(
            f"Unsupported checkpoint type: {type(checkpoint)}. Expected nn.Module or state_dict-like dict."
        )

    if hyp_params.use_cuda:
        model = model.cuda()
    return model


def parse_eval_modalities(eval_modalities):
    if eval_modalities is None:
        return None

    requested = []
    for item in eval_modalities.split(","):
        modality = item.strip().lower()
        if not modality:
            continue
        if modality not in EVAL_MODALITY_TO_MISSING_MODE:
            valid = ", ".join(EVAL_MODALITY_TO_MISSING_MODE.keys())
            raise ValueError(
                f"Unsupported eval modality '{modality}'. Expected one of: {valid}."
            )
        if modality not in requested:
            requested.append(modality)
    return requested or None


def set_fixed_missing_mode(loader, missing_mode):
    if hasattr(loader, "dataset"):
        loader.dataset.fixed_missing_mode = missing_mode


def mask_modalities_by_missing_mode(text, audio, vision, missing_mod):
    if missing_mod is None:
        return text, audio, vision

    missing_mod = missing_mod.to(text.device).view(-1, 1, 1)
    miss_text = (missing_mod == 0) | (missing_mod == 3) | (missing_mod == 4)
    miss_audio = (missing_mod == 1) | (missing_mod == 3) | (missing_mod == 5)
    miss_vision = (missing_mod == 2) | (missing_mod == 4) | (missing_mod == 5)

    text = text.masked_fill(miss_text, 0.0)
    audio = audio.masked_fill(miss_audio, 0.0)
    vision = vision.masked_fill(miss_vision, 0.0)
    return text, audio, vision


def evaluate_split(model, criterion, hyp_params, valid_loader, test_loader, test=False):
    model.eval()
    base_model = get_underlying_model(model)
    supports_prompt_missing = hasattr(base_model, "get_complete_data") and hasattr(
        base_model, "missing_type_prompt"
    )

    loader = test_loader if test else valid_loader
    total_loss = 0.0
    results = []
    truths = []

    with torch.no_grad():
        for batch_X, batch_Y, missing_mod in loader:
            text, audio, vision = batch_X
            eval_attr = batch_Y.squeeze(dim=-1)

            if hyp_params.use_cuda:
                with torch.cuda.device(0):
                    text, audio, vision, eval_attr = (
                        text.cuda(),
                        audio.cuda(),
                        vision.cuda(),
                        eval_attr.cuda(),
                    )
                    if hyp_params.dataset == "iemocap":
                        eval_attr = eval_attr.long()

            if not supports_prompt_missing:
                text, audio, vision = mask_modalities_by_missing_mode(
                    text, audio, vision, missing_mod
                )

            batch_size = text.size(0)
            net = nn.DataParallel(model) if batch_size > 10 else model
            preds = net(text, audio, vision, missing_mod)
            if hyp_params.dataset == "iemocap":
                preds = preds.view(-1, 4)
                eval_attr = eval_attr.view(-1)

            total_loss += criterion(preds, eval_attr).item() * batch_size
            results.append(preds)
            truths.append(eval_attr)

    avg_loss = total_loss / (hyp_params.n_test if test else hyp_params.n_valid)
    results = torch.cat(results)
    truths = torch.cat(truths)
    return avg_loss, results, truths


def print_metrics(hyp_params, results, truths):
    if hyp_params.dataset == "mosei":
        eval_mosei_senti(results, truths, True)
    elif hyp_params.dataset == "mosi":
        eval_mosi(results, truths, True)
    elif hyp_params.dataset == "iemocap":
        eval_iemocap(results, truths)
    elif hyp_params.dataset == "sims":
        eval_sims(results, truths)


def evaluate_only(hyp_params, valid_loader, test_loader):
    checkpoint_path = hyp_params.checkpoint or hyp_params.name
    if checkpoint_path is None:
        raise ValueError(
            "`--eval_only` requires `--checkpoint` or `--name` to point to a saved model."
        )

    model = load_model(checkpoint_path, hyp_params)
    base_model = get_underlying_model(model)
    supports_prompt_missing = hasattr(base_model, "get_complete_data") and hasattr(
        base_model, "missing_type_prompt"
    )
    criterion = getattr(nn, hyp_params.criterion)()
    test = hyp_params.eval_split == "test"
    split_name = "test" if test else "validation"
    requested_modalities = parse_eval_modalities(hyp_params.eval_modalities)

    if requested_modalities is None:
        print("Eval-only uses complete samples (fixed_missing_mode=6); --drop_rate is ignored.")

    if requested_modalities is not None and not supports_prompt_missing:
        print(
            "Checkpoint is MULTModel (no prompt missing handler). "
            "Applying input masking by missing mode during evaluation."
        )

    if requested_modalities is None:
        try:
            set_fixed_missing_mode(valid_loader, 6)
            set_fixed_missing_mode(test_loader, 6)
            print(f"Evaluating checkpoint {checkpoint_path} on {split_name} split")
            eval_loss, results, truths = evaluate_split(
                model, criterion, hyp_params, valid_loader, test_loader, test=test
            )
            print(f"{split_name.title()} Loss: {eval_loss:.4f}")
            print_metrics(hyp_params, results, truths)
            print_prompted_sample(
                model,
                test_loader if test else valid_loader,
                hyp_params,
                f"{split_name.title()} prompted sample",
            )
            return
        finally:
            set_fixed_missing_mode(valid_loader, None)
            set_fixed_missing_mode(test_loader, None)

    try:
        for modality in requested_modalities:
            missing_mode = EVAL_MODALITY_TO_MISSING_MODE[modality]
            set_fixed_missing_mode(valid_loader, missing_mode)
            set_fixed_missing_mode(test_loader, missing_mode)

            print(
                f"Evaluating checkpoint {checkpoint_path} on {split_name} split with modality '{modality}'"
            )
            eval_loss, results, truths = evaluate_split(
                model, criterion, hyp_params, valid_loader, test_loader, test=test
            )
            print(f"{split_name.title()} Loss: {eval_loss:.4f}")
            print_metrics(hyp_params, results, truths)
            print_prompted_sample(
                model,
                test_loader if test else valid_loader,
                hyp_params,
                f"{split_name.title()} prompted sample ({modality})",
            )
    finally:
        set_fixed_missing_mode(valid_loader, None)
        set_fixed_missing_mode(test_loader, None)
