import torch
from torch import nn
from src import model as mm
from src.utils import *
import torch.optim as optim
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau


from src.eval_metrics import *

EVAL_MODALITY_TO_MISSING_MODE = {
    "av": 0,
    "tv": 1,
    "at": 2,
    "v": 3,
    "a": 4,
    "t": 5,
}


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


def _load_model(checkpoint_path, hyp_params):
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


def _parse_eval_modalities(eval_modalities):
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


def _set_fixed_missing_mode(loader, missing_mode):
    if hasattr(loader, "dataset"):
        loader.dataset.fixed_missing_mode = missing_mode


def _evaluate(model, criterion, hyp_params, valid_loader, test_loader, test=False):
    model.eval()
    loader = test_loader if test else valid_loader
    total_loss = 0.0
    results = []
    truths = []

    with torch.no_grad():
        for i_batch, (batch_X, batch_Y, missing_mod) in enumerate(loader):
            text, audio, vision = batch_X
            eval_attr = batch_Y.squeeze(dim=-1)  # if num of labels is 1

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


def _print_metrics(hyp_params, results, truths):
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

    model = _load_model(checkpoint_path, hyp_params)
    criterion = getattr(nn, hyp_params.criterion)()
    test = hyp_params.eval_split == "test"
    split_name = "test" if test else "validation"
    requested_modalities = _parse_eval_modalities(hyp_params.eval_modalities)

    if requested_modalities is None:
        print(f"Evaluating checkpoint {checkpoint_path} on {split_name} split")
        eval_loss, results, truths = _evaluate(
            model, criterion, hyp_params, valid_loader, test_loader, test=test
        )
        print(f"{split_name.title()} Loss: {eval_loss:.4f}")
        _print_metrics(hyp_params, results, truths)
        return

    try:
        for modality in requested_modalities:
            missing_mode = EVAL_MODALITY_TO_MISSING_MODE[modality]
            _set_fixed_missing_mode(valid_loader, missing_mode)
            _set_fixed_missing_mode(test_loader, missing_mode)

            print(
                f"Evaluating checkpoint {checkpoint_path} on {split_name} split with modality '{modality}'"
            )
            eval_loss, results, truths = _evaluate(
                model, criterion, hyp_params, valid_loader, test_loader, test=test
            )
            print(f"{split_name.title()} Loss: {eval_loss:.4f}")
            _print_metrics(hyp_params, results, truths)
    finally:
        _set_fixed_missing_mode(valid_loader, None)
        _set_fixed_missing_mode(test_loader, None)


def initiate(hyp_params, train_loader, valid_loader, test_loader):
    if hyp_params.eval_only:
        return evaluate_only(hyp_params, valid_loader, test_loader)

    if hyp_params.pretrained_model is not None:
        model = getattr(mm, "PromptModel")(hyp_params)
        model = transfer_model(model, hyp_params.pretrained_model)
    else:
        model = getattr(mm, "MULTModel")(hyp_params)

    if hyp_params.use_cuda:
        model = model.cuda()

    optimizer = getattr(optim, hyp_params.optim)(model.parameters(), lr=hyp_params.lr)
    criterion = getattr(nn, hyp_params.criterion)()

    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", patience=hyp_params.when, factor=0.1, verbose=True
    )
    settings = {
        "model": model,
        "optimizer": optimizer,
        "criterion": criterion,
        "scheduler": scheduler,
    }
    return train_model(settings, hyp_params, train_loader, valid_loader, test_loader)


def train_model(settings, hyp_params, train_loader, valid_loader, test_loader):
    model = settings["model"]
    optimizer = settings["optimizer"]
    criterion = settings["criterion"]
    scheduler = settings["scheduler"]

    def train(model, optimizer, criterion):
        model.train()
        num_batches = hyp_params.n_train // hyp_params.batch_size
        proc_loss, proc_size = 0, 0
        start_time = time.time()
        for i_batch, (batch_X, batch_Y, missing_mod) in enumerate(train_loader):
            text, audio, vision = batch_X
            eval_attr = batch_Y.squeeze(-1)
            model.zero_grad()

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

            batch_size = text.size(0)
            net = nn.DataParallel(model) if batch_size > 10 else model
            preds = net(text, audio, vision, missing_mod)

            if hyp_params.dataset == "iemocap":
                preds = preds.view(-1, 4)
                eval_attr = eval_attr.view(-1)
            raw_loss = criterion(preds, eval_attr)
            raw_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
            optimizer.step()

            proc_loss += raw_loss.item() * batch_size
            proc_size += batch_size
            if i_batch % hyp_params.log_interval == 0 and i_batch > 0:
                avg_loss = proc_loss / proc_size
                elapsed_time = time.time() - start_time
                print(
                    "Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Train Loss {:5.4f}".format(
                        epoch,
                        i_batch,
                        num_batches,
                        elapsed_time * 1000 / hyp_params.log_interval,
                        avg_loss,
                    )
                )
                proc_loss, proc_size = 0, 0
                start_time = time.time()

    best_valid = 1e8
    for epoch in range(1, hyp_params.num_epochs + 1):
        start = time.time()
        train(model, optimizer, criterion)
        val_loss, _, _ = _evaluate(
            model, criterion, hyp_params, valid_loader, test_loader, test=False
        )
        test_loss, _, _ = _evaluate(
            model, criterion, hyp_params, valid_loader, test_loader, test=True
        )

        end = time.time()
        duration = end - start
        scheduler.step(val_loss)

        print("-" * 50)
        print(
            "Epoch {:2d} | Time {:5.4f} sec | Valid Loss {:5.4f} | Test Loss {:5.4f}".format(
                epoch, duration, val_loss, test_loss
            )
        )
        print("-" * 50)

        if val_loss < best_valid:
            print(f"Saved model at {hyp_params.name}")
            torch.save(model, hyp_params.name)
            best_valid = val_loss

    model = _load_model(hyp_params.name, hyp_params)
    best_valid_loss, valid_results, valid_truths = _evaluate(
        model, criterion, hyp_params, valid_loader, test_loader, test=False
    )
    best_test_loss, test_results, test_truths = _evaluate(
        model, criterion, hyp_params, valid_loader, test_loader, test=True
    )

    print("=" * 50)
    print(
        "Best checkpoint summary | Valid Loss {:5.4f} | Test Loss {:5.4f}".format(
            best_valid_loss, best_test_loss
        )
    )
    print("[Valid metrics]")
    _print_metrics(hyp_params, valid_results, valid_truths)
    print("[Test metrics]")
    _print_metrics(hyp_params, test_results, test_truths)
    print("=" * 50)
