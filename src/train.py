import time

import torch
import torch.optim as optim
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src import model as mm
from src.eval_metrics import get_metrics
from src.evaluate import (
    evaluate_only,
    evaluate_split,
    load_model,
    print_metrics,
    print_prompted_sample,
)
from src.utils import transfer_model


def _build_optimizer(model, hyp_params):
    optim_name = hyp_params.optim
    optimizer_cls = getattr(optim, optim_name)
    weight_decay = float(getattr(hyp_params, "weight_decay", 0.0))
    optimizer_kwargs = {
        "lr": hyp_params.lr,
        "betas": (
            float(getattr(hyp_params, "adam_beta1", 0.9)),
            float(getattr(hyp_params, "adam_beta2", 0.999)),
        ),
        "eps": float(getattr(hyp_params, "adam_eps", 1e-8)),
    }

    if weight_decay > 0.0:
        decay_params = []
        no_decay_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            lower_name = name.lower()
            if (
                param.ndim == 1
                or name.endswith(".bias")
                or "norm" in lower_name
                or "prompt" in lower_name
            ):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        param_groups = []
        if decay_params:
            param_groups.append({"params": decay_params, "weight_decay": weight_decay})
        if no_decay_params:
            param_groups.append({"params": no_decay_params, "weight_decay": 0.0})
    else:
        param_groups = model.parameters()
        optimizer_kwargs["weight_decay"] = 0.0

    try:
        return optimizer_cls(param_groups, **optimizer_kwargs)
    except TypeError:
        optimizer_kwargs.pop("betas", None)
        optimizer_kwargs.pop("eps", None)
        return optimizer_cls(param_groups, **optimizer_kwargs)


def initiate(hyp_params, train_loader, valid_loader, test_loader):
    if hyp_params.eval_only:
        return evaluate_only(hyp_params, valid_loader, test_loader)

    if hyp_params.pretrained_model is not None:
        model = mm.DualStreamPromptLearningNetwork(hyp_params)
        model = transfer_model(model, hyp_params.pretrained_model)
    else:
        model = mm.DualStreamPromptLearningNetwork(hyp_params)

    model = model.to(hyp_params.device)

    optimizer = _build_optimizer(model, hyp_params)
    criterion = getattr(nn, hyp_params.criterion)()

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=hyp_params.when,
        factor=float(getattr(hyp_params, "scheduler_factor", 0.5)),
        min_lr=float(getattr(hyp_params, "min_lr", 1e-6)),
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

    def train_epoch(model, optimizer, criterion, epoch):
        model.train()
        num_batches = hyp_params.n_train // hyp_params.batch_size
        is_classification_dataset = hyp_params.dataset in {
            "iemocap",
            "meld",
            "msp-improv",
        }
        use_dataparallel = bool(getattr(hyp_params, "use_dataparallel", False))
        proc_total_loss, proc_task_loss, proc_size = (
            0,
            0,
            0,
        )
        max_missing_prob = float(getattr(hyp_params, "max_missing_prob", 0.5))
        double_missing_prob = float(getattr(hyp_params, "double_missing_prob", 0.25))
        start_time = time.time()

        for i_batch, (batch_X, batch_Y, missing_mod) in enumerate(train_loader):
            text, audio, vision = batch_X
            eval_attr = batch_Y
            model.zero_grad()

            text = text.to(hyp_params.device)
            audio = audio.to(hyp_params.device)
            vision = vision.to(hyp_params.device)
            eval_attr = eval_attr.to(hyp_params.device)
            if is_classification_dataset:
                eval_attr = eval_attr.squeeze(-1)
                eval_attr = eval_attr.long()

            batch_size = text.size(0)
            net = nn.DataParallel(model) if use_dataparallel and batch_size > 10 else model
            missing_mod = mm.sample_missing_mod(
                batch_size=batch_size,
                device=hyp_params.device,
                epoch=epoch,
                max_epoch=hyp_params.num_epochs,
                max_missing_prob=max_missing_prob,
                double_missing_prob=double_missing_prob,
            )

            aux_outputs = net(text, audio, vision, missing_mod, return_aux=True)
            preds = aux_outputs["logits"]
            if is_classification_dataset:
                preds = preds.view(-1, hyp_params.output_dim)
                task_loss = criterion(preds, eval_attr.view(-1))
            else:
                eval_attr = eval_attr.view_as(preds)
                task_loss = criterion(preds, eval_attr)
            total_loss = task_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
            optimizer.step()

            proc_total_loss += total_loss.item() * batch_size
            proc_task_loss += task_loss.item() * batch_size
            proc_size += batch_size

            if i_batch % hyp_params.log_interval == 0 and i_batch > 0:
                avg_total_loss = proc_total_loss / proc_size
                avg_task_loss = proc_task_loss / proc_size
                elapsed_time = time.time() - start_time
                if is_classification_dataset:
                    print(
                        "Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | "
                        "Cls Loss {:5.4f} | Total Loss {:5.4f}".format(
                            epoch,
                            i_batch,
                            num_batches,
                            elapsed_time * 1000 / hyp_params.log_interval,
                            avg_task_loss,
                            avg_total_loss,
                        )
                    )
                else:
                    print(
                        "Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Train Loss {:5.4f}".format(
                            epoch,
                            i_batch,
                            num_batches,
                            elapsed_time * 1000 / hyp_params.log_interval,
                            avg_total_loss,
                        )
                    )
                proc_total_loss, proc_task_loss, proc_size = (
                    0,
                    0,
                    0,
                )
                start_time = time.time()

    best_valid = 1e8
    best_test_at_valid = None
    for epoch in range(1, hyp_params.num_epochs + 1):
        start = time.time()
        train_epoch(model, optimizer, criterion, epoch)
        val_loss, _, _ = evaluate_split(
            model, criterion, hyp_params, valid_loader, test_loader, test=False
        )
        skip_epoch_test_eval = getattr(hyp_params, "skip_epoch_test_eval", False)
        skip_epoch_test_eval = skip_epoch_test_eval or (
            hyp_params.dataset in {"mosi", "mosei"}
            and not getattr(hyp_params, "eval_test_each_epoch", False)
        )
        if skip_epoch_test_eval:
            test_loss = float("nan")
        else:
            test_loss, _, _ = evaluate_split(
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
            torch.save({"model_state_dict": model.state_dict()}, hyp_params.name)
            best_valid = val_loss
            best_test_at_valid = None if test_loss != test_loss else test_loss

    if getattr(hyp_params, "skip_final_eval", False):
        del model
        if hyp_params.use_cuda:
            torch.cuda.empty_cache()
        return {
            "valid_loss": float(best_valid),
            "test_loss": float(best_test_at_valid) if best_test_at_valid is not None else None,
            "valid_metrics": {},
            "test_metrics": {},
        }

    del model
    if hyp_params.use_cuda:
        torch.cuda.empty_cache()

    model = load_model(hyp_params.name, hyp_params)
    best_valid_loss, valid_results, valid_truths = evaluate_split(
        model, criterion, hyp_params, valid_loader, test_loader, test=False
    )
    best_test_loss, test_results, test_truths = evaluate_split(
        model, criterion, hyp_params, valid_loader, test_loader, test=True
    )

    print("=" * 50)
    print(
        "Best checkpoint summary | Valid Loss {:5.4f} | Test Loss {:5.4f}".format(
            best_valid_loss, best_test_loss
        )
    )
    print("[Valid metrics]")
    print_metrics(hyp_params, valid_results, valid_truths)
    print("[Test metrics]")
    print_metrics(hyp_params, test_results, test_truths)
    print("=" * 50)
    print_prompted_sample(model, test_loader, hyp_params, "Best checkpoint prompted sample")
    return {
        "valid_loss": float(best_valid_loss),
        "test_loss": float(best_test_loss),
        "valid_metrics": get_metrics(hyp_params.dataset, valid_results, valid_truths),
        "test_metrics": get_metrics(hyp_params.dataset, test_results, test_truths),
    }
