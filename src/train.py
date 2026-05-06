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


def initiate(hyp_params, train_loader, valid_loader, test_loader):
    if hyp_params.eval_only:
        return evaluate_only(hyp_params, valid_loader, test_loader)

    if hyp_params.pretrained_model is not None:
        model = mm.DualStreamPromptLearningNetwork(hyp_params)
        model = transfer_model(model, hyp_params.pretrained_model)
    else:
        model = mm.DualStreamPromptLearningNetwork(hyp_params)

    if float(getattr(hyp_params, "lambda_consistency", 0.0)) > 0.0:
        # TODO: add paired full/missing forward consistency once the training
        # loop has a dedicated auxiliary-loss contract.
        print("lambda_consistency is set, but consistency loss is not active yet.")

    model = model.to(hyp_params.device)

    optimizer = getattr(optim, hyp_params.optim)(model.parameters(), lr=hyp_params.lr)
    criterion = getattr(nn, hyp_params.criterion)()

    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", patience=hyp_params.when, factor=0.1
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
        is_classification_dataset = hyp_params.dataset in {"iemocap", "meld"}
        use_dataparallel = hyp_params.use_cuda and torch.cuda.device_count() > 1
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
            eval_attr = batch_Y.squeeze(-1)
            model.zero_grad()

            text = text.to(hyp_params.device)
            audio = audio.to(hyp_params.device)
            vision = vision.to(hyp_params.device)
            eval_attr = eval_attr.to(hyp_params.device)
            if is_classification_dataset:
                eval_attr = eval_attr.long()

            batch_size = text.size(0)
            net = nn.DataParallel(model) if use_dataparallel and batch_size > 10 else model
            if is_classification_dataset:
                missing_mod = mm.sample_missing_mod(
                    batch_size=batch_size,
                    device=hyp_params.device,
                    epoch=epoch,
                    max_epoch=hyp_params.num_epochs,
                    max_missing_prob=max_missing_prob,
                    double_missing_prob=double_missing_prob,
                )
            else:
                missing_mod = missing_mod.to(hyp_params.device)

            aux_outputs = net(text, audio, vision, missing_mod, return_aux=True)
            preds = aux_outputs["logits"]
            if is_classification_dataset:
                preds = preds.view(-1, hyp_params.output_dim)
                task_loss = criterion(preds, eval_attr.view(-1))
            else:
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
    for epoch in range(1, hyp_params.num_epochs + 1):
        start = time.time()
        train_epoch(model, optimizer, criterion, epoch)
        val_loss, _, _ = evaluate_split(
            model, criterion, hyp_params, valid_loader, test_loader, test=False
        )
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
            torch.save(model, hyp_params.name)
            best_valid = val_loss

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
