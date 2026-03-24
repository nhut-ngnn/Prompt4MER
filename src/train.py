import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src import model as mm
from src.evaluate import (
    evaluate_only,
    evaluate_split,
    get_underlying_model,
    load_model,
    print_metrics,
    print_prompted_sample,
)
from src.utils import transfer_model


def _supports_feature_alignment(model):
    base_model = get_underlying_model(model)
    return hasattr(base_model, "get_complete_data") and hasattr(
        base_model, "missing_type_prompt"
    )


def _get_lambda_gen(epoch, hyp_params):
    base_lambda = float(getattr(hyp_params, "lambda_gen", 0.0))
    warmup_epochs = int(getattr(hyp_params, "lambda_gen_warmup_epochs", 0))
    if warmup_epochs <= 0:
        return base_lambda
    warmup_scale = min(1.0, float(epoch) / float(warmup_epochs))
    return base_lambda * warmup_scale


def _compute_feature_alignment_loss(aux_outputs, missing_mod, hyp_params):
    generated = aux_outputs.get("generated_features")
    real = aux_outputs.get("real_features")
    if generated is None or real is None:
        zero = aux_outputs["logits"].new_tensor(0.0)
        return {
            "mse_loss": zero,
            "cos_loss": zero,
            "gen_loss": zero,
            "num_active_modalities": 0,
        }

    ref_tensor = generated["text"]
    missing_mod = missing_mod.to(ref_tensor.device).long().view(-1)

    missing_masks = {
        "text": (missing_mod == 0) | (missing_mod == 3) | (missing_mod == 4),
        "audio": (missing_mod == 1) | (missing_mod == 3) | (missing_mod == 5),
        "vision": (missing_mod == 2) | (missing_mod == 4) | (missing_mod == 5),
    }

    mse_terms, cos_terms, total_terms = [], [], []
    reduction = getattr(hyp_params, "gen_loss_reduction", "mean")
    for modality, mask in missing_masks.items():
        if not torch.any(mask):
            continue

        mask_count = int(mask.sum().item())
        gen_feat = generated[modality][mask].reshape(mask_count, -1)
        real_feat = real[modality][mask]
        if getattr(hyp_params, "detach_alignment_target", False):
            real_feat = real_feat.detach()
        real_feat = real_feat.reshape(mask_count, -1)

        mse_loss = F.mse_loss(gen_feat, real_feat, reduction=reduction)
        cos_vec = 1.0 - F.cosine_similarity(gen_feat, real_feat, dim=-1)
        cos_loss = cos_vec.mean() if reduction == "mean" else cos_vec.sum()
        total_loss = (
            float(getattr(hyp_params, "alpha_mse", 1.0)) * mse_loss
            + float(getattr(hyp_params, "beta_cos", 1.0)) * cos_loss
        )
        mse_terms.append(mse_loss)
        cos_terms.append(cos_loss)
        total_terms.append(total_loss)

    if not total_terms:
        zero = ref_tensor.new_tensor(0.0)
        return {
            "mse_loss": zero,
            "cos_loss": zero,
            "gen_loss": zero,
            "num_active_modalities": 0,
        }

    if reduction == "mean":
        mse_loss = torch.stack(mse_terms).mean()
        cos_loss = torch.stack(cos_terms).mean()
        gen_loss = torch.stack(total_terms).mean()
    else:
        mse_loss = torch.stack(mse_terms).sum()
        cos_loss = torch.stack(cos_terms).sum()
        gen_loss = torch.stack(total_terms).sum()

    return {
        "mse_loss": mse_loss,
        "cos_loss": cos_loss,
        "gen_loss": gen_loss,
        "num_active_modalities": len(total_terms),
    }


def initiate(hyp_params, train_loader, valid_loader, test_loader):
    if hyp_params.eval_only:
        return evaluate_only(hyp_params, valid_loader, test_loader)

    if (
        getattr(hyp_params, "enable_feature_alignment_loss", False)
        and getattr(hyp_params, "pretrained_model", None) is None
    ):
        print(
            "Feature-alignment loss is enabled, but no pretrained backbone is provided. "
            "Running backbone training without auxiliary alignment loss."
        )

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

    def train_epoch(model, optimizer, criterion, epoch):
        model.train()
        num_batches = hyp_params.n_train // hyp_params.batch_size
        proc_total_loss, proc_task_loss, proc_size = 0, 0, 0
        proc_gen_mse, proc_gen_cos, proc_gen_total = 0, 0, 0
        use_alignment_loss = (
            bool(getattr(hyp_params, "enable_feature_alignment_loss", False))
            and getattr(hyp_params, "pretrained_model", None) is not None
            and _supports_feature_alignment(model)
        )
        lambda_gen = _get_lambda_gen(epoch, hyp_params) if use_alignment_loss else 0.0
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
            aux_outputs = None
            if use_alignment_loss:
                aux_outputs = net(text, audio, vision, missing_mod, return_aux=True)
                preds = aux_outputs["logits"]
            else:
                preds = net(text, audio, vision, missing_mod)

            if hyp_params.dataset == "iemocap":
                preds = preds.view(-1, 4)
                eval_attr = eval_attr.view(-1)
            task_loss = criterion(preds, eval_attr)

            if use_alignment_loss:
                align_loss_terms = _compute_feature_alignment_loss(
                    aux_outputs, missing_mod, hyp_params
                )
                gen_mse_loss = align_loss_terms["mse_loss"]
                gen_cos_loss = align_loss_terms["cos_loss"]
                gen_total_loss = align_loss_terms["gen_loss"]
                total_loss = task_loss + lambda_gen * gen_total_loss
            else:
                zero = task_loss.new_tensor(0.0)
                gen_mse_loss = zero
                gen_cos_loss = zero
                gen_total_loss = zero
                total_loss = task_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
            optimizer.step()

            proc_total_loss += total_loss.item() * batch_size
            proc_task_loss += task_loss.item() * batch_size
            proc_gen_mse += gen_mse_loss.item() * batch_size
            proc_gen_cos += gen_cos_loss.item() * batch_size
            proc_gen_total += gen_total_loss.item() * batch_size
            proc_size += batch_size

            if i_batch % hyp_params.log_interval == 0 and i_batch > 0:
                avg_total_loss = proc_total_loss / proc_size
                avg_task_loss = proc_task_loss / proc_size
                elapsed_time = time.time() - start_time
                if use_alignment_loss:
                    avg_gen_mse = proc_gen_mse / proc_size
                    avg_gen_cos = proc_gen_cos / proc_size
                    avg_gen_total = proc_gen_total / proc_size
                    print(
                        "Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | "
                        "Task Loss {:5.4f} | Gen MSE {:5.4f} | Gen Cos {:5.4f} | "
                        "Gen Total {:5.4f} | lambda_gen {:5.4f} | Total Loss {:5.4f}".format(
                            epoch,
                            i_batch,
                            num_batches,
                            elapsed_time * 1000 / hyp_params.log_interval,
                            avg_task_loss,
                            avg_gen_mse,
                            avg_gen_cos,
                            avg_gen_total,
                            lambda_gen,
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
                proc_total_loss, proc_task_loss, proc_size = 0, 0, 0
                proc_gen_mse, proc_gen_cos, proc_gen_total = 0, 0, 0
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
