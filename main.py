import random
import os
import csv

import numpy as np
import torch
import argparse
from src.utils import *
from src import train


parser = argparse.ArgumentParser(description="Missing modality")

# Tasks
parser.add_argument(
    "--pretrained_model",
    type=str,
    default=None,
    help="name of the model to use (Transformer, etc.)",
)
parser.add_argument(
    "--eval_only",
    action="store_true",
    help="evaluate a saved checkpoint without training",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="path to a saved checkpoint for eval-only mode",
)
parser.add_argument(
    "--eval_split",
    type=str,
    default="test",
    choices=["valid", "test"],
    help="split to use in eval-only mode",
)
parser.add_argument(
    "--eval_modalities",
    type=str,
    default=None,
    help="comma-separated eval-only modality cases: a,t,v,at,av,tv,atv (or full/all)",
)
parser.add_argument(
    "--eval_csv",
    type=str,
    default=None,
    help="optional CSV path for eval-only per-seed and aggregate metrics",
)
parser.add_argument(
    "--dataset",
    type=str,
    default="iemocap",
    choices=["iemocap", "msp-improv"],
    help="dataset to use (iemocap, msp-improv)",
)
parser.add_argument(
    "--data_path",
    type=str,
    default=None,
    help="path for storing the dataset",
)

# Dropouts
parser.add_argument("--attn_dropout", type=float, default=0.1, help="attention dropout")
parser.add_argument(
    "--attn_dropout_a", type=float, default=0.1, help="attention dropout (for audio)"
)
parser.add_argument(
    "--attn_dropout_v", type=float, default=0.1, help="attention dropout (for visual)"
)
parser.add_argument("--relu_dropout", type=float, default=0.1, help="relu dropout")
parser.add_argument(
    "--embed_dropout", type=float, default=0.25, help="embedding dropout"
)
parser.add_argument(
    "--res_dropout", type=float, default=0.1, help="residual block dropout"
)
parser.add_argument(
    "--out_dropout", type=float, default=0.1, help="output layer dropout"
)

# Architecture
parser.add_argument(
    "--nlevels", type=int, default=5, help="number of layers in the network"
)
parser.add_argument(
    "--proj_dim", type=int, default=30, help="projection dimension of the network"
)
parser.add_argument(
    "--num_heads",
    type=int,
    default=5,
    help="number of heads for the transformer network",
)
parser.add_argument(
    "--attn_mask", action="store_false", help="use attention mask for Transformer"
)
parser.add_argument("--prompt_dim", type=int, default=30)
parser.add_argument("--prompt_length", type=int, default=16)
parser.add_argument(
    "--cross_attn_heads",
    type=int,
    default=0,
    help="number of cross-attention heads for the dual-stream prompt model",
)
parser.add_argument(
    "--prompt_dropout",
    type=float,
    default=0.0,
    help="dropout applied to dual-stream prompt parameters",
)
parser.add_argument(
    "--missing_modality_dropout",
    type=float,
    default=0.0,
    help="training-time probability of dropping an available modality in the dual-stream model",
)
parser.add_argument(
    "--fusion_head_output_type",
    type=str,
    default="attn",
    choices=["mean", "max", "attn"],
    help="readout strategy for 4M-SER fusion head",
)
parser.add_argument(
    "--audio_norm_type",
    type=str,
    default="none",
    choices=["none", "min_max"],
    help="audio feature normalization in 4M-SER",
)
parser.add_argument(
    "--linear_layer_output",
    type=str,
    default="",
    help="comma-separated hidden dimensions for 4M-SER classifier head, e.g. 256,128",
)


# Tuning
parser.add_argument(
    "--batch_size", type=int, default=64, metavar="N", help="batch size"
)
parser.add_argument("--clip", type=float, default=0.8, help="gradient clip value")
parser.add_argument("--lr", type=float, default=5e-4, help="initial learning rate")
parser.add_argument("--optim", type=str, default="AdamW", help="optimizer to use")
parser.add_argument(
    "--weight_decay",
    type=float,
    default=1e-4,
    help="decoupled weight decay for AdamW-compatible optimizers",
)
parser.add_argument("--adam_beta1", type=float, default=0.9, help="Adam beta1")
parser.add_argument("--adam_beta2", type=float, default=0.999, help="Adam beta2")
parser.add_argument("--adam_eps", type=float, default=1e-8, help="Adam epsilon")
parser.add_argument("--num_epochs", type=int, default=40, help="number of epochs")
parser.add_argument("--when", type=int, default=5, help="LR scheduler patience")
parser.add_argument(
    "--scheduler_factor",
    type=float,
    default=0.5,
    help="multiplicative LR decay factor on validation plateau",
)
parser.add_argument(
    "--min_lr",
    type=float,
    default=1e-6,
    help="minimum learning rate after scheduler decay",
)


# Logistics
parser.add_argument(
    "--log_interval",
    type=int,
    default=30,
    help="frequency of result logging (default: 30)",
)
parser.add_argument("--seed", type=int, default=32, help="random seed")
parser.add_argument("--num_seeds", type=int, default=5, help="number of seeds to run")
parser.add_argument("--seed_stride", type=int, default=1, help="increment between seeds")
parser.add_argument("--no_cuda", action="store_true", help="do not use cuda")
parser.add_argument("--gpu_id", type=int, default=0, help="CUDA device index to use")
parser.add_argument("--use_dataparallel", action="store_true", help="enable multi-GPU DataParallel")
parser.add_argument("--name", type=str, default=None, help="name of the trial")
parser.add_argument(
    "--print_prompt_sample",
    action="store_true",
    help="print one sample after prompt generation",
)
parser.add_argument(
    "--prompt_sample_out",
    type=str,
    default=None,
    help="optional .pt path to save one prompted sample",
)
parser.add_argument(
    "--skip_final_eval",
    action="store_true",
    help="skip reloading the best checkpoint for final train/valid/test summary after training",
)
parser.add_argument(
    "--skip_epoch_test_eval",
    action="store_true",
    help="skip test-set evaluation after each training epoch",
)
parser.add_argument(
    "--eval_test_each_epoch",
    action="store_true",
    help="force test-set evaluation after each training epoch",
)
parser.add_argument(
    "--max_missing_prob",
    type=float,
    default=50,
    help="maximum missing-modality probability for training sampler",
)
parser.add_argument(
    "--double_missing_prob",
    type=float,
    default=0.25,
    help="probability of sampling two missing modalities when a sample is masked",
)
args = parser.parse_args()


dataset = str.lower(args.dataset.strip())
args.dataset = dataset

output_dim_dict = {"iemocap": 4, "msp-improv": 4}

criterion_dict = {"iemocap": "CrossEntropyLoss", "msp-improv": "CrossEntropyLoss"}


def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

use_cuda = False
device = torch.device("cpu")
if torch.cuda.is_available():
    if args.no_cuda:
        print(
            "WARNING: You have a CUDA device, so you should probably not run with --no_cuda"
        )
    else:
        torch.cuda.manual_seed(args.seed)
        use_cuda = True
        device = torch.device(f"cuda:{args.gpu_id}")

setup_seed(args.seed)
print(f"Using device: {device}")

####################################################################
#
# Load the dataset
#
####################################################################

dataloaders, orig_dims, n_nums, seq_len = get_loader(args)
trainloder = dataloaders["train"]
validloder = dataloaders["valid"]
testloder = dataloaders["test"]
####################################################################
#
# Hyperparameters
#
####################################################################
hyp_params = args
hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v = orig_dims
hyp_params.layers = args.nlevels
hyp_params.use_cuda = use_cuda
hyp_params.device = device
hyp_params.dataset = dataset
hyp_params.when = args.when
hyp_params.n_train, hyp_params.n_valid, hyp_params.n_test = n_nums
hyp_params.output_dim = output_dim_dict.get(dataset, 1)
hyp_params.criterion = criterion_dict.get(dataset, "L1Loss")
hyp_params.seq_len = seq_len


def _seed_run_name(base_name, seed):
    if base_name is None:
        return None
    root, ext = os.path.splitext(base_name)
    if ext:
        return f"{root}.seed{seed}{ext}"
    return f"{base_name}.seed{seed}"


def _collect_scalar_metrics(run_summaries):
    if not run_summaries:
        return {}

    def flatten_scalars(data, prefix=""):
        scalars = {}
        if isinstance(data, dict):
            for key, value in data.items():
                next_prefix = f"{prefix}.{key}" if prefix else str(key)
                scalars.update(flatten_scalars(value, next_prefix))
        elif isinstance(data, (int, float, np.floating)):
            scalars[prefix] = float(data)
        return scalars

    metric_groups = {}
    for summary in run_summaries:
        for key, value in flatten_scalars(summary).items():
            metric_groups.setdefault(key, []).append(value)

    aggregated = {}
    for key, values in metric_groups.items():
        if len(values) != len(run_summaries):
            continue
        arr = np.asarray(values, dtype=np.float64)
        aggregated[key] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
            "values": [float(v) for v in arr.tolist()],
        }
    return aggregated


def _flatten_scalars(data, prefix=""):
    scalars = {}
    if isinstance(data, dict):
        for key, value in data.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            scalars.update(_flatten_scalars(value, next_prefix))
    elif isinstance(data, (int, float, np.floating)):
        scalars[prefix] = float(data)
    return scalars


def _write_eval_csv(path, run_summaries, aggregated, seeds):
    metric_names = set()
    rows = []

    for seed, summary in zip(seeds, run_summaries):
        for modality, values in summary.items():
            flat_values = _flatten_scalars(values)
            metric_names.update(flat_values.keys())
            rows.append(
                {
                    "row_type": "seed",
                    "seed": seed,
                    "modality": modality,
                    **flat_values,
                }
            )

    for stat_name in ["mean", "std"]:
        grouped = {}
        for key, stats in aggregated.items():
            modality, metric = key.split(".", 1) if "." in key else ("overall", key)
            grouped.setdefault(modality, {})[metric] = stats[stat_name]
            metric_names.add(metric)
        for modality, values in grouped.items():
            rows.append(
                {
                    "row_type": stat_name,
                    "seed": "",
                    "modality": modality,
                    **values,
                }
            )

    fieldnames = ["row_type", "seed", "modality"] + sorted(metric_names)
    output_path = os.path.abspath(path)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved eval CSV to {output_path}")


def _default_eval_csv_path():
    base_path = args.checkpoint or args.name or "eval_results"
    root, _ = os.path.splitext(base_path)
    return f"{root}_eval.csv"


if __name__ == "__main__":
    seeds = [args.seed + i * args.seed_stride for i in range(args.num_seeds)]
    base_name = args.name
    base_checkpoint = args.checkpoint
    base_pretrained_model = args.pretrained_model
    eval_csv_path = args.eval_csv or (_default_eval_csv_path() if args.eval_only else None)

    if args.num_seeds == 1:
        run_name = _seed_run_name(args.name, seeds[0]) if args.num_seeds > 1 else args.name
        hyp_params.seed = seeds[0]
        hyp_params.name = run_name
        hyp_params.checkpoint = base_checkpoint
        hyp_params.pretrained_model = base_pretrained_model
        setup_seed(hyp_params.seed)
        summary = train.initiate(hyp_params, trainloder, validloder, testloder)
        if eval_csv_path is not None:
            aggregated = _collect_scalar_metrics([summary])
            _write_eval_csv(eval_csv_path, [summary], aggregated, seeds)
    else:
        run_summaries = []
        for run_idx, seed in enumerate(seeds, start=1):
            print("=" * 60)
            print(f"Seed run {run_idx}/{len(seeds)} | seed={seed}")
            print("=" * 60)
            setup_seed(seed)
            hyp_params.seed = seed
            hyp_params.name = _seed_run_name(base_name, seed)
            hyp_params.checkpoint = _seed_run_name(base_checkpoint, seed)
            hyp_params.pretrained_model = _seed_run_name(base_pretrained_model, seed)
            summary = train.initiate(hyp_params, trainloder, validloder, testloder)
            run_summaries.append(summary)

        aggregated = _collect_scalar_metrics(run_summaries)
        print("=" * 60)
        print(f"Aggregate over {len(seeds)} seeds")
        print("=" * 60)
        for key, stats in aggregated.items():
            print(
                f"{key}: mean={stats['mean']:.6f} std={stats['std']:.6f} values={stats['values']}"
            )
        if eval_csv_path is not None:
            _write_eval_csv(eval_csv_path, run_summaries, aggregated, seeds)
