#!/usr/bin/env python3
"""Command-line entrypoint for Prompt4MSER training and evaluation."""

import argparse
import csv
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from src import train
from src.utils import get_loader


OUTPUT_DIMS = {
    "mosi": 1,
    "mosei": 1,
    "sims": 1,
    "iemocap": 4,
    "meld": 7,
    "msp-improv": 4,
}
CRITERIA = {
    "iemocap": "CrossEntropyLoss",
    "meld": "CrossEntropyLoss",
    "msp-improv": "CrossEntropyLoss",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prompt4MSER missing-modality runner")

    # Tasks / data.
    parser.add_argument("--pretrained_model", type=str, default=None)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--eval_split", type=str, default="test", choices=["valid", "test"])
    parser.add_argument(
        "--eval_modalities",
        type=str,
        default=None,
        help="comma-separated eval modality cases: a,t,v,at,av,tv,atv (or full/all)",
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
        default="mosi",
        choices=["mosi", "mosei", "iemocap", "meld", "msp-improv", "sims"],
    )
    parser.add_argument("--data_path", type=str, default=None)

    # Dropouts.
    parser.add_argument("--attn_dropout", type=float, default=0.1)
    parser.add_argument("--attn_dropout_a", type=float, default=0.1)
    parser.add_argument("--attn_dropout_v", type=float, default=0.1)
    parser.add_argument("--relu_dropout", type=float, default=0.1)
    parser.add_argument("--embed_dropout", type=float, default=0.25)
    parser.add_argument("--res_dropout", type=float, default=0.1)
    parser.add_argument("--out_dropout", type=float, default=0.1)

    # Architecture.
    parser.add_argument("--nlevels", type=int, default=5)
    parser.add_argument("--proj_dim", type=int, default=30)
    parser.add_argument("--num_heads", type=int, default=5)
    parser.add_argument("--attn_mask", action="store_false")
    parser.add_argument("--prompt_dim", type=int, default=30)
    parser.add_argument("--prompt_length", type=int, default=16)
    parser.add_argument("--cross_attn_heads", type=int, default=0)
    parser.add_argument("--prompt_dropout", type=float, default=0.0)
    parser.add_argument("--missing_modality_dropout", type=float, default=0.0)
    parser.add_argument(
        "--fusion_head_output_type",
        type=str,
        default="attn",
        choices=["mean", "max", "attn"],
    )
    parser.add_argument("--audio_norm_type", type=str, default="none", choices=["none", "min_max"])
    parser.add_argument("--linear_layer_output", type=str, default="")

    # Optimization.
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--clip", type=float, default=0.8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--optim", type=str, default="AdamW")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_eps", type=float, default=1e-8)
    parser.add_argument("--num_epochs", type=int, default=40)
    parser.add_argument("--when", type=int, default=5)
    parser.add_argument("--scheduler_factor", type=float, default=0.5)
    parser.add_argument("--min_lr", type=float, default=1e-6)

    # Missing-modality sampler.
    parser.add_argument("--max_missing_prob", type=float, default=0.5)
    parser.add_argument("--double_missing_prob", type=float, default=0.25)

    # Runtime / logging.
    parser.add_argument("--log_interval", type=int, default=30)
    parser.add_argument("--seed", type=int, default=32)
    parser.add_argument("--num_seeds", type=int, default=5)
    parser.add_argument("--seed_stride", type=int, default=1)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--use_dataparallel", action="store_true")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--print_prompt_sample", action="store_true")
    parser.add_argument("--prompt_sample_out", type=str, default=None)
    parser.add_argument("--skip_final_eval", action="store_true")
    parser.add_argument("--skip_epoch_test_eval", action="store_true")
    parser.add_argument("--eval_test_each_epoch", action="store_true")

    return parser


def setup_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def select_device(args: argparse.Namespace) -> Tuple[bool, torch.device]:
    if not torch.cuda.is_available():
        return False, torch.device("cpu")
    if args.no_cuda:
        print("WARNING: CUDA is available, but --no_cuda was set")
        return False, torch.device("cpu")
    torch.cuda.manual_seed(args.seed)
    return True, torch.device(f"cuda:{args.gpu_id}")


def seed_run_name(base_name: Optional[str], seed: int) -> Optional[str]:
    if base_name is None:
        return None
    root, ext = os.path.splitext(base_name)
    if ext:
        return f"{root}.seed{seed}{ext}"
    return f"{base_name}.seed{seed}"


def collect_scalar_metrics(run_summaries: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    if not run_summaries:
        return {}

    def flatten_scalars(data: Any, prefix: str = "") -> Dict[str, float]:
        scalars: Dict[str, float] = {}
        if isinstance(data, dict):
            for key, value in data.items():
                next_prefix = f"{prefix}.{key}" if prefix else str(key)
                scalars.update(flatten_scalars(value, next_prefix))
        elif isinstance(data, (int, float, np.floating)):
            scalars[prefix] = float(data)
        return scalars

    metric_groups: Dict[str, List[float]] = {}
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


def flatten_scalars(data: Any, prefix: str = "") -> Dict[str, float]:
    scalars: Dict[str, float] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            scalars.update(flatten_scalars(value, next_prefix))
    elif isinstance(data, (int, float, np.floating)):
        scalars[prefix] = float(data)
    return scalars


def write_eval_csv(
    path: str,
    run_summaries: List[Dict[str, Any]],
    aggregated: Dict[str, Dict[str, Any]],
    seeds: List[int],
) -> None:
    metric_names = set()
    rows: List[Dict[str, Any]] = []

    for seed, summary in zip(seeds, run_summaries):
        for modality, values in summary.items():
            flat_values = flatten_scalars(values)
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
        grouped: Dict[str, Dict[str, float]] = {}
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


def default_eval_csv_path(args: argparse.Namespace) -> str:
    base_path = args.checkpoint or args.name or "eval_results"
    root, _ = os.path.splitext(base_path)
    return f"{root}_eval.csv"


def prepare_hyp_params(args: argparse.Namespace, use_cuda: bool, device: torch.device):
    dataloaders, orig_dims, n_nums, seq_len = get_loader(args)

    args.orig_d_l, args.orig_d_a, args.orig_d_v = orig_dims
    args.layers = args.nlevels
    args.use_cuda = use_cuda
    args.device = device
    args.when = args.when
    args.n_train, args.n_valid, args.n_test = n_nums
    args.output_dim = OUTPUT_DIMS.get(args.dataset, 1)
    args.criterion = CRITERIA.get(args.dataset, "L1Loss")
    args.seq_len = seq_len
    return dataloaders, args


def run(args: argparse.Namespace):
    args.dataset = args.dataset.strip().lower()
    setup_seed(args.seed)
    use_cuda, device = select_device(args)
    print(f"Using device: {device}")

    dataloaders, hyp_params = prepare_hyp_params(args, use_cuda, device)
    train_loader = dataloaders["train"]
    valid_loader = dataloaders["valid"]
    test_loader = dataloaders["test"]

    seeds = [args.seed + i * args.seed_stride for i in range(args.num_seeds)]
    base_name = args.name
    base_checkpoint = args.checkpoint
    base_pretrained_model = args.pretrained_model
    eval_csv_path = args.eval_csv or (default_eval_csv_path(args) if args.eval_only else None)

    if args.num_seeds == 1:
        hyp_params.seed = seeds[0]
        hyp_params.name = args.name
        hyp_params.checkpoint = base_checkpoint
        hyp_params.pretrained_model = base_pretrained_model
        setup_seed(hyp_params.seed)
        summary = train.initiate(hyp_params, train_loader, valid_loader, test_loader)
        if eval_csv_path is not None:
            aggregated = collect_scalar_metrics([summary])
            write_eval_csv(eval_csv_path, [summary], aggregated, seeds)
        return summary

    run_summaries = []
    for run_idx, seed in enumerate(seeds, start=1):
        print("=" * 60)
        print(f"Seed run {run_idx}/{len(seeds)} | seed={seed}")
        print("=" * 60)
        setup_seed(seed)
        hyp_params.seed = seed
        hyp_params.name = seed_run_name(base_name, seed)
        hyp_params.checkpoint = seed_run_name(base_checkpoint, seed)
        hyp_params.pretrained_model = seed_run_name(base_pretrained_model, seed)
        run_summaries.append(train.initiate(hyp_params, train_loader, valid_loader, test_loader))

    aggregated = collect_scalar_metrics(run_summaries)
    print("=" * 60)
    print(f"Aggregate over {len(seeds)} seeds")
    print("=" * 60)
    for key, stats in aggregated.items():
        print(f"{key}: mean={stats['mean']:.6f} std={stats['std']:.6f} values={stats['values']}")
    if eval_csv_path is not None:
        write_eval_csv(eval_csv_path, run_summaries, aggregated, seeds)
    return aggregated


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    main()
