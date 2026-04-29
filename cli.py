#!/usr/bin/env python3
"""Command-line entrypoint for Prompt4MSER training and evaluation."""

import argparse
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from src import train
from src.utils import get_loader


OUTPUT_DIMS = {"mosi": 1, "mosei": 1, "sims": 1, "iemocap": 4, "meld": 7}
CRITERIA = {"iemocap": "CrossEntropyLoss", "meld": "CrossEntropyLoss"}


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
        "--dataset",
        type=str,
        default="mosi",
        choices=["mosi", "mosei", "iemocap", "meld", "sims"],
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
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optim", type=str, default="Adam")
    parser.add_argument("--num_epochs", type=int, default=40)
    parser.add_argument("--when", type=int, default=10)

    # New Prompt4MSER training loss / missing-modality sampler.
    parser.add_argument("--lambda_rec", type=float, default=0.1)
    parser.add_argument("--lambda_cos", type=float, default=0.05)
    parser.add_argument("--max_missing_prob", type=float, default=0.5)
    parser.add_argument("--double_missing_prob", type=float, default=0.25)

    # Runtime / logging.
    parser.add_argument("--log_interval", type=int, default=30)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--num_seeds", type=int, default=1)
    parser.add_argument("--seed_stride", type=int, default=1)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--print_prompt_sample", action="store_true")
    parser.add_argument("--prompt_sample_out", type=str, default=None)

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

    metric_groups: Dict[str, List[float]] = {
        "valid_loss": [summary["valid_loss"] for summary in run_summaries],
        "test_loss": [summary["test_loss"] for summary in run_summaries],
    }
    for split in ["valid_metrics", "test_metrics"]:
        for key in run_summaries[0][split].keys():
            metric_groups[f"{split}.{key}"] = [summary[split][key] for summary in run_summaries]

    aggregated = {}
    for key, values in metric_groups.items():
        arr = np.asarray(values, dtype=np.float64)
        aggregated[key] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
            "values": [float(v) for v in arr.tolist()],
        }
    return aggregated


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
    if args.eval_only or args.num_seeds == 1:
        hyp_params.seed = seeds[0]
        hyp_params.name = seed_run_name(args.name, seeds[0]) if args.num_seeds > 1 else args.name
        setup_seed(hyp_params.seed)
        return train.initiate(hyp_params, train_loader, valid_loader, test_loader)

    run_summaries = []
    base_name = args.name
    for run_idx, seed in enumerate(seeds, start=1):
        print("=" * 60)
        print(f"Seed run {run_idx}/{len(seeds)} | seed={seed}")
        print("=" * 60)
        setup_seed(seed)
        hyp_params.seed = seed
        hyp_params.name = seed_run_name(base_name, seed)
        run_summaries.append(train.initiate(hyp_params, train_loader, valid_loader, test_loader))

    aggregated = collect_scalar_metrics(run_summaries)
    print("=" * 60)
    print(f"Aggregate over {len(seeds)} seeds")
    print("=" * 60)
    for key, stats in aggregated.items():
        print(f"{key}: mean={stats['mean']:.6f} std={stats['std']:.6f} values={stats['values']}")
    return aggregated


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    main()
