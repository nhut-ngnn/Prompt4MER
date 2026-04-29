import random
import os

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
    "--dataset",
    type=str,
    default="mosi",
    help="dataset to use (mosei, mosi, iemocap, meld, sims)",
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
parser.add_argument("--lr", type=float, default=1e-3, help="initial learning rate")
parser.add_argument("--optim", type=str, default="Adam", help="optimizer to use")
parser.add_argument("--num_epochs", type=int, default=40, help="number of epochs")
parser.add_argument("--when", type=int, default=10, help="when to decay learning rate")
parser.add_argument("--drop_rate", type=float, default=0.6)


# Logistics
parser.add_argument(
    "--log_interval",
    type=int,
    default=30,
    help="frequency of result logging (default: 30)",
)
parser.add_argument("--seed", type=int, default=666, help="random seed")
parser.add_argument("--num_seeds", type=int, default=1, help="number of seeds to run")
parser.add_argument("--seed_stride", type=int, default=1, help="increment between seeds")
parser.add_argument("--no_cuda", action="store_true", help="do not use cuda")
parser.add_argument("--gpu_id", type=int, default=0, help="CUDA device index to use")
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
    "--lambda_rec",
    type=float,
    default=0.1,
    help="weight for reconstruction loss in Prompt4MSER loss",
)
parser.add_argument(
    "--lambda_cos",
    type=float,
    default=0.05,
    help="weight for cosine loss in Prompt4MSER loss",
)
parser.add_argument(
    "--max_missing_prob",
    type=float,
    default=0.5,
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

output_dim_dict = {"mosi": 1, "mosei": 1, "sims": 1, "iemocap": 4, "meld": 7}

criterion_dict = {"iemocap": "CrossEntropyLoss", "meld": "CrossEntropyLoss"}


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

    aggregated = {}
    metric_groups = {
        "valid_loss": [summary["valid_loss"] for summary in run_summaries],
        "test_loss": [summary["test_loss"] for summary in run_summaries],
    }
    for split in ["valid_metrics", "test_metrics"]:
        keys = run_summaries[0][split].keys()
        for key in keys:
            metric_groups[f"{split}.{key}"] = [summary[split][key] for summary in run_summaries]

    for key, values in metric_groups.items():
        arr = np.asarray(values, dtype=np.float64)
        aggregated[key] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
            "values": [float(v) for v in arr.tolist()],
        }
    return aggregated


if __name__ == "__main__":
    seeds = [args.seed + i * args.seed_stride for i in range(args.num_seeds)]

    if args.eval_only or args.num_seeds == 1:
        run_name = _seed_run_name(args.name, seeds[0]) if args.num_seeds > 1 else args.name
        hyp_params.seed = seeds[0]
        hyp_params.name = run_name
        setup_seed(hyp_params.seed)
        train.initiate(hyp_params, trainloder, validloder, testloder)
    else:
        run_summaries = []
        base_name = args.name
        for run_idx, seed in enumerate(seeds, start=1):
            print("=" * 60)
            print(f"Seed run {run_idx}/{len(seeds)} | seed={seed}")
            print("=" * 60)
            setup_seed(seed)
            hyp_params.seed = seed
            hyp_params.name = _seed_run_name(base_name, seed)
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
