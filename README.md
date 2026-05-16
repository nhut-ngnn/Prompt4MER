## Prompt-Guided Missing Modality Completion for Multimodal Emotion Recognition

### CLI Examples

Run these commands from the repository root.

Pretrain:

```bash
python main.py \
  --dataset iemocap \
  --data_path feature/ \
  --linear_layer_output 512,256 \
  --optim AdamW \
  --lr 5e-4 \
  --weight_decay 1e-4 \
  --when 5 \
  --scheduler_factor 0.5 \
  --max_missing_prob 0 \
  --double_missing_prob 0 \
  --num_seeds 5 \
  --name ./checkpoints/iemocap_4mser_concat_pretrain.pt
```

Fine-tune:

```bash
python main.py \
  --pretrained_model ./checkpoints/iemocap_4mser_concat_pretrain.pt \
  --dataset iemocap \
  --data_path feature/ \
  --linear_layer_output 512,256 \
  --optim AdamW \
  --lr 5e-4 \
  --weight_decay 1e-4 \
  --when 5 \
  --scheduler_factor 0.5 \
  --num_seeds 5 \
  --name ./checkpoints/iemocap_4mser_concat_finetune.pt
```

With `--num_seeds 5` and the default `--seed 32 --seed_stride 1`, training runs seeds
`32, 33, 34, 35, 36` and writes per-seed checkpoints such as
`iemocap_4mser_concat_finetune.seed32.pt`.
When fine-tuning with `--pretrained_model ./checkpoints/iemocap_4mser_concat_pretrain.pt`,
each seed automatically loads the matching pretrain checkpoint, for example seed `32`
loads `iemocap_4mser_concat_pretrain.seed32.pt`.

Run IEMOCAP missing-sampler ablation:

```bash
scripts/ablate_iemocap_missing.sh
```

By default this pretrains once with no synthetic missing modality, then fine-tunes/evaluates
`max_missing_prob` from `0.0` to `1.0` with step `0.1`. `double_missing_prob` is fixed at
`0.25`; when `max_missing_prob < 0.25`, missing samples are still randomly sampled with that
lower missing probability.

```text
max_missing_prob = 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
double_missing_prob = 0.25
```

Override the grid or skip stages with environment variables:

```bash
MAX_MISSING_VALUES="0.0 0.2 0.4 0.6 0.8 1.0" scripts/ablate_iemocap_missing.sh
DOUBLE_MISSING_PROB=0.50 scripts/ablate_iemocap_missing.sh
RUN_PRETRAIN=0 RUN_FINETUNE=0 RUN_EVAL=1 scripts/ablate_iemocap_missing.sh
```

Results are written under `./results/iemocap_missing_ablation/`, with one CSV per setting
and `summary.csv` containing the one-seed result for each setting. Set `NUM_SEEDS=5` if you
want the full five-seed mean/std run.

Prepare, train, and evaluate MSP-IMPROV from `/home/minhnhutngnn/MSP-IMPROV`:

```bash
scripts/preprocess_msp_improv.sh
scripts/extract_msp_improv_features.sh
scripts/train_msp_improv.sh
scripts/eval_msp_improv.sh
```

MSP-IMPROV is treated as a 4-class classification dataset with filename labels
`A,H,S,N` mapped to the same class order as IEMOCAP: angry, happy, sad, neutral.
The raw MSP-IMPROV tree currently provides audio/transcripts; feature extraction allows
missing video by default and fills the video branch with a zero embedding.

Run MSP-IMPROV missing-sampler ablation:

```bash
scripts/ablate_msp_improv_missing.sh
```

This mirrors the IEMOCAP ablation: one seed by default, `double_missing_prob=0.25`,
and `max_missing_prob` from `0.0` to `1.0` with step `0.1`. Results are written under
`./results/msp_improv_missing_ablation/`.

Evaluate the IEMOCAP checkpoint:

```bash
python main.py \
  --eval_only \
  --dataset iemocap \
  --data_path feature/ \
  --checkpoint ./checkpoints/iemocap_4mser_concat_finetune.pt \
  --linear_layer_output 512,256 \
  --eval_split test \
  --eval_modalities atv,t,a,v,at,av,tv
```

Eval-only also follows `--num_seeds 5` by default: pass the base checkpoint path and the
runner evaluates the matching per-seed checkpoints, then prints mean/std metrics across seeds.
Eval-only writes CSV results by default next to the base checkpoint, for example
`./checkpoints/iemocap_4mser_concat_finetune_eval.csv`. Use `--eval_csv <path>` to choose
a different output file. The CSV contains one row per seed/modality plus mean and std rows
for each modality.
