## Prompt-Guided Missing Modality Completion for Multimodal Emotion Recognition

### CLI Examples

Run these commands from the repository root.

Pretrain:

```bash
python main.py \
  --dataset iemocap \
  --data_path feature/ \
  --linear_layer_output 512,256 \
  --max_missing_prob 0\
  --double_missing_prob 0\
  --name ./checkpoints/iemocap_4mser_concat_pretrain.pt
```

Fine-tune:

```bash
python main.py \
  --pretrained_model ./checkpoints/mosei_4mser_concat_pretrain.pt \
  --dataset iemocap \
  --data_path feature/ \
  --linear_layer_output 512,256 \
  --name ./checkpoints/iemocap_4mser_concat_finetune.pt
```

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
