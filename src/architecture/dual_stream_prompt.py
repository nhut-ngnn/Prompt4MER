import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple


def _get_valid_num_heads(d_model: int, requested_heads: int) -> int:
    if requested_heads <= 0:
        return 1
    if d_model % requested_heads == 0:
        return requested_heads
    for h in range(requested_heads, 0, -1):
        if d_model % h == 0:
            return h
    return 1


def missing_mod_to_availability_mask(
    missing_mod: Optional[torch.Tensor],
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Convert the repository's 7-state missing code to [B, 3] availability masks.

    Mask order:
        [text, audio, visual]

    Values:
        1 = available, 0 = missing
    """
    if missing_mod is None:
        return torch.ones(batch_size, 3, device=device, dtype=torch.float32)

    if missing_mod.dim() == 2 and missing_mod.size(-1) == 3:
        mask = missing_mod.to(device=device, dtype=torch.float32)
        if mask.size(0) != batch_size:
            raise ValueError(
                f"missing_mask batch size {mask.size(0)} does not match input batch size {batch_size}."
            )
        return mask.clamp(0.0, 1.0)

    missing_mod = missing_mod.to(device=device, dtype=torch.long).view(-1)
    if missing_mod.numel() == 1 and batch_size != 1:
        missing_mod = missing_mod.expand(batch_size)
    if missing_mod.numel() != batch_size:
        raise ValueError(
            f"missing_mod has {missing_mod.numel()} entries, expected batch size {batch_size}."
        )
    if not bool(((missing_mod >= 0) & (missing_mod <= 6)).all().item()):
        raise ValueError("missing_mod values must be in the range [0, 6].")

    mask = torch.ones(batch_size, 3, device=device, dtype=torch.float32)
    miss_text = (missing_mod == 0) | (missing_mod == 3) | (missing_mod == 4)
    miss_audio = (missing_mod == 1) | (missing_mod == 3) | (missing_mod == 5)
    miss_visual = (missing_mod == 2) | (missing_mod == 4) | (missing_mod == 5)
    mask[:, 0] = (~miss_text).float()
    mask[:, 1] = (~miss_audio).float()
    mask[:, 2] = (~miss_visual).float()
    return mask


def apply_missing_modality_dropout(
    missing_mask: torch.Tensor,
    dropout_prob: float,
    training: bool,
) -> torch.Tensor:
    """
    Randomly mark available modalities as missing while preserving at least one
    available modality for every sample.
    """
    if (not training) or dropout_prob <= 0.0:
        return missing_mask

    keep_source = missing_mask > 0.5
    drop = torch.rand_like(missing_mask) < float(dropout_prob)
    dropped = missing_mask.masked_fill(drop & keep_source, 0.0)

    empty_rows = dropped.sum(dim=1) < 1.0
    if empty_rows.any():
        candidates = keep_source[empty_rows]
        random_scores = torch.rand(
            candidates.size(0),
            candidates.size(1),
            device=missing_mask.device,
        )
        random_scores = random_scores.masked_fill(~candidates, -1.0)
        restore_idx = random_scores.argmax(dim=1)
        dropped[empty_rows, restore_idx] = 1.0

    return dropped


class MissingModalityPromptBank(nn.Module):
    """
    Prompt bank for observed-modality offsets and missing-modality fallbacks.
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.modality_prompts = nn.Parameter(torch.empty(3, self.d_model))
        self.missing_prompts = nn.Parameter(torch.empty(3, self.d_model))
        self.dropout = nn.Dropout(float(dropout))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.modality_prompts, mean=0.0, std=0.02)
        nn.init.normal_(self.missing_prompts, mean=0.0, std=0.02)

    def forward(
        self,
        text_feat: torch.Tensor,
        audio_feat: torch.Tensor,
        visual_feat: torch.Tensor,
        missing_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if text_feat.dim() != 2 or audio_feat.dim() != 2 or visual_feat.dim() != 2:
            raise ValueError("PromptBank expects modality features with shape [B, D].")
        if missing_mask.dim() != 2 or missing_mask.size(1) != 3:
            raise ValueError("missing_mask must have shape [B, 3].")

        batch_size, d_model = text_feat.shape
        if audio_feat.shape != (batch_size, d_model) or visual_feat.shape != (
            batch_size,
            d_model,
        ):
            raise ValueError("All modality features must share shape [B, D].")
        if d_model != self.d_model:
            raise ValueError(f"Expected D={self.d_model}, got D={d_model}.")

        raw_modalities = torch.stack([text_feat, audio_feat, visual_feat], dim=1)
        available = missing_mask.to(device=text_feat.device, dtype=text_feat.dtype).unsqueeze(-1)

        modality_prompt = self.dropout(self.modality_prompts).unsqueeze(0)
        missing_prompt = self.dropout(self.missing_prompts).unsqueeze(0)
        observed = raw_modalities + modality_prompt
        missing = missing_prompt + modality_prompt

        processed = available * observed + (1.0 - available) * missing

        processed_dict = {
            "text": processed[:, 0, :],
            "audio": processed[:, 1, :],
            "visual": processed[:, 2, :],
        }
        return processed, processed_dict


class TextGuidedCrossAttentionPromptStream(nn.Module):
    """
    Text-query cross-attention over text/audio/visual prompt-completed nodes.
    When text is missing, the prompt bank has already replaced the text node
    with the missing-text prompt plus the text modality prompt.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.num_heads = _get_valid_num_heads(self.d_model, int(num_heads))
        self.attn = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.dropout = nn.Dropout(float(dropout))
        self.norm = nn.LayerNorm(self.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, 2 * self.d_model),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(2 * self.d_model, self.d_model),
            nn.Dropout(float(dropout)),
        )
        self.ffn_norm = nn.LayerNorm(self.d_model)

    def forward(self, modality_nodes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if modality_nodes.dim() != 3 or modality_nodes.size(1) != 3:
            raise ValueError("CrossAttentionPromptStream expects [B, 3, D].")
        if modality_nodes.size(-1) != self.d_model:
            raise ValueError(f"Expected D={self.d_model}, got D={modality_nodes.size(-1)}.")

        query = modality_nodes[:, 0:1, :]
        attn_out, attn_weights = self.attn(
            query,
            modality_nodes,
            modality_nodes,
            need_weights=True,
            average_attn_weights=False,
        )
        z_cross = self.norm(query.squeeze(1) + self.dropout(attn_out.squeeze(1)))
        z_cross = self.ffn_norm(z_cross + self.ffn(z_cross))
        return z_cross, attn_weights


class DualStreamPromptLearningNetwork(nn.Module):
    """
    Text-guided prompt-learning network for missing-modality robustness.

    Forward remains compatible with the repository pipeline:
        forward(x_l, x_a, x_v, missing_mod=None, missing_mask=None, return_aux=False)
    """

    is_dual_stream_prompt = True

    def __init__(self, hyp_params):
        super().__init__()
        self.orig_d_l = int(hyp_params.orig_d_l)
        self.orig_d_a = int(hyp_params.orig_d_a)
        self.orig_d_v = int(hyp_params.orig_d_v)
        self.d_model = int(hyp_params.proj_dim)
        self.output_dim = int(hyp_params.output_dim)
        self.embed_dropout = float(getattr(hyp_params, "embed_dropout", 0.0))
        self.dropout = float(getattr(hyp_params, "out_dropout", 0.0))
        self.prompt_dropout = float(getattr(hyp_params, "prompt_dropout", 0.0))
        self.missing_modality_dropout = float(
            getattr(hyp_params, "missing_modality_dropout", 0.0)
        )

        requested_heads = int(getattr(hyp_params, "cross_attn_heads", 0))
        if requested_heads <= 0:
            requested_heads = int(getattr(hyp_params, "num_heads", 1))
        self.cross_attn_heads = _get_valid_num_heads(self.d_model, requested_heads)

        self.proj_l = nn.Linear(self.orig_d_l, self.d_model, bias=False)
        self.proj_a = nn.Linear(self.orig_d_a, self.d_model, bias=False)
        self.proj_v = nn.Linear(self.orig_d_v, self.d_model, bias=False)
        self.proj_norm_l = nn.LayerNorm(self.d_model)
        self.proj_norm_a = nn.LayerNorm(self.d_model)
        self.proj_norm_v = nn.LayerNorm(self.d_model)
        self.input_dropout = nn.Dropout(self.embed_dropout)

        self.prompt_bank = MissingModalityPromptBank(
            d_model=self.d_model,
            dropout=self.prompt_dropout,
        )
        self.cross_stream = TextGuidedCrossAttentionPromptStream(
            d_model=self.d_model,
            num_heads=self.cross_attn_heads,
            dropout=self.dropout,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.output_dim),
        )

    def _project_and_pool(
        self,
        x: torch.Tensor,
        projector: nn.Linear,
        norm: nn.LayerNorm,
        name: str,
    ) -> torch.Tensor:
        if x.dim() == 2:
            return norm(projector(self.input_dropout(x)))

        if x.dim() != 3:
            raise ValueError(f"{name} must have shape [B, D] or [B, T, D].")

        non_padding = (x.abs().sum(dim=-1) > 0.0).to(dtype=x.dtype)
        h = norm(projector(self.input_dropout(x)))
        denom = non_padding.sum(dim=1, keepdim=True).clamp_min(1.0)
        return torch.sum(h * non_padding.unsqueeze(-1), dim=1) / denom

    def _resolve_missing_mask(
        self,
        missing_mod: Optional[torch.Tensor],
        missing_mask: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if missing_mask is not None:
            mask = missing_mod_to_availability_mask(missing_mask, batch_size, device)
        else:
            mask = missing_mod_to_availability_mask(missing_mod, batch_size, device)

        if (mask.sum(dim=1) < 1.0).any():
            raise ValueError("Every sample must have at least one available modality.")
        return apply_missing_modality_dropout(
            mask,
            self.missing_modality_dropout,
            self.training,
        )

    def forward(
        self,
        x_l: torch.Tensor,
        x_a: torch.Tensor,
        x_v: torch.Tensor,
        missing_mod: Optional[torch.Tensor] = None,
        missing_mask: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ):
        batch_size = x_l.size(0)
        device = x_l.device
        if x_a.size(0) != batch_size or x_v.size(0) != batch_size:
            raise ValueError("All modalities must share the same batch size.")

        text_feat = self._project_and_pool(x_l, self.proj_l, self.proj_norm_l, "text")
        audio_feat = self._project_and_pool(x_a, self.proj_a, self.proj_norm_a, "audio")
        visual_feat = self._project_and_pool(x_v, self.proj_v, self.proj_norm_v, "visual")

        missing_mask = self._resolve_missing_mask(
            missing_mod=missing_mod,
            missing_mask=missing_mask,
            batch_size=batch_size,
            device=device,
        )

        modality_nodes, processed_modalities = self.prompt_bank(
            text_feat,
            audio_feat,
            visual_feat,
            missing_mask,
        )
        z_cross, cross_attn_weights = self.cross_stream(modality_nodes)
        logits = self.classifier(z_cross)

        if not return_aux:
            return logits

        return {
            "logits": logits,
            "z_cross": z_cross,
            "processed_modalities": processed_modalities,
            "missing_mask": missing_mask,
            "cross_attn_weights": cross_attn_weights,
        }
