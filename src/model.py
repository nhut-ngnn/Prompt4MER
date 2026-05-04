import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple


class PromptGenerator(nn.Module):
    """
    Stable prompt generator for reconstructing a missing modality.

    Input shape:
        x: [B, D, T_source]

    Output shape:
        y: [B, D, T_target]

    Important improvement:
    - The temporal mapper is registered in __init__, not created inside forward().
      This ensures all parameters are visible to the optimizer.
    """

    def __init__(
        self,
        d_model: int,
        source_len: int,
        target_len: int,
        dropout: float = 0.1,
        expansion: int = 2,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.source_len = int(source_len)
        self.target_len = int(target_len)

        hidden_dim = expansion * self.d_model

        self.feature_mlp = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.d_model),
            nn.Dropout(dropout),
        )

        self.feature_norm = nn.LayerNorm(self.d_model)
        self.temporal_mapper = nn.Linear(self.source_len, self.target_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, D, T_source]
        if x.dim() != 3:
            raise ValueError(f"PromptGenerator expects [B, D, T], but got {tuple(x.shape)}")

        # Feature refinement over D.
        h = x.transpose(1, 2)  # [B, T_source, D]
        h = self.feature_norm(h + self.feature_mlp(h))
        h = h.transpose(1, 2)  # [B, D, T_source]

        # Robust fallback when the input sequence length is not exactly the expected source length.
        # This avoids creating a new Linear layer inside forward().
        if h.size(-1) != self.source_len:
            h = F.interpolate(
                h,
                size=self.source_len,
                mode="linear",
                align_corners=False,
            )

        h = self.temporal_mapper(h)  # [B, D, T_target]
        return h


class ModalitySelfAttention(nn.Module):
    """
    Lightweight Transformer-style encoder block for each modality before joint fusion.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float,
        ffn_expansion: int = 4,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)

        hidden_dim = ffn_expansion * d_model
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn_weights = self.attn(
            x,
            x,
            x,
            average_attn_weights=False,
        )
        x = self.attn_norm(x + attn_out)
        x = self.ffn_norm(x + self.ffn(x))
        return x, attn_weights


class TriModalGMUFusion(nn.Module):
    """
    Feature-wise GMU fusion for text, audio, and vision representations.

    Input:
        h_l, h_a, h_v: [B, D]

    Output:
        fused: [B, D]
        gates: [B, 3, D]
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = int(d_model)

        self.text_proj = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        self.audio_proj = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        self.vision_proj = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.Tanh(),
            nn.Dropout(dropout),
        )

        self.gate_layer = nn.Sequential(
            nn.LayerNorm(3 * self.d_model),
            nn.Linear(3 * self.d_model, 3 * self.d_model),
        )

        self.output_norm = nn.LayerNorm(self.d_model)
        self.output_dropout = nn.Dropout(dropout)

    def forward(
        self,
        h_l: torch.Tensor,
        h_a: torch.Tensor,
        h_v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_l_tilde = self.text_proj(h_l)
        h_a_tilde = self.audio_proj(h_a)
        h_v_tilde = self.vision_proj(h_v)

        gate_input = torch.cat([h_l, h_a, h_v], dim=-1)
        gate_logits = self.gate_layer(gate_input)

        batch_size = h_l.size(0)
        gates = gate_logits.view(batch_size, 3, self.d_model)
        gates = torch.softmax(gates, dim=1)

        fused = (
            gates[:, 0, :] * h_l_tilde
            + gates[:, 1, :] * h_a_tilde
            + gates[:, 2, :] * h_v_tilde
        )
        fused = self.output_norm(fused)
        fused = self.output_dropout(fused)

        return fused, gates


class Prompt4MSER(nn.Module):
    """
    Revised Prompt-augmented 4M-SER model for three modalities:
        - text
        - audio
        - vision

    Forward signature remains compatible with the previous pipeline:
        forward(x_l, x_a, x_v, missing_mod=None, return_aux=False)

    Missing mode convention:
        0: missing text
        1: missing audio
        2: missing vision
        3: missing text + audio
        4: missing text + vision
        5: missing audio + vision
        6: no missing
    """

    def __init__(self, hyp_params):
        super().__init__()

        # Original feature dimensions.
        self.orig_d_l = int(hyp_params.orig_d_l)
        self.orig_d_a = int(hyp_params.orig_d_a)
        self.orig_d_v = int(hyp_params.orig_d_v)

        # Model dimensions.
        self.d_model = int(hyp_params.proj_dim)
        self.output_dim = int(hyp_params.output_dim)
        self.dropout = float(getattr(hyp_params, "out_dropout", 0.2))
        self.embed_dropout = float(getattr(hyp_params, "embed_dropout", 0.1))
        self.prompt_length = int(getattr(hyp_params, "prompt_length", 4))

        self.llen, self.alen, self.vlen = [int(x) for x in hyp_params.seq_len]

        requested_heads = int(getattr(hyp_params, "num_heads", 4))
        self.num_heads = self._get_valid_num_heads(self.d_model, requested_heads)

        self.pooling_type = getattr(hyp_params, "fusion_head_output_type", "attn").lower()
        valid_pooling = {"mean", "max", "attn", "concat"}
        if self.pooling_type not in valid_pooling:
            raise ValueError(
                f"Invalid fusion_head_output_type={self.pooling_type}. Use one of {valid_pooling}."
            )

        # 1) Modality projection.
        self.proj_l = nn.Linear(self.orig_d_l, self.d_model, bias=False)
        self.proj_a = nn.Linear(self.orig_d_a, self.d_model, bias=False)
        self.proj_v = nn.Linear(self.orig_d_v, self.d_model, bias=False)

        self.proj_norm_l = nn.LayerNorm(self.d_model)
        self.proj_norm_a = nn.LayerNorm(self.d_model)
        self.proj_norm_v = nn.LayerNorm(self.d_model)

        # 2) Prompt parameters.
        # generative_prompt[m] is used when generating modality m.
        self.generative_prompt = nn.Parameter(
            torch.zeros(3, self.d_model, self.prompt_length)
        )

        # Observed-modality prompts.
        self.promptl_obs = nn.Parameter(torch.zeros(self.d_model, self.llen))
        self.prompta_obs = nn.Parameter(torch.zeros(self.d_model, self.alen))
        self.promptv_obs = nn.Parameter(torch.zeros(self.d_model, self.vlen))

        # Missing-modality prompts.
        self.promptl_miss = nn.Parameter(torch.zeros(self.d_model, self.llen))
        self.prompta_miss = nn.Parameter(torch.zeros(self.d_model, self.alen))
        self.promptv_miss = nn.Parameter(torch.zeros(self.d_model, self.vlen))

        # Missing-state prompt: 7 missing modes x prompt_length x d_model.
        self.missing_type_prompt = nn.Parameter(
            torch.zeros(7, self.prompt_length, self.d_model)
        )

        # 3) Gated blending between missing prompt and generated feature.
        self.text_gate = self._make_gate()
        self.audio_gate = self._make_gate()
        self.vision_gate = self._make_gate()

        # 4) Prompt generators.
        # Source length = prompt length + available modality lengths.
        self.av_to_l = PromptGenerator(
            self.d_model,
            self.prompt_length + self.alen + self.vlen,
            self.llen,
            self.dropout,
        )
        self.lv_to_a = PromptGenerator(
            self.d_model,
            self.prompt_length + self.llen + self.vlen,
            self.alen,
            self.dropout,
        )
        self.la_to_v = PromptGenerator(
            self.d_model,
            self.prompt_length + self.llen + self.alen,
            self.vlen,
            self.dropout,
        )

        self.a_to_l = PromptGenerator(
            self.d_model,
            self.prompt_length + self.alen,
            self.llen,
            self.dropout,
        )
        self.v_to_l = PromptGenerator(
            self.d_model,
            self.prompt_length + self.vlen,
            self.llen,
            self.dropout,
        )
        self.l_to_a = PromptGenerator(
            self.d_model,
            self.prompt_length + self.llen,
            self.alen,
            self.dropout,
        )
        self.v_to_a = PromptGenerator(
            self.d_model,
            self.prompt_length + self.vlen,
            self.alen,
            self.dropout,
        )
        self.l_to_v = PromptGenerator(
            self.d_model,
            self.prompt_length + self.llen,
            self.vlen,
            self.dropout,
        )
        self.a_to_v = PromptGenerator(
            self.d_model,
            self.prompt_length + self.alen,
            self.vlen,
            self.dropout,
        )

        # 5) Per-modality self-attention.
        self.text_attention = ModalitySelfAttention(
            self.d_model,
            self.num_heads,
            self.dropout,
        )
        self.audio_attention = ModalitySelfAttention(
            self.d_model,
            self.num_heads,
            self.dropout,
        )
        self.vision_attention = ModalitySelfAttention(
            self.d_model,
            self.num_heads,
            self.dropout,
        )

        # 6) Modality readout + feature-wise GMU fusion.
        self.modality_pool_score = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.Tanh(),
            nn.Linear(self.d_model, 1),
        )
        self.modality_concat_project = nn.Sequential(
            nn.LayerNorm(2 * self.d_model),
            nn.Linear(2 * self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )

        self.gmu_fusion = TriModalGMUFusion(self.d_model, self.dropout)
        self.head_dim = self.d_model

        # 7) Classifier head.
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.head_dim),
            nn.Linear(self.head_dim, self.head_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.head_dim, self.head_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.head_dim, self.output_dim),
        )

        # Backward-compatible aliases.
        self.out_layer = self.classifier[-1]
        self.classifer = self.out_layer  # keep original typo compatibility if external code refers to it

        # Identity projection matrix buffer, if external code expects self.mp.
        mp = torch.eye(self.d_model).unsqueeze(0).repeat(7, 1, 1)
        self.register_buffer("mp", mp, persistent=False)

        self._init_parameters()

    @staticmethod
    def _get_valid_num_heads(d_model: int, requested_heads: int) -> int:
        if requested_heads <= 0:
            return 1
        if d_model % requested_heads == 0:
            return requested_heads
        for h in range(requested_heads, 0, -1):
            if d_model % h == 0:
                return h
        return 1

    def _make_gate(self) -> nn.Module:
        return nn.Sequential(
            nn.Conv1d(2 * self.d_model, self.d_model, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def _init_parameters(self):
        prompt_std = 0.02
        nn.init.normal_(self.generative_prompt, mean=0.0, std=prompt_std)
        nn.init.normal_(self.promptl_obs, mean=0.0, std=prompt_std)
        nn.init.normal_(self.prompta_obs, mean=0.0, std=prompt_std)
        nn.init.normal_(self.promptv_obs, mean=0.0, std=prompt_std)
        nn.init.normal_(self.promptl_miss, mean=0.0, std=prompt_std)
        nn.init.normal_(self.prompta_miss, mean=0.0, std=prompt_std)
        nn.init.normal_(self.promptv_miss, mean=0.0, std=prompt_std)
        nn.init.normal_(self.missing_type_prompt, mean=0.0, std=prompt_std)

    def _project_inputs(
        self,
        x_l: torch.Tensor,
        x_a: torch.Tensor,
        x_v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Apply embedding dropout to all modalities, not only text.
        x_l = F.dropout(x_l, p=self.embed_dropout, training=self.training)
        x_a = F.dropout(x_a, p=self.embed_dropout, training=self.training)
        x_v = F.dropout(x_v, p=self.embed_dropout, training=self.training)

        # Project [B, T, D_orig] -> [B, T, D_model], then transpose to [B, D_model, T].
        l = self.proj_norm_l(self.proj_l(x_l)).transpose(1, 2)
        a = self.proj_norm_a(self.proj_a(x_a)).transpose(1, 2)
        v = self.proj_norm_v(self.proj_v(x_v)).transpose(1, 2)
        return l, a, v

    @staticmethod
    def _add_prompt(x: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        return x + prompt.unsqueeze(0)

    @staticmethod
    def _prompt_observation(prompt: torch.Tensor, batch_size: int) -> torch.Tensor:
        return prompt.unsqueeze(0).expand(batch_size, -1, -1)

    @staticmethod
    def _blend_features(
        obs_feat: torch.Tensor,
        gen_feat: torch.Tensor,
        gate_layer: nn.Module,
    ) -> torch.Tensor:
        gate_input = torch.cat([obs_feat, gen_feat], dim=1)
        z = gate_layer(gate_input)
        return z * obs_feat + (1.0 - z) * gen_feat

    def _generate_text(
        self,
        a: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        ref = a if a is not None else v
        if ref is None:
            raise ValueError("At least one available modality is required to generate text.")

        pieces = [self.generative_prompt[0].unsqueeze(0).expand(ref.size(0), -1, -1)]
        if a is not None:
            pieces.append(a)
        if v is not None:
            pieces.append(v)

        x = torch.cat(pieces, dim=2)
        if a is not None and v is not None:
            return self.av_to_l(x)
        if a is not None:
            return self.a_to_l(x)
        return self.v_to_l(x)

    def _generate_audio(
        self,
        l: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        ref = l if l is not None else v
        if ref is None:
            raise ValueError("At least one available modality is required to generate audio.")

        pieces = [self.generative_prompt[1].unsqueeze(0).expand(ref.size(0), -1, -1)]
        if l is not None:
            pieces.append(l)
        if v is not None:
            pieces.append(v)

        x = torch.cat(pieces, dim=2)
        if l is not None and v is not None:
            return self.lv_to_a(x)
        if l is not None:
            return self.l_to_a(x)
        return self.v_to_a(x)

    def _generate_vision(
        self,
        l: Optional[torch.Tensor] = None,
        a: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        ref = l if l is not None else a
        if ref is None:
            raise ValueError("At least one available modality is required to generate vision.")

        pieces = [self.generative_prompt[2].unsqueeze(0).expand(ref.size(0), -1, -1)]
        if l is not None:
            pieces.append(l)
        if a is not None:
            pieces.append(a)

        x = torch.cat(pieces, dim=2)
        if l is not None and a is not None:
            return self.la_to_v(x)
        if l is not None:
            return self.l_to_v(x)
        return self.a_to_v(x)

    def _complete_batch(
        self,
        l: torch.Tensor,
        a: torch.Tensor,
        v: torch.Tensor,
        missing_mod: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        batch_size = l.size(0)

        completed_l, completed_a, completed_v = [], [], []
        raw_l_list, raw_a_list, raw_v_list = [], [], []
        real_l_list, real_a_list, real_v_list = [], [], []

        for i in range(batch_size):
            mode = int(missing_mod[i].item())

            li = l[i : i + 1]
            ai = a[i : i + 1]
            vi = v[i : i + 1]

            real_l_list.append(li)
            real_a_list.append(ai)
            real_v_list.append(vi)

            l_gen = torch.zeros_like(li)
            a_gen = torch.zeros_like(ai)
            v_gen = torch.zeros_like(vi)

            if mode == 0:  # missing text
                l_gen = self._generate_text(a=ai, v=vi)
                l_gen = self._add_prompt(l_gen, self.promptl_miss)
                l_obs = self._prompt_observation(self.promptl_miss, li.size(0))
                l_hat = self._blend_features(l_obs, l_gen, self.text_gate)
                a_hat = self._add_prompt(ai, self.prompta_obs)
                v_hat = self._add_prompt(vi, self.promptv_obs)

            elif mode == 1:  # missing audio
                a_gen = self._generate_audio(l=li, v=vi)
                a_gen = self._add_prompt(a_gen, self.prompta_miss)
                a_obs = self._prompt_observation(self.prompta_miss, ai.size(0))
                a_hat = self._blend_features(a_obs, a_gen, self.audio_gate)
                l_hat = self._add_prompt(li, self.promptl_obs)
                v_hat = self._add_prompt(vi, self.promptv_obs)

            elif mode == 2:  # missing vision
                v_gen = self._generate_vision(l=li, a=ai)
                v_gen = self._add_prompt(v_gen, self.promptv_miss)
                v_obs = self._prompt_observation(self.promptv_miss, vi.size(0))
                v_hat = self._blend_features(v_obs, v_gen, self.vision_gate)
                l_hat = self._add_prompt(li, self.promptl_obs)
                a_hat = self._add_prompt(ai, self.prompta_obs)

            elif mode == 3:  # missing text + audio
                l_gen = self._generate_text(v=vi)
                a_gen = self._generate_audio(v=vi)
                l_gen = self._add_prompt(l_gen, self.promptl_miss)
                a_gen = self._add_prompt(a_gen, self.prompta_miss)
                l_obs = self._prompt_observation(self.promptl_miss, li.size(0))
                a_obs = self._prompt_observation(self.prompta_miss, ai.size(0))
                l_hat = self._blend_features(l_obs, l_gen, self.text_gate)
                a_hat = self._blend_features(a_obs, a_gen, self.audio_gate)
                v_hat = self._add_prompt(vi, self.promptv_obs)

            elif mode == 4:  # missing text + vision
                l_gen = self._generate_text(a=ai)
                v_gen = self._generate_vision(a=ai)
                l_gen = self._add_prompt(l_gen, self.promptl_miss)
                v_gen = self._add_prompt(v_gen, self.promptv_miss)
                l_obs = self._prompt_observation(self.promptl_miss, li.size(0))
                v_obs = self._prompt_observation(self.promptv_miss, vi.size(0))
                l_hat = self._blend_features(l_obs, l_gen, self.text_gate)
                v_hat = self._blend_features(v_obs, v_gen, self.vision_gate)
                a_hat = self._add_prompt(ai, self.prompta_obs)

            elif mode == 5:  # missing audio + vision
                a_gen = self._generate_audio(l=li)
                v_gen = self._generate_vision(l=li)
                a_gen = self._add_prompt(a_gen, self.prompta_miss)
                v_gen = self._add_prompt(v_gen, self.promptv_miss)
                a_obs = self._prompt_observation(self.prompta_miss, ai.size(0))
                v_obs = self._prompt_observation(self.promptv_miss, vi.size(0))
                a_hat = self._blend_features(a_obs, a_gen, self.audio_gate)
                v_hat = self._blend_features(v_obs, v_gen, self.vision_gate)
                l_hat = self._add_prompt(li, self.promptl_obs)

            else:  # 6 = no missing
                l_hat = self._add_prompt(li, self.promptl_obs)
                a_hat = self._add_prompt(ai, self.prompta_obs)
                v_hat = self._add_prompt(vi, self.promptv_obs)

            completed_l.append(l_hat)
            completed_a.append(a_hat)
            completed_v.append(v_hat)

            raw_l_list.append(l_gen)
            raw_a_list.append(a_gen)
            raw_v_list.append(v_gen)

        completed_l = torch.cat(completed_l, dim=0)
        completed_a = torch.cat(completed_a, dim=0)
        completed_v = torch.cat(completed_v, dim=0)

        raw_l = torch.cat(raw_l_list, dim=0)
        raw_a = torch.cat(raw_a_list, dim=0)
        raw_v = torch.cat(raw_v_list, dim=0)

        real_l = torch.cat(real_l_list, dim=0)
        real_a = torch.cat(real_a_list, dim=0)
        real_v = torch.cat(real_v_list, dim=0)

        return completed_l, completed_a, completed_v, raw_l, raw_a, raw_v, real_l, real_a, real_v

    @staticmethod
    def _to_sequence_first(
        l: torch.Tensor,
        a: torch.Tensor,
        v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # [B, D, T] -> [B, T, D]
        return l.transpose(1, 2), a.transpose(1, 2), v.transpose(1, 2)

    def _project_complete_input(
        self,
        x: torch.Tensor,
        projector: nn.Linear,
        orig_dim: int,
    ) -> torch.Tensor:
        """
        Convert input to [B, D_model, T] for get_complete_data().
        Supports either projected or original feature input.
        """
        if x.size(1) == self.d_model:
            return x
        if x.size(-1) == self.d_model:
            return x.transpose(1, 2)
        if x.size(-1) == orig_dim:
            return projector(x).transpose(1, 2)
        if x.size(1) == orig_dim:
            return projector(x.transpose(1, 2)).transpose(1, 2)

        raise ValueError(
            f"Cannot project tensor with shape {tuple(x.shape)}. "
            f"Expected feature dimension {orig_dim} or {self.d_model}."
        )

    def get_proj_matrix(self):
        """
        Backward-compatible method. The matrix is now a registered buffer.
        """
        return self.mp

    def get_complete_data(
        self,
        x_l: torch.Tensor,
        x_a: torch.Tensor,
        x_v: torch.Tensor,
        missing_mode: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Complete one sample or one batch for a given missing mode.
        Returns tensors in [B, D_model, T] format.
        """
        if x_l.dim() == 2:
            x_l = x_l.unsqueeze(0)
        if x_a.dim() == 2:
            x_a = x_a.unsqueeze(0)
        if x_v.dim() == 2:
            x_v = x_v.unsqueeze(0)

        x_l = self._project_complete_input(x_l, self.proj_l, self.orig_d_l)
        x_a = self._project_complete_input(x_a, self.proj_a, self.orig_d_a)
        x_v = self._project_complete_input(x_v, self.proj_v, self.orig_d_v)

        batch_size = x_l.size(0)
        mode_tensor = torch.full(
            (batch_size,),
            int(missing_mode),
            device=x_l.device,
            dtype=torch.long,
        )

        completed_l, completed_a, completed_v, *_ = self._complete_batch(
            x_l,
            x_a,
            x_v,
            mode_tensor,
        )
        return completed_l, completed_a, completed_v

    def _pool_modality(self, modality_seq: torch.Tensor) -> torch.Tensor:
        if self.pooling_type == "mean":
            return modality_seq.mean(dim=1)

        if self.pooling_type == "max":
            return modality_seq.max(dim=1).values

        if self.pooling_type == "concat":
            mean_pool = modality_seq.mean(dim=1)
            max_pool = modality_seq.max(dim=1).values
            return self.modality_concat_project(torch.cat([mean_pool, max_pool], dim=-1))

        # Default option: attention pooling over each modality sequence.
        score = self.modality_pool_score(modality_seq)  # [B, T, 1]
        weight = torch.softmax(score, dim=1)       # [B, T, 1]
        pooled = torch.sum(weight * modality_seq, dim=1)
        return pooled

    def forward(
        self,
        x_l: torch.Tensor,
        x_a: torch.Tensor,
        x_v: torch.Tensor,
        missing_mod: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ):
        batch_size = x_l.size(0)
        device = x_l.device

        if missing_mod is None:
            missing_mod = torch.full(
                (batch_size,),
                6,
                device=device,
                dtype=torch.long,
            )
        else:
            missing_mod = missing_mod.to(device=device, dtype=torch.long)

        # 1) Project inputs to [B, D, T].
        l, a, v = self._project_inputs(x_l, x_a, x_v)

        # 2) Generate and complete missing modality features.
        completed_l, completed_a, completed_v, raw_l, raw_a, raw_v, real_l, real_a, real_v = self._complete_batch(
            l,
            a,
            v,
            missing_mod,
        )

        # 3) Convert [B, D, T] -> [B, T, D].
        completed_l, completed_a, completed_v = self._to_sequence_first(
            completed_l,
            completed_a,
            completed_v,
        )
        raw_l_n, raw_a_n, raw_v_n = self._to_sequence_first(raw_l, raw_a, raw_v)
        real_l_n, real_a_n, real_v_n = self._to_sequence_first(real_l, real_a, real_v)

        # 4) Per-modality encoding.
        completed_l, text_attn = self.text_attention(completed_l)
        completed_a, audio_attn = self.audio_attention(completed_a)
        completed_v, vision_attn = self.vision_attention(completed_v)

        # 5) Pool each modality and fuse with feature-wise tri-modal GMU.
        text_pooled = self._pool_modality(completed_l)
        audio_pooled = self._pool_modality(completed_a)
        vision_pooled = self._pool_modality(completed_v)
        pooled, fusion_gates = self.gmu_fusion(text_pooled, audio_pooled, vision_pooled)

        # 6) Classify.
        logits = self.classifier(pooled)

        if not return_aux:
            return logits

        return {
            "logits": logits,
            "pooled_output": pooled,
            "completed_features": {
                "text": completed_l,
                "audio": completed_a,
                "vision": completed_v,
            },
            "raw_generated_features": {
                "text": raw_l_n,
                "audio": raw_a_n,
                "vision": raw_v_n,
            },
            "real_features": {
                "text": real_l_n,
                "audio": real_a_n,
                "vision": real_v_n,
            },
            "modality_pooled_features": {
                "text": text_pooled,
                "audio": audio_pooled,
                "vision": vision_pooled,
            },
            "fusion_features": pooled,
            "fusion_gates": fusion_gates,
            "attentions": {
                "text_attn": text_attn,
                "audio_attn": audio_attn,
                "vision_attn": vision_attn,
            },
            "missing_mod": missing_mod,
        }


def prompt4mser_loss(
    outputs,
    labels,
    missing_mod,
    class_weights=None,
    lambda_rec=0.1,
    lambda_cos=0.05,
):
    """
    Classification + reconstruction + cosine loss.

    Required:
        outputs = model(..., return_aux=True)
    """

    logits = outputs["logits"]
    labels = labels.to(device=logits.device, dtype=torch.long)
    missing_mod = missing_mod.to(device=logits.device, dtype=torch.long)
    if class_weights is not None:
        class_weights = class_weights.to(device=logits.device)

    loss_cls = F.cross_entropy(
        logits,
        labels,
        weight=class_weights,
    )

    raw = outputs["raw_generated_features"]
    real = outputs["real_features"]

    rec_loss = logits.new_tensor(0.0)
    cos_loss = logits.new_tensor(0.0)
    count = 0

    missing_map = {
        "text": [0, 3, 4],
        "audio": [1, 3, 5],
        "vision": [2, 4, 5],
    }

    for mod_name, modes in missing_map.items():
        mask = torch.zeros_like(missing_mod, dtype=torch.bool)

        for m in modes:
            mask = mask | (missing_mod == m)

        if mask.any():
            gen_feat = raw[mod_name][mask]
            real_feat = real[mod_name][mask].detach()

            rec_loss = rec_loss + F.smooth_l1_loss(gen_feat, real_feat)

            gen_pool = gen_feat.mean(dim=1)
            real_pool = real_feat.mean(dim=1)

            cos_loss = cos_loss + (
                1.0 - F.cosine_similarity(gen_pool, real_pool, dim=-1).mean()
            )

            count += 1

    if count > 0:
        rec_loss = rec_loss / count
        cos_loss = cos_loss / count

    total_loss = loss_cls + lambda_rec * rec_loss + lambda_cos * cos_loss

    return {
        "loss": total_loss,
        "loss_cls": loss_cls.detach(),
        "loss_rec": rec_loss.detach(),
        "loss_cos": cos_loss.detach(),
    }


class Prompt4MSERLoss(nn.Module):
    """
    Classification + reconstruction + cosine loss.

    Use this loss only when model(..., return_aux=True) is used.
    """

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        lambda_rec: float = 0.10,
        lambda_cos: float = 0.05,
        use_focal: bool = False,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.lambda_rec = float(lambda_rec)
        self.lambda_cos = float(lambda_cos)
        self.use_focal = bool(use_focal)
        self.focal_gamma = float(focal_gamma)

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None

    def _classification_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_focal:
            return F.cross_entropy(logits, labels, weight=self.class_weights)

        ce = F.cross_entropy(
            logits,
            labels,
            weight=self.class_weights,
            reduction="none",
        )
        pt = torch.exp(-ce)
        return (((1.0 - pt) ** self.focal_gamma) * ce).mean()

    def forward(
        self,
        outputs: Dict[str, Any],
        labels: torch.Tensor,
        missing_mod: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        logits = outputs["logits"]
        if missing_mod is None:
            missing_mod = outputs.get("missing_mod", None)
        if missing_mod is None:
            raise ValueError("missing_mod must be provided or included in model outputs.")

        missing_mod = missing_mod.to(device=logits.device, dtype=torch.long)
        labels = labels.to(device=logits.device, dtype=torch.long)

        loss_cls = self._classification_loss(logits, labels)

        raw = outputs["raw_generated_features"]
        real = outputs["real_features"]

        rec_loss = logits.new_tensor(0.0)
        cos_loss = logits.new_tensor(0.0)
        count = 0

        missing_map = {
            "text": (0, 3, 4),
            "audio": (1, 3, 5),
            "vision": (2, 4, 5),
        }

        for mod_name, modes in missing_map.items():
            mask = torch.zeros_like(missing_mod, dtype=torch.bool)
            for m in modes:
                mask = mask | (missing_mod == m)

            if mask.any():
                gen_feat = raw[mod_name][mask]
                real_feat = real[mod_name][mask].detach()

                rec_loss = rec_loss + F.smooth_l1_loss(gen_feat, real_feat)

                gen_pool = gen_feat.mean(dim=1)
                real_pool = real_feat.mean(dim=1)
                cos_loss = cos_loss + (
                    1.0 - F.cosine_similarity(gen_pool, real_pool, dim=-1).mean()
                )
                count += 1

        if count > 0:
            rec_loss = rec_loss / count
            cos_loss = cos_loss / count

        total_loss = loss_cls + self.lambda_rec * rec_loss + self.lambda_cos * cos_loss

        return {
            "loss": total_loss,
            "loss_cls": loss_cls.detach(),
            "loss_rec": rec_loss.detach(),
            "loss_cos": cos_loss.detach(),
        }


def sample_missing_mod(
    batch_size: int,
    device: torch.device,
    epoch: Optional[int] = None,
    max_epoch: Optional[int] = None,
    max_missing_prob: float = 0.50,
    double_missing_prob: float = 0.25,
) -> torch.Tensor:
    """
    Curriculum-style random missing-modality sampler.

    Returns:
        missing_mod: [B]

    Missing mode convention:
        0: missing text
        1: missing audio
        2: missing vision
        3: missing text + audio
        4: missing text + vision
        5: missing audio + vision
        6: no missing
    """
    if epoch is None or max_epoch is None or max_epoch <= 0:
        p_missing = max_missing_prob
    else:
        progress = min(max(float(epoch) / float(max_epoch), 0.0), 1.0)
        p_missing = min(max_missing_prob, 0.10 + progress * (max_missing_prob - 0.10))

    missing_mod = torch.full(
        (batch_size,),
        6,
        device=device,
        dtype=torch.long,
    )

    missing_mask = torch.rand(batch_size, device=device) < p_missing
    n_missing = int(missing_mask.sum().item())

    if n_missing == 0:
        return missing_mod

    use_double = torch.rand(n_missing, device=device) < double_missing_prob

    sampled_modes = torch.empty(n_missing, device=device, dtype=torch.long)

    # Single missing: 0, 1, 2.
    n_single = int((~use_double).sum().item())
    if n_single > 0:
        sampled_modes[~use_double] = torch.randint(
            low=0,
            high=3,
            size=(n_single,),
            device=device,
        )

    # Double missing: 3, 4, 5.
    n_double = int(use_double.sum().item())
    if n_double > 0:
        sampled_modes[use_double] = torch.randint(
            low=3,
            high=6,
            size=(n_double,),
            device=device,
        )

    missing_mod[missing_mask] = sampled_modes
    return missing_mod


def compute_class_weights(class_counts, device=None) -> torch.Tensor:
    """
    Compute balanced class weights for CrossEntropyLoss.

    class_counts can be a list, tuple, numpy array, or torch tensor.
    """
    counts = torch.as_tensor(class_counts, dtype=torch.float)
    counts = torch.clamp(counts, min=1.0)
    weights = counts.sum() / (len(counts) * counts)

    if device is not None:
        weights = weights.to(device)
    return weights


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
