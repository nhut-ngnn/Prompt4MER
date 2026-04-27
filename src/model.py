import torch
import torch.nn as nn
import torch.nn.functional as F


class PromptGenerator(nn.Module):
    """
    Generate missing modality features from available modalities + prompt.
    """

    def __init__(self, in_channels: int, out_channels: int, target_len: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=1, padding=0),
            nn.GELU(),
        )
        self.target_len = target_len
        self.length_mapper = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        if x.size(-1) != self.target_len:
            if self.length_mapper is None or self.length_mapper.in_features != x.size(-1):
                self.length_mapper = nn.Linear(x.size(-1), self.target_len).to(x.device)
            x = self.length_mapper(x)
        return x


class ModalitySelfAttention(nn.Module):
    """
    Lightweight self-attention block for each modality before joint fusion.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.linear = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        h, attn = self.attn(x, x, x, average_attn_weights=False)
        h = self.linear(h)
        h = self.dropout(h)
        h = self.norm(h + x)
        return h, attn


class Prompt4MSER(nn.Module):
    """
    Prompt-augmented 4M-SER style model for 3 modalities:
    - text
    - audio
    - vision

    Compatible forward signature with the previous training pipeline:
        forward(x_l, x_a, x_v, missing_mod=None, return_aux=False)
    """

    def __init__(self, hyp_params):
        super().__init__()
        self.orig_d_l = hyp_params.orig_d_l
        self.orig_d_a = hyp_params.orig_d_a
        self.orig_d_v = hyp_params.orig_d_v

        self.d_model = int(hyp_params.proj_dim)
        requested_heads = int(hyp_params.num_heads)
        if self.d_model % requested_heads == 0:
            self.num_heads = requested_heads
        else:
            candidates = [h for h in range(requested_heads, 0, -1) if self.d_model % h == 0]
            self.num_heads = candidates[0] if candidates else 1
            print(
                f"Adjusted num_heads from {requested_heads} to {self.num_heads} "
                f"to match d_model={self.d_model}"
            )

        self.dropout = float(hyp_params.out_dropout)
        self.embed_dropout = float(hyp_params.embed_dropout)
        self.prompt_length = int(hyp_params.prompt_length)
        self.output_dim = int(hyp_params.output_dim)
        self.pooling_type = getattr(hyp_params, "fusion_head_output_type", "mean").lower()
        if self.pooling_type not in {"cls", "mean", "max", "concat"}:
            raise ValueError(
                "Invalid fusion_head_output_type. Use one of: cls, mean, max, concat."
            )

        self.llen, self.alen, self.vlen = hyp_params.seq_len

        # 1) modality projection (same style as MULT/Prompt: Conv1d after transpose)
        self.proj_l = nn.Conv1d(
            self.orig_d_l, self.d_model, kernel_size=1, padding=0, bias=False
        )
        self.proj_a = nn.Conv1d(
            self.orig_d_a, self.d_model, kernel_size=1, padding=0, bias=False
        )
        self.proj_v = nn.Conv1d(
            self.orig_d_v, self.d_model, kernel_size=1, padding=0, bias=False
        )

        self.norm_l = nn.LayerNorm(self.d_model)
        self.norm_a = nn.LayerNorm(self.d_model)
        self.norm_v = nn.LayerNorm(self.d_model)

        # 2) prompt parameters
        self.generative_prompt = nn.Parameter(
            torch.zeros(3, self.d_model, self.prompt_length)
        )

        self.promptl_obs = nn.Parameter(torch.zeros(self.d_model, self.llen))
        self.prompta_obs = nn.Parameter(torch.zeros(self.d_model, self.alen))
        self.promptv_obs = nn.Parameter(torch.zeros(self.d_model, self.vlen))

        self.promptl_miss = nn.Parameter(torch.zeros(self.d_model, self.llen))
        self.prompta_miss = nn.Parameter(torch.zeros(self.d_model, self.alen))
        self.promptv_miss = nn.Parameter(torch.zeros(self.d_model, self.vlen))

        self.text_gate = nn.Sequential(
            nn.Conv1d(2 * self.d_model, self.d_model, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )
        self.audio_gate = nn.Sequential(
            nn.Conv1d(2 * self.d_model, self.d_model, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )
        self.vision_gate = nn.Sequential(
            nn.Conv1d(2 * self.d_model, self.d_model, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

        # 7 possible missing states x prompt_len x d_model
        self.missing_type_prompt = nn.Parameter(
            torch.zeros(7, self.prompt_length, self.d_model)
        )

        # 3) prompt generators
        self.av_to_l = PromptGenerator(
            in_channels=self.d_model,
            out_channels=self.d_model,
            target_len=self.llen,
        )
        self.lv_to_a = PromptGenerator(
            in_channels=self.d_model,
            out_channels=self.d_model,
            target_len=self.alen,
        )
        self.la_to_v = PromptGenerator(
            in_channels=self.d_model,
            out_channels=self.d_model,
            target_len=self.vlen,
        )

        self.a_to_l = PromptGenerator(
            in_channels=self.d_model,
            out_channels=self.d_model,
            target_len=self.llen,
        )
        self.v_to_l = PromptGenerator(
            in_channels=self.d_model,
            out_channels=self.d_model,
            target_len=self.llen,
        )
        self.l_to_a = PromptGenerator(
            in_channels=self.d_model,
            out_channels=self.d_model,
            target_len=self.alen,
        )
        self.v_to_a = PromptGenerator(
            in_channels=self.d_model,
            out_channels=self.d_model,
            target_len=self.alen,
        )
        self.l_to_v = PromptGenerator(
            in_channels=self.d_model,
            out_channels=self.d_model,
            target_len=self.vlen,
        )
        self.a_to_v = PromptGenerator(
            in_channels=self.d_model,
            out_channels=self.d_model,
            target_len=self.vlen,
        )

        # 4) per-modality self-attention
        self.text_attention = ModalitySelfAttention(
            self.d_model, self.num_heads, self.dropout
        )
        self.audio_attention = ModalitySelfAttention(
            self.d_model, self.num_heads, self.dropout
        )
        self.vision_attention = ModalitySelfAttention(
            self.d_model, self.num_heads, self.dropout
        )

        # 5) 4M-SER style joint fusion
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True,
        )
        self.fusion_linear = nn.Linear(self.d_model, self.d_model)
        self.fusion_layer_norm = nn.LayerNorm(self.d_model)
        self.dropout_layer = nn.Dropout(self.dropout)

        # classifier head
        self.head_dim = self.d_model * 4 if self.pooling_type == "concat" else self.d_model
        self.proj1 = nn.Linear(self.head_dim, self.head_dim)
        self.proj2 = nn.Linear(self.head_dim, self.head_dim)
        self.out_layer = nn.Linear(self.head_dim, self.output_dim)
        self.classifer = self.out_layer

        self._init_prompt_parameters()
        self.get_proj_matrix()

    def _init_prompt_parameters(self):
        nn.init.normal_(self.generative_prompt, mean=0.0, std=0.02)

        nn.init.normal_(self.promptl_obs, mean=0.0, std=0.02)
        nn.init.normal_(self.prompta_obs, mean=0.0, std=0.02)
        nn.init.normal_(self.promptv_obs, mean=0.0, std=0.02)

        nn.init.normal_(self.promptl_miss, mean=0.0, std=0.02)
        nn.init.normal_(self.prompta_miss, mean=0.0, std=0.02)
        nn.init.normal_(self.promptv_miss, mean=0.0, std=0.02)

        nn.init.normal_(self.missing_type_prompt, mean=0.0, std=0.02)

    def _project_inputs(self, x_l, x_a, x_v):
        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_a = x_a.transpose(1, 2)
        x_v = x_v.transpose(1, 2)

        l = self.proj_l(x_l)
        a = self.proj_a(x_a)
        v = self.proj_v(x_v)
        return l, a, v

    @staticmethod
    def _add_obs_prompt(x: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        return x + prompt.unsqueeze(0)

    @staticmethod
    def _add_miss_prompt(x: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        return x + prompt.unsqueeze(0)

    @staticmethod
    def _missing_observation(prompt: torch.Tensor, batch_size: int) -> torch.Tensor:
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

    def _generate_text(self, a: torch.Tensor = None, v: torch.Tensor = None) -> torch.Tensor:
        ref = a if a is not None else v
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

    def _generate_audio(self, l: torch.Tensor = None, v: torch.Tensor = None) -> torch.Tensor:
        ref = l if l is not None else v
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

    def _generate_vision(self, l: torch.Tensor = None, a: torch.Tensor = None) -> torch.Tensor:
        ref = l if l is not None else a
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

    def _complete_batch(self, l, a, v, missing_mod):
        batch_size = l.size(0)

        ll_list, aa_list, vv_list = [], [], []
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
                l_gen = self._add_miss_prompt(l_gen, self.promptl_miss)
                l_obs = self._missing_observation(self.promptl_miss, li.size(0))
                l_hat = self._blend_features(l_obs, l_gen, self.text_gate)
                a_hat = self._add_obs_prompt(ai, self.prompta_obs)
                v_hat = self._add_obs_prompt(vi, self.promptv_obs)
            elif mode == 1:  # missing audio
                a_gen = self._generate_audio(l=li, v=vi)
                a_gen = self._add_miss_prompt(a_gen, self.prompta_miss)
                a_obs = self._missing_observation(self.prompta_miss, ai.size(0))
                a_hat = self._blend_features(a_obs, a_gen, self.audio_gate)
                l_hat = self._add_obs_prompt(li, self.promptl_obs)
                v_hat = self._add_obs_prompt(vi, self.promptv_obs)
            elif mode == 2:  # missing vision
                v_gen = self._generate_vision(l=li, a=ai)
                v_gen = self._add_miss_prompt(v_gen, self.promptv_miss)
                v_obs = self._missing_observation(self.promptv_miss, vi.size(0))
                v_hat = self._blend_features(v_obs, v_gen, self.vision_gate)
                l_hat = self._add_obs_prompt(li, self.promptl_obs)
                a_hat = self._add_obs_prompt(ai, self.prompta_obs)
            elif mode == 3:  # missing text + audio
                l_gen = self._generate_text(v=vi)
                a_gen = self._generate_audio(v=vi)
                l_gen = self._add_miss_prompt(l_gen, self.promptl_miss)
                a_gen = self._add_miss_prompt(a_gen, self.prompta_miss)
                l_obs = self._missing_observation(self.promptl_miss, li.size(0))
                a_obs = self._missing_observation(self.prompta_miss, ai.size(0))
                l_hat = self._blend_features(l_obs, l_gen, self.text_gate)
                a_hat = self._blend_features(a_obs, a_gen, self.audio_gate)
                v_hat = self._add_obs_prompt(vi, self.promptv_obs)
            elif mode == 4:  # missing text + vision
                l_gen = self._generate_text(a=ai)
                v_gen = self._generate_vision(a=ai)
                l_gen = self._add_miss_prompt(l_gen, self.promptl_miss)
                v_gen = self._add_miss_prompt(v_gen, self.promptv_miss)
                l_obs = self._missing_observation(self.promptl_miss, li.size(0))
                v_obs = self._missing_observation(self.promptv_miss, vi.size(0))
                l_hat = self._blend_features(l_obs, l_gen, self.text_gate)
                v_hat = self._blend_features(v_obs, v_gen, self.vision_gate)
                a_hat = self._add_obs_prompt(ai, self.prompta_obs)
            elif mode == 5:  # missing audio + vision
                a_gen = self._generate_audio(l=li)
                v_gen = self._generate_vision(l=li)
                a_gen = self._add_miss_prompt(a_gen, self.prompta_miss)
                v_gen = self._add_miss_prompt(v_gen, self.promptv_miss)
                a_obs = self._missing_observation(self.prompta_miss, ai.size(0))
                v_obs = self._missing_observation(self.promptv_miss, vi.size(0))
                a_hat = self._blend_features(a_obs, a_gen, self.audio_gate)
                v_hat = self._blend_features(v_obs, v_gen, self.vision_gate)
                l_hat = self._add_obs_prompt(li, self.promptl_obs)
            else:  # 6 = no missing
                l_hat = self._add_obs_prompt(li, self.promptl_obs)
                a_hat = self._add_obs_prompt(ai, self.prompta_obs)
                v_hat = self._add_obs_prompt(vi, self.promptv_obs)

            ll_list.append(l_hat)
            aa_list.append(a_hat)
            vv_list.append(v_hat)
            raw_l_list.append(l_gen)
            raw_a_list.append(a_gen)
            raw_v_list.append(v_gen)

        ll = torch.cat(ll_list, dim=0)
        aa = torch.cat(aa_list, dim=0)
        vv = torch.cat(vv_list, dim=0)
        raw_l = torch.cat(raw_l_list, dim=0)
        raw_a = torch.cat(raw_a_list, dim=0)
        raw_v = torch.cat(raw_v_list, dim=0)

        real_l = torch.cat(real_l_list, dim=0)
        real_a = torch.cat(real_a_list, dim=0)
        real_v = torch.cat(real_v_list, dim=0)

        return ll, aa, vv, raw_l, raw_a, raw_v, real_l, real_a, real_v

    def _normalize_modalities(self, l, a, v):
        l = self.norm_l(l.transpose(1, 2))
        a = self.norm_a(a.transpose(1, 2))
        v = self.norm_v(v.transpose(1, 2))
        return l, a, v

    def get_proj_matrix(self):
        eye = torch.eye(
            self.d_model,
            device=self.missing_type_prompt.device,
            dtype=self.missing_type_prompt.dtype,
        )
        self.mp = eye.unsqueeze(0).repeat(7, 1, 1)

    def get_complete_data(self, x_l, x_a, x_v, missing_mode):
        if x_l.dim() == 2:
            x_l = x_l.unsqueeze(0)
        if x_a.dim() == 2:
            x_a = x_a.unsqueeze(0)
        if x_v.dim() == 2:
            x_v = x_v.unsqueeze(0)

        if x_l.size(1) != self.d_model:
            x_l = self.proj_l(x_l)
        if x_a.size(1) != self.d_model:
            x_a = self.proj_a(x_a)
        if x_v.size(1) != self.d_model:
            x_v = self.proj_v(x_v)

        mode_tensor = torch.tensor(
            [int(missing_mode)], device=x_l.device, dtype=torch.long
        )
        ll, aa, vv, _, _, _, _, _, _ = self._complete_batch(
            x_l, x_a, x_v, mode_tensor
        )
        return ll, aa, vv

    def _pool_fusion(self, fusion_norm):
        if self.pooling_type == "concat":
            l_end = self.llen
            a_end = l_end + self.alen
            v_end = a_end + self.vlen

            text_vec = fusion_norm[:, :l_end, :].mean(dim=1)
            audio_vec = fusion_norm[:, l_end:a_end, :].mean(dim=1)
            vision_vec = fusion_norm[:, a_end:v_end, :].mean(dim=1)
            type_vec = fusion_norm[:, v_end:, :].mean(dim=1)
            return torch.cat([text_vec, audio_vec, vision_vec, type_vec], dim=1)

        if self.pooling_type == "cls":
            return fusion_norm[:, 0, :]
        if self.pooling_type == "max":
            return fusion_norm.max(dim=1).values
        return fusion_norm.mean(dim=1)

    def forward(self, x_l, x_a, x_v, missing_mod=None, return_aux=False):
        batch_size = x_l.size(0)
        device = x_l.device

        if missing_mod is None:
            missing_mod = torch.full((batch_size,), 6, device=device, dtype=torch.long)
        else:
            missing_mod = missing_mod.to(device).long()

        l, a, v = self._project_inputs(x_l, x_a, x_v)  # [B, D, T]
        ll, aa, vv, raw_l, raw_a, raw_v, real_l, real_a, real_v = self._complete_batch(
            l, a, v, missing_mod
        )

        ll, aa, vv = self._normalize_modalities(ll, aa, vv)  # [B, T, D]
        raw_l_n, raw_a_n, raw_v_n = self._normalize_modalities(raw_l, raw_a, raw_v)
        real_l_n, real_a_n, real_v_n = self._normalize_modalities(real_l, real_a, real_v)

        ll, text_attn = self.text_attention(ll)
        aa, audio_attn = self.audio_attention(aa)
        vv, vision_attn = self.vision_attention(vv)

        type_prompt = self.missing_type_prompt[missing_mod]  # [B, P, D]
        fusion_embeddings = torch.cat([ll, aa, vv, type_prompt], dim=1)

        fusion_attention, fusion_attn_output_weights = self.fusion_attention(
            fusion_embeddings,
            fusion_embeddings,
            fusion_embeddings,
            average_attn_weights=False,
        )
        fusion_linear = self.fusion_linear(self.dropout_layer(fusion_attention))
        fusion_norm = self.fusion_layer_norm(fusion_linear + fusion_embeddings)

        pooled = self._pool_fusion(fusion_norm)
        last_hs_proj = self.proj2(
            F.dropout(
                F.relu(self.proj1(pooled)),
                p=self.dropout,
                training=self.training,
            )
        )
        last_hs_proj = last_hs_proj + pooled
        output = self.out_layer(last_hs_proj)

        if not return_aux:
            return output

        return {
            "logits": output,
            "pooled_output": last_hs_proj,
            "completed_features": {
                "text": ll,
                "audio": aa,
                "vision": vv,
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
            "fusion_features": fusion_norm,
            "attentions": {
                "text_attn": text_attn,
                "audio_attn": audio_attn,
                "vision_attn": vision_attn,
                "fusion_attn": fusion_attn_output_weights,
            },
            "missing_mod": missing_mod,
        }

class PromptModel(Prompt4MSER):
    """Keep existing training/eval interfaces while switching to Prompt4MSER."""

    pass


class MULTModel(Prompt4MSER):
    """Legacy checkpoint alias. New code should instantiate Prompt4MSER directly."""

    pass
