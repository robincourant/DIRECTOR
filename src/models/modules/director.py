import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
from einops import rearrange

from typing import Optional, List
from torchtyping import TensorType
from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1

allow_ops_in_compiled_graph()

batch_size, num_cond_feats = None, None


class FusedMLP(nn.Sequential):
    def __init__(
        self,
        dim_model: int,
        dropout: float,
        activation: nn.Module,
        hidden_layer_multiplier: int = 4,
        bias: bool = True,
    ):
        super().__init__(
            nn.Linear(dim_model, dim_model * hidden_layer_multiplier, bias=bias),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(dim_model * hidden_layer_multiplier, dim_model, bias=bias),
        )


def _cast_if_autocast_enabled(tensor):
    if torch.is_autocast_enabled():
        if tensor.device.type == "cuda":
            dtype = torch.get_autocast_gpu_dtype()
        elif tensor.device.type == "cpu":
            dtype = torch.get_autocast_cpu_dtype()
        else:
            raise NotImplementedError()
        return tensor.to(dtype=dtype)
    return tensor


class LayerNorm16Bits(torch.nn.LayerNorm):
    """
    16-bit friendly version of torch.nn.LayerNorm
    """

    def __init__(
        self,
        normalized_shape,
        eps=1e-06,
        elementwise_affine=True,
        device=None,
        dtype=None,
    ):
        super().__init__(
            normalized_shape=normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
            device=device,
            dtype=dtype,
        )

    def forward(self, x):
        module_device = x.device
        downcast_x = _cast_if_autocast_enabled(x)
        downcast_weight = (
            _cast_if_autocast_enabled(self.weight)
            if self.weight is not None
            else self.weight
        )
        downcast_bias = (
            _cast_if_autocast_enabled(self.bias) if self.bias is not None else self.bias
        )
        with torch.autocast(enabled=False, device_type=module_device.type):
            return nn.functional.layer_norm(
                downcast_x,
                self.normalized_shape,
                downcast_weight,
                downcast_bias,
                self.eps,
            )


class StochatichDepth(nn.Module):
    def __init__(self, p: float):
        super().__init__()
        self.survival_prob = 1.0 - p

    def forward(self, x: Tensor) -> Tensor:
        if self.training and self.survival_prob < 1:
            mask = (
                torch.empty(x.shape[0], 1, 1, device=x.device).uniform_()
                + self.survival_prob
            )
            mask = mask.floor()
            if self.survival_prob > 0:
                mask = mask / self.survival_prob
            return x * mask
        else:
            return x


class CrossAttentionOp(nn.Module):
    def __init__(
        self, attention_dim, num_heads, dim_q, dim_kv, use_biases=True, is_sa=False
    ):
        super().__init__()
        self.dim_q = dim_q
        self.dim_kv = dim_kv
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.use_biases = use_biases
        self.is_sa = is_sa
        if self.is_sa:
            self.qkv = nn.Linear(dim_q, attention_dim * 3, bias=use_biases)
        else:
            self.q = nn.Linear(dim_q, attention_dim, bias=use_biases)
            self.kv = nn.Linear(dim_kv, attention_dim * 2, bias=use_biases)
        self.out = nn.Linear(attention_dim, dim_q, bias=use_biases)

    def forward(self, x_to, x_from=None, attention_mask=None):
        if x_from is None:
            x_from = x_to
        if self.is_sa:
            q, k, v = self.qkv(x_to).chunk(3, dim=-1)
        else:
            q = self.q(x_to)
            k, v = self.kv(x_from).chunk(2, dim=-1)
        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1)
        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask
        )
        x = rearrange(x, "b h n d -> b n (h d)")
        x = self.out(x)
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        dim_q: int,
        dim_kv: int,
        num_heads: int,
        attention_dim: int = 0,
        mlp_multiplier: int = 4,
        dropout: float = 0.0,
        stochastic_depth: float = 0.0,
        use_biases: bool = True,
        retrieve_attention_scores: bool = False,
        use_layernorm16: bool = True,
    ):
        super().__init__()
        layer_norm = (
            nn.LayerNorm
            if not use_layernorm16 or retrieve_attention_scores
            else LayerNorm16Bits
        )
        self.retrieve_attention_scores = retrieve_attention_scores
        self.initial_to_ln = layer_norm(dim_q, eps=1e-6)
        attention_dim = min(dim_q, dim_kv) if attention_dim == 0 else attention_dim
        self.ca = CrossAttentionOp(
            attention_dim, num_heads, dim_q, dim_kv, is_sa=False, use_biases=use_biases
        )
        self.ca_stochastic_depth = StochatichDepth(stochastic_depth)
        self.middle_ln = layer_norm(dim_q, eps=1e-6)
        self.ffn = FusedMLP(
            dim_model=dim_q,
            dropout=dropout,
            activation=nn.GELU,
            hidden_layer_multiplier=mlp_multiplier,
            bias=use_biases,
        )
        self.ffn_stochastic_depth = StochatichDepth(stochastic_depth)

    def forward(
        self,
        to_tokens: Tensor,
        from_tokens: Tensor,
        to_token_mask: Optional[Tensor] = None,
        from_token_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if to_token_mask is None and from_token_mask is None:
            attention_mask = None
        else:
            if to_token_mask is None:
                to_token_mask = torch.ones(
                    to_tokens.shape[0],
                    to_tokens.shape[1],
                    dtype=torch.bool,
                    device=to_tokens.device,
                )
            if from_token_mask is None:
                from_token_mask = torch.ones(
                    from_tokens.shape[0],
                    from_tokens.shape[1],
                    dtype=torch.bool,
                    device=from_tokens.device,
                )
            attention_mask = from_token_mask.unsqueeze(1) * to_token_mask.unsqueeze(2)
        attention_output = self.ca(
            self.initial_to_ln(to_tokens),
            from_tokens,
            attention_mask=attention_mask,
        )
        to_tokens = to_tokens + self.ca_stochastic_depth(attention_output)
        to_tokens = to_tokens + self.ffn_stochastic_depth(
            self.ffn(self.middle_ln(to_tokens))
        )
        return to_tokens


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim_qkv: int,
        num_heads: int,
        attention_dim: int = 0,
        mlp_multiplier: int = 4,
        dropout: float = 0.0,
        stochastic_depth: float = 0.0,
        use_biases: bool = True,
        use_layer_scale: bool = False,
        layer_scale_value: float = 0.0,
        use_layernorm16: bool = True,
    ):
        super().__init__()
        layer_norm = LayerNorm16Bits if use_layernorm16 else nn.LayerNorm
        self.initial_ln = layer_norm(dim_qkv, eps=1e-6)
        attention_dim = dim_qkv if attention_dim == 0 else attention_dim
        self.sa = CrossAttentionOp(
            attention_dim,
            num_heads,
            dim_qkv,
            dim_qkv,
            is_sa=True,
            use_biases=use_biases,
        )
        self.sa_stochastic_depth = StochatichDepth(stochastic_depth)
        self.middle_ln = layer_norm(dim_qkv, eps=1e-6)
        self.ffn = FusedMLP(
            dim_model=dim_qkv,
            dropout=dropout,
            activation=nn.GELU,
            hidden_layer_multiplier=mlp_multiplier,
            bias=use_biases,
        )
        self.ffn_stochastic_depth = StochatichDepth(stochastic_depth)
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                torch.ones(dim_qkv) * layer_scale_value, requires_grad=True
            )
            self.layer_scale_2 = nn.Parameter(
                torch.ones(dim_qkv) * layer_scale_value, requires_grad=True
            )

    def forward(
        self,
        tokens: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
    ):
        if token_mask is None:
            attention_mask = None
        else:
            attention_mask = token_mask.unsqueeze(1) * torch.ones(
                tokens.shape[0],
                tokens.shape[1],
                1,
                dtype=torch.bool,
                device=tokens.device,
            )
        attention_output = self.sa(
            self.initial_ln(tokens),
            attention_mask=attention_mask,
        )
        if self.use_layer_scale:
            tokens = tokens + self.sa_stochastic_depth(
                self.layer_scale_1 * attention_output
            )
            tokens = tokens + self.ffn_stochastic_depth(
                self.layer_scale_2 * self.ffn(self.middle_ln(tokens))
            )
        else:
            tokens = tokens + self.sa_stochastic_depth(attention_output)
            tokens = tokens + self.ffn_stochastic_depth(
                self.ffn(self.middle_ln(tokens))
            )
        return tokens


class AdaLNSABlock(nn.Module):
    def __init__(
        self,
        dim_qkv: int,
        dim_cond: int,
        num_heads: int,
        attention_dim: int = 0,
        mlp_multiplier: int = 4,
        dropout: float = 0.0,
        stochastic_depth: float = 0.0,
        use_biases: bool = True,
        use_layer_scale: bool = False,
        layer_scale_value: float = 0.1,
        use_layernorm16: bool = True,
    ):
        super().__init__()
        layer_norm = LayerNorm16Bits if use_layernorm16 else nn.LayerNorm
        self.initial_ln = layer_norm(dim_qkv, eps=1e-6, elementwise_affine=False)
        attention_dim = dim_qkv if attention_dim == 0 else attention_dim
        self.adaln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim_cond, dim_qkv * 6, bias=use_biases),
        )
        # Zero init
        nn.init.zeros_(self.adaln_modulation[1].weight)
        nn.init.zeros_(self.adaln_modulation[1].bias)

        self.sa = CrossAttentionOp(
            attention_dim,
            num_heads,
            dim_qkv,
            dim_qkv,
            is_sa=True,
            use_biases=use_biases,
        )
        self.sa_stochastic_depth = StochatichDepth(stochastic_depth)
        self.middle_ln = layer_norm(dim_qkv, eps=1e-6, elementwise_affine=False)
        self.ffn = FusedMLP(
            dim_model=dim_qkv,
            dropout=dropout,
            activation=nn.GELU,
            hidden_layer_multiplier=mlp_multiplier,
            bias=use_biases,
        )
        self.ffn_stochastic_depth = StochatichDepth(stochastic_depth)
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                torch.ones(dim_qkv) * layer_scale_value, requires_grad=True
            )
            self.layer_scale_2 = nn.Parameter(
                torch.ones(dim_qkv) * layer_scale_value, requires_grad=True
            )

    def forward(
        self,
        tokens: torch.Tensor,
        cond: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
    ):
        if token_mask is None:
            attention_mask = None
        else:
            attention_mask = token_mask.unsqueeze(1) * torch.ones(
                tokens.shape[0],
                tokens.shape[1],
                1,
                dtype=torch.bool,
                device=tokens.device,
            )
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaln_modulation(cond).chunk(6, dim=-1)
        )
        attention_output = self.sa(
            modulate_shift_and_scale(self.initial_ln(tokens), shift_msa, scale_msa),
            attention_mask=attention_mask,
        )
        if self.use_layer_scale:
            tokens = tokens + self.sa_stochastic_depth(
                gate_msa.unsqueeze(1) * self.layer_scale_1 * attention_output
            )
            tokens = tokens + self.ffn_stochastic_depth(
                gate_mlp.unsqueeze(1)
                * self.layer_scale_2
                * self.ffn(
                    modulate_shift_and_scale(
                        self.middle_ln(tokens), shift_mlp, scale_mlp
                    )
                )
            )
        else:
            tokens = tokens + gate_msa.unsqueeze(1) * self.sa_stochastic_depth(
                attention_output
            )
            tokens = tokens + self.ffn_stochastic_depth(
                gate_mlp.unsqueeze(1)
                * self.ffn(
                    modulate_shift_and_scale(
                        self.middle_ln(tokens), shift_mlp, scale_mlp
                    )
                )
            )
        return tokens


class CrossAttentionSABlock(nn.Module):
    def __init__(
        self,
        dim_qkv: int,
        dim_cond: int,
        num_heads: int,
        attention_dim: int = 0,
        mlp_multiplier: int = 4,
        dropout: float = 0.0,
        stochastic_depth: float = 0.0,
        use_biases: bool = True,
        use_layer_scale: bool = False,
        layer_scale_value: float = 0.0,
        use_layernorm16: bool = True,
    ):
        super().__init__()
        layer_norm = LayerNorm16Bits if use_layernorm16 else nn.LayerNorm
        attention_dim = dim_qkv if attention_dim == 0 else attention_dim
        self.ca = CrossAttentionOp(
            attention_dim,
            num_heads,
            dim_qkv,
            dim_cond,
            is_sa=False,
            use_biases=use_biases,
        )
        self.ca_stochastic_depth = StochatichDepth(stochastic_depth)
        self.ca_ln = layer_norm(dim_qkv, eps=1e-6)

        self.initial_ln = layer_norm(dim_qkv, eps=1e-6)
        attention_dim = dim_qkv if attention_dim == 0 else attention_dim

        self.sa = CrossAttentionOp(
            attention_dim,
            num_heads,
            dim_qkv,
            dim_qkv,
            is_sa=True,
            use_biases=use_biases,
        )
        self.sa_stochastic_depth = StochatichDepth(stochastic_depth)
        self.middle_ln = layer_norm(dim_qkv, eps=1e-6)
        self.ffn = FusedMLP(
            dim_model=dim_qkv,
            dropout=dropout,
            activation=nn.GELU,
            hidden_layer_multiplier=mlp_multiplier,
            bias=use_biases,
        )
        self.ffn_stochastic_depth = StochatichDepth(stochastic_depth)
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                torch.ones(dim_qkv) * layer_scale_value, requires_grad=True
            )
            self.layer_scale_2 = nn.Parameter(
                torch.ones(dim_qkv) * layer_scale_value, requires_grad=True
            )

    def forward(
        self,
        tokens: torch.Tensor,
        cond: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
        cond_mask: Optional[torch.Tensor] = None,
    ):
        if cond_mask is None:
            cond_attention_mask = None
        else:
            cond_attention_mask = torch.ones(
                cond.shape[0],
                1,
                cond.shape[1],
                dtype=torch.bool,
                device=tokens.device,
            ) * token_mask.unsqueeze(2)
        if token_mask is None:
            attention_mask = None
        else:
            attention_mask = token_mask.unsqueeze(1) * torch.ones(
                tokens.shape[0],
                tokens.shape[1],
                1,
                dtype=torch.bool,
                device=tokens.device,
            )
        ca_output = self.ca(
            self.ca_ln(tokens),
            cond,
            attention_mask=cond_attention_mask,
        )
        ca_output = torch.nan_to_num(
            ca_output, nan=0.0, posinf=0.0, neginf=0.0
        )  # Needed as some tokens get attention from no token so Nan
        tokens = tokens + self.ca_stochastic_depth(ca_output)
        attention_output = self.sa(
            self.initial_ln(tokens),
            attention_mask=attention_mask,
        )
        if self.use_layer_scale:
            tokens = tokens + self.sa_stochastic_depth(
                self.layer_scale_1 * attention_output
            )
            tokens = tokens + self.ffn_stochastic_depth(
                self.layer_scale_2 * self.ffn(self.middle_ln(tokens))
            )
        else:
            tokens = tokens + self.sa_stochastic_depth(attention_output)
            tokens = tokens + self.ffn_stochastic_depth(
                self.ffn(self.middle_ln(tokens))
            )
        return tokens


class CAAdaLNSABlock(nn.Module):
    def __init__(
        self,
        dim_qkv: int,
        dim_cond: int,
        num_heads: int,
        attention_dim: int = 0,
        mlp_multiplier: int = 4,
        dropout: float = 0.0,
        stochastic_depth: float = 0.0,
        use_biases: bool = True,
        use_layer_scale: bool = False,
        layer_scale_value: float = 0.1,
        use_layernorm16: bool = True,
    ):
        super().__init__()
        layer_norm = LayerNorm16Bits if use_layernorm16 else nn.LayerNorm
        self.ca = CrossAttentionOp(
            attention_dim,
            num_heads,
            dim_qkv,
            dim_cond,
            is_sa=False,
            use_biases=use_biases,
        )
        self.ca_stochastic_depth = StochatichDepth(stochastic_depth)
        self.ca_ln = layer_norm(dim_qkv, eps=1e-6)
        self.initial_ln = layer_norm(dim_qkv, eps=1e-6)
        attention_dim = dim_qkv if attention_dim == 0 else attention_dim
        self.adaln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim_cond, dim_qkv * 6, bias=use_biases),
        )
        # Zero init
        nn.init.zeros_(self.adaln_modulation[1].weight)
        nn.init.zeros_(self.adaln_modulation[1].bias)

        self.sa = CrossAttentionOp(
            attention_dim,
            num_heads,
            dim_qkv,
            dim_qkv,
            is_sa=True,
            use_biases=use_biases,
        )
        self.sa_stochastic_depth = StochatichDepth(stochastic_depth)
        self.middle_ln = layer_norm(dim_qkv, eps=1e-6)
        self.ffn = FusedMLP(
            dim_model=dim_qkv,
            dropout=dropout,
            activation=nn.GELU,
            hidden_layer_multiplier=mlp_multiplier,
            bias=use_biases,
        )
        self.ffn_stochastic_depth = StochatichDepth(stochastic_depth)
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                torch.ones(dim_qkv) * layer_scale_value, requires_grad=True
            )
            self.layer_scale_2 = nn.Parameter(
                torch.ones(dim_qkv) * layer_scale_value, requires_grad=True
            )

    def forward(
        self,
        tokens: torch.Tensor,
        cond_1: torch.Tensor,
        cond_2: torch.Tensor,
        cond_1_mask: Optional[torch.Tensor] = None,
        token_mask: Optional[torch.Tensor] = None,
    ):
        if token_mask is None and cond_1_mask is None:
            cond_attention_mask = None
        elif token_mask is None:
            cond_attention_mask = cond_1_mask.unsqueeze(1) * torch.ones(
                cond_1.shape[0],
                cond_1.shape[1],
                1,
                dtype=torch.bool,
                device=cond_1.device,
            )
        elif cond_1_mask is None:
            cond_attention_mask = torch.ones(
                tokens.shape[0],
                1,
                tokens.shape[1],
                dtype=torch.bool,
                device=tokens.device,
            ) * token_mask.unsqueeze(2)
        else:
            cond_attention_mask = cond_1_mask.unsqueeze(1) * token_mask.unsqueeze(2)
        if token_mask is None:
            attention_mask = None
        else:
            attention_mask = token_mask.unsqueeze(1) * torch.ones(
                tokens.shape[0],
                tokens.shape[1],
                1,
                dtype=torch.bool,
                device=tokens.device,
            )
        ca_output = self.ca(
            self.ca_ln(tokens),
            cond_1,
            attention_mask=cond_attention_mask,
        )
        ca_output = torch.nan_to_num(ca_output, nan=0.0, posinf=0.0, neginf=0.0)
        tokens = tokens + self.ca_stochastic_depth(ca_output)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaln_modulation(cond_2).chunk(6, dim=-1)
        )
        attention_output = self.sa(
            modulate_shift_and_scale(self.initial_ln(tokens), shift_msa, scale_msa),
            attention_mask=attention_mask,
        )
        if self.use_layer_scale:
            tokens = tokens + self.sa_stochastic_depth(
                gate_msa.unsqueeze(1) * self.layer_scale_1 * attention_output
            )
            tokens = tokens + self.ffn_stochastic_depth(
                gate_mlp.unsqueeze(1)
                * self.layer_scale_2
                * self.ffn(
                    modulate_shift_and_scale(
                        self.middle_ln(tokens), shift_mlp, scale_mlp
                    )
                )
            )
        else:
            tokens = tokens + gate_msa.unsqueeze(1) * self.sa_stochastic_depth(
                attention_output
            )
            tokens = tokens + self.ffn_stochastic_depth(
                gate_mlp.unsqueeze(1)
                * self.ffn(
                    modulate_shift_and_scale(
                        self.middle_ln(tokens), shift_mlp, scale_mlp
                    )
                )
            )
        return tokens


class PositionalEmbedding(nn.Module):
    """
    Taken from https://github.com/NVlabs/edm
    """

    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint
        freqs = torch.arange(start=0, end=self.num_channels // 2, dtype=torch.float32)
        freqs = 2 * freqs / self.num_channels
        freqs = (1 / self.max_positions) ** freqs
        self.register_buffer("freqs", freqs)

    def forward(self, x):
        x = torch.outer(x, self.freqs)
        out = torch.cat([x.cos(), x.sin()], dim=1)
        return out.to(x.dtype)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=10000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:, : x.shape[1], :]
        return self.dropout(x)


class TimeEmbedder(nn.Module):
    def __init__(
        self,
        dim: int,
        time_scaling: float,
        expansion: int = 4,
    ):
        super().__init__()
        self.encode_time = PositionalEmbedding(num_channels=dim, endpoint=True)

        self.time_scaling = time_scaling
        self.map_time = nn.Sequential(
            nn.Linear(dim, dim * expansion),
            nn.SiLU(),
            nn.Linear(dim * expansion, dim * expansion),
        )

    def forward(self, t: Tensor) -> Tensor:
        time = self.encode_time(t * self.time_scaling)
        time_mean = time.mean(dim=-1, keepdim=True)
        time_std = time.std(dim=-1, keepdim=True)
        time = (time - time_mean) / time_std
        return self.map_time(time)


def modulate_shift_and_scale(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale).unsqueeze(1) + shift.unsqueeze(1)


# ------------------------------------------------------------------------------------- #


class BaseDirector(nn.Module):
    def __init__(
        self,
        name: str,
        num_feats: int,
        num_cond_feats: int,
        num_cams: int,
        latent_dim: int,
        mlp_multiplier: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        stochastic_depth: float,
        label_dropout: float,
        num_rawfeats: int,
        clip_sequential: bool = False,
        cond_sequential: bool = False,
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__()
        self.name = name
        self.label_dropout = label_dropout
        self.num_rawfeats = num_rawfeats
        self.num_feats = num_feats
        self.num_cams = num_cams
        self.clip_sequential = clip_sequential
        self.cond_sequential = cond_sequential
        self.use_layernorm16 = device == "cuda"

        self.input_projection = nn.Sequential(
            nn.Linear(num_feats, latent_dim),
            PositionalEncoding(latent_dim),
        )
        self.time_embedding = TimeEmbedder(latent_dim // 4, time_scaling=1000)
        self.init_conds_mappings(num_cond_feats, latent_dim)
        self.init_backbone(
            num_layers, latent_dim, mlp_multiplier, num_heads, dropout, stochastic_depth
        )
        self.init_output_projection(num_feats, latent_dim)

    def forward(
        self,
        x: Tensor,
        timesteps: Tensor,
        y: List[Tensor] = None,
        mask: Tensor = None,
    ) -> Tensor:
        mask = mask.logical_not() if mask is not None else None
        x = rearrange(x, "b c n -> b n c")
        x = self.input_projection(x)
        t = self.time_embedding(timesteps)
        if y is not None:
            y = self.mask_cond(y)
            y = self.cond_mapping(y, mask, t)

        x = self.backbone(x, y, mask)
        x = self.output_projection(x, y)
        return rearrange(x, "b n c -> b c n")

    def init_conds_mappings(self, num_cond_feats, latent_dim):
        raise NotImplementedError(
            "This method should be implemented in the derived class"
        )

    def init_backbone(self):
        raise NotImplementedError(
            "This method should be implemented in the derived class"
        )

    def cond_mapping(self, cond: List[Tensor], mask: Tensor, t: Tensor) -> Tensor:
        raise NotImplementedError(
            "This method should be implemented in the derived class"
        )

    def backbone(self, x: Tensor, y: Tensor, mask: Tensor) -> Tensor:
        raise NotImplementedError(
            "This method should be implemented in the derived class"
        )

    def mask_cond(
        self, cond: List[TensorType["batch_size", "num_cond_feats"]]
    ) -> TensorType["batch_size", "num_cond_feats"]:
        bs = cond[0].shape[0]
        if self.training and self.label_dropout > 0.0:
            # 1-> use null_cond, 0-> use real cond
            prob = torch.ones(bs, device=cond[0].device) * self.label_dropout
            masked_cond = []
            common_mask = torch.bernoulli(prob)  # Common to all modalities
            for _cond in cond:
                modality_mask = torch.bernoulli(prob)  # Modality only
                mask = torch.clip(common_mask + modality_mask, 0, 1)
                mask = mask.view(bs, 1, 1) if _cond.dim() == 3 else mask.view(bs, 1)
                masked_cond.append(_cond * (1.0 - mask))
            return masked_cond
        else:
            return cond

    def init_output_projection(self, num_feats, latent_dim):
        raise NotImplementedError(
            "This method should be implemented in the derived class"
        )

    def output_projection(self, x: Tensor, y: Tensor) -> Tensor:
        raise NotImplementedError(
            "This method should be implemented in the derived class"
        )


class AdaLNDirector(BaseDirector):
    def __init__(
        self,
        name: str,
        num_feats: int,
        num_cond_feats: int,
        num_cams: int,
        latent_dim: int,
        mlp_multiplier: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        stochastic_depth: float,
        label_dropout: float,
        num_rawfeats: int,
        clip_sequential: bool = False,
        cond_sequential: bool = False,
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__(
            name=name,
            num_feats=num_feats,
            num_cond_feats=num_cond_feats,
            num_cams=num_cams,
            latent_dim=latent_dim,
            mlp_multiplier=mlp_multiplier,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            stochastic_depth=stochastic_depth,
            label_dropout=label_dropout,
            num_rawfeats=num_rawfeats,
            clip_sequential=clip_sequential,
            cond_sequential=cond_sequential,
            device=device,
        )
        assert not (clip_sequential and cond_sequential)

    def init_conds_mappings(self, num_cond_feats, latent_dim):
        self.joint_cond_projection = nn.Linear(sum(num_cond_feats), latent_dim)

    def cond_mapping(self, cond: List[Tensor], mask: Tensor, t: Tensor) -> Tensor:
        c_emb = torch.cat(cond, dim=-1)
        return self.joint_cond_projection(c_emb) + t

    def init_backbone(
        self,
        num_layers,
        latent_dim,
        mlp_multiplier,
        num_heads,
        dropout,
        stochastic_depth,
    ):
        self.backbone_module = nn.ModuleList(
            [
                AdaLNSABlock(
                    dim_qkv=latent_dim,
                    dim_cond=latent_dim,
                    num_heads=num_heads,
                    mlp_multiplier=mlp_multiplier,
                    dropout=dropout,
                    stochastic_depth=stochastic_depth,
                    use_layernorm16=self.use_layernorm16,
                )
                for _ in range(num_layers)
            ]
        )

    def backbone(self, x: Tensor, y: Tensor, mask: Tensor) -> Tensor:
        for block in self.backbone_module:
            x = block(x, y, mask)
        return x

    def init_output_projection(self, num_feats, latent_dim):
        layer_norm = LayerNorm16Bits if self.use_layernorm16 else nn.LayerNorm

        self.final_norm = layer_norm(latent_dim, eps=1e-6, elementwise_affine=False)
        self.final_linear = nn.Linear(latent_dim, num_feats, bias=True)
        self.final_adaln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim * 2, bias=True),
        )
        # Zero init
        nn.init.zeros_(self.final_adaln[1].weight)
        nn.init.zeros_(self.final_adaln[1].bias)

    def output_projection(self, x: Tensor, y: Tensor) -> Tensor:
        shift, scale = self.final_adaln(y).chunk(2, dim=-1)
        x = modulate_shift_and_scale(self.final_norm(x), shift, scale)
        return self.final_linear(x)


class CrossAttentionDirector(BaseDirector):
    def __init__(
        self,
        name: str,
        num_feats: int,
        num_cond_feats: int,
        num_cams: int,
        latent_dim: int,
        mlp_multiplier: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        stochastic_depth: float,
        label_dropout: float,
        num_rawfeats: int,
        num_text_registers: int,
        clip_sequential: bool = True,
        cond_sequential: bool = True,
        device: str = "cuda",
        **kwargs,
    ):
        self.num_text_registers = num_text_registers
        self.num_heads = num_heads
        self.dropout = dropout
        self.mlp_multiplier = mlp_multiplier
        self.stochastic_depth = stochastic_depth
        super().__init__(
            name=name,
            num_feats=num_feats,
            num_cond_feats=num_cond_feats,
            num_cams=num_cams,
            latent_dim=latent_dim,
            mlp_multiplier=mlp_multiplier,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            stochastic_depth=stochastic_depth,
            label_dropout=label_dropout,
            num_rawfeats=num_rawfeats,
            clip_sequential=clip_sequential,
            cond_sequential=cond_sequential,
            device=device,
        )
        assert clip_sequential and cond_sequential

    def init_conds_mappings(self, num_cond_feats, latent_dim):
        self.cond_projection = nn.ModuleList(
            [nn.Linear(num_cond_feat, latent_dim) for num_cond_feat in num_cond_feats]
        )
        self.cond_registers = nn.Parameter(
            torch.randn(self.num_text_registers, latent_dim), requires_grad=True
        )
        nn.init.trunc_normal_(self.cond_registers, std=0.02, a=-2 * 0.02, b=2 * 0.02)
        self.cond_sa = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim_qkv=latent_dim,
                    num_heads=self.num_heads,
                    mlp_multiplier=self.mlp_multiplier,
                    dropout=self.dropout,
                    stochastic_depth=self.stochastic_depth,
                    use_layernorm16=self.use_layernorm16,
                )
                for _ in range(2)
            ]
        )
        self.cond_positional_embedding = PositionalEncoding(latent_dim, max_len=10000)

    def cond_mapping(self, cond: List[Tensor], mask: Tensor, t: Tensor) -> Tensor:
        batch_size = cond[0].shape[0]
        cond_emb = [
            cond_proj(rearrange(c, "b c n -> b n c"))
            for cond_proj, c in zip(self.cond_projection, cond)
        ]
        cond_emb = [
            self.cond_registers.unsqueeze(0).expand(batch_size, -1, -1),
            t.unsqueeze(1),
        ] + cond_emb
        cond_emb = torch.cat(cond_emb, dim=1)
        cond_emb = self.cond_positional_embedding(cond_emb)
        for block in self.cond_sa:
            cond_emb = block(cond_emb)
        return cond_emb

    def init_backbone(
        self,
        num_layers,
        latent_dim,
        mlp_multiplier,
        num_heads,
        dropout,
        stochastic_depth,
    ):
        self.backbone_module = nn.ModuleList(
            [
                CrossAttentionSABlock(
                    dim_qkv=latent_dim,
                    dim_cond=latent_dim,
                    num_heads=num_heads,
                    mlp_multiplier=mlp_multiplier,
                    dropout=dropout,
                    stochastic_depth=stochastic_depth,
                    use_layernorm16=self.use_layernorm16,
                )
                for _ in range(num_layers)
            ]
        )

    def backbone(self, x: Tensor, y: Tensor, mask: Tensor) -> Tensor:
        for block in self.backbone_module:
            x = block(x, y, mask, None)
        return x

    def init_output_projection(self, num_feats, latent_dim):
        layer_norm = LayerNorm16Bits if self.use_layernorm16 else nn.LayerNorm

        self.final_norm = layer_norm(latent_dim, eps=1e-6)
        self.final_linear = nn.Linear(latent_dim, num_feats, bias=True)

    def output_projection(self, x: Tensor, y: Tensor) -> Tensor:
        return self.final_linear(self.final_norm(x))


class InContextDirector(BaseDirector):
    def __init__(
        self,
        name: str,
        num_feats: int,
        num_cond_feats: int,
        num_cams: int,
        latent_dim: int,
        mlp_multiplier: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        stochastic_depth: float,
        label_dropout: float,
        num_rawfeats: int,
        clip_sequential: bool = False,
        cond_sequential: bool = False,
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__(
            name=name,
            num_feats=num_feats,
            num_cond_feats=num_cond_feats,
            num_cams=num_cams,
            latent_dim=latent_dim,
            mlp_multiplier=mlp_multiplier,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            stochastic_depth=stochastic_depth,
            label_dropout=label_dropout,
            num_rawfeats=num_rawfeats,
            clip_sequential=clip_sequential,
            cond_sequential=cond_sequential,
            device=device,
        )

    def init_conds_mappings(self, num_cond_feats, latent_dim):
        self.cond_projection = nn.ModuleList(
            [nn.Linear(num_cond_feat, latent_dim) for num_cond_feat in num_cond_feats]
        )

    def cond_mapping(self, cond: List[Tensor], mask: Tensor, t: Tensor) -> Tensor:
        for i in range(len(cond)):
            if cond[i].dim() == 3:
                cond[i] = rearrange(cond[i], "b c n -> b n c")
        cond_emb = [cond_proj(c) for cond_proj, c in zip(self.cond_projection, cond)]
        cond_emb = [c.unsqueeze(1) if c.dim() == 2 else cond_emb for c in cond_emb]
        cond_emb = torch.cat([t.unsqueeze(1)] + cond_emb, dim=1)
        return cond_emb

    def init_backbone(
        self,
        num_layers,
        latent_dim,
        mlp_multiplier,
        num_heads,
        dropout,
        stochastic_depth,
    ):
        self.backbone_module = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim_qkv=latent_dim,
                    num_heads=num_heads,
                    mlp_multiplier=mlp_multiplier,
                    dropout=dropout,
                    stochastic_depth=stochastic_depth,
                    use_layernorm16=self.use_layernorm16,
                )
                for _ in range(num_layers)
            ]
        )

    def backbone(self, x: Tensor, y: Tensor, mask: Tensor) -> Tensor:
        bs, n_y, _ = y.shape
        mask = torch.cat([torch.ones(bs, n_y, device=y.device), mask], dim=1)
        x = torch.cat([y, x], dim=1)
        for block in self.backbone_module:
            x = block(x, mask)
        return x

    def init_output_projection(self, num_feats, latent_dim):
        layer_norm = LayerNorm16Bits if self.use_layernorm16 else nn.LayerNorm

        self.final_norm = layer_norm(latent_dim, eps=1e-6)
        self.final_linear = nn.Linear(latent_dim, num_feats, bias=True)

    def output_projection(self, x: Tensor, y: Tensor) -> Tensor:
        num_y = y.shape[1]
        x = x[:, num_y:]
        return self.final_linear(self.final_norm(x))
