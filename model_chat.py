# -*- coding: utf-8 -*-
import math
import warnings
import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import Cache, DynamicCache

_UNSET = object()

# ============================================================
# 输入校验与采样器
# ============================================================

def validate_input_ids(input_ids, vocab_size: int, allow_empty: bool = False):
    if isinstance(input_ids, dict):
        input_ids = input_ids.get("input_ids")
    if not torch.is_tensor(input_ids):
        raise TypeError(f"input_ids 必须是 torch.Tensor，实际为 {type(input_ids)}")
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    elif input_ids.dim() != 2:
        raise ValueError(f"input_ids 必须是 1D/2D 张量，实际维度为 {input_ids.dim()}")
    if not torch.is_floating_point(input_ids):
        input_ids = input_ids.long()
    else:
        raise TypeError(f"input_ids 必须是整数 dtype，实际为 {input_ids.dtype}")
    if not allow_empty and input_ids.numel() == 0:
        raise ValueError("input_ids 序列长度为 0（空 prompt），至少需要一个 token")
    if input_ids.numel() > 0:
        min_id = int(input_ids.min().item())
        max_id = int(input_ids.max().item())
        if min_id < 0 or max_id >= vocab_size:
            raise IndexError(
                f"token id 越界：input_ids 取值区间 [{min_id}, {max_id}]，"
                f"合法区间 [0, {vocab_size - 1}]"
            )
    return input_ids

def validate_attention_mask(attention_mask, batch_size: int, kv_len: int):
    if attention_mask is None:
        return None
    if not torch.is_tensor(attention_mask):
        raise TypeError("attention_mask 必须是 torch.Tensor")
    if attention_mask.dim() != 2 or tuple(attention_mask.shape) != (batch_size, kv_len):
        raise ValueError(
            f"attention_mask 形状 {tuple(attention_mask.shape)} 与期望 "
            f"({batch_size}, {kv_len}) 不符（长度须等于 历史缓存+当前序列）"
        )
    if attention_mask.dtype != torch.bool:
        if not torch.all((attention_mask == 0) | (attention_mask == 1)):
            raise ValueError("attention_mask 只能取 0/1（或 bool）")
        attention_mask = attention_mask.bool()
    return attention_mask

def validate_sampling_params(temperature, top_p, top_k, repetition_penalty,
                             num_return_sequences, max_new_tokens):
    if not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature < 0:
        raise ValueError(f"temperature 必须是非负有限实数，得到 {temperature}")
    if not isinstance(top_p, (int, float)) or not (0.0 < top_p <= 1.0):
        raise ValueError(f"top_p 必须落在 (0, 1]，得到 {top_p}")
    if not isinstance(top_k, int) or top_k < 0:
        raise ValueError(f"top_k 必须是非负整数，得到 {top_k!r}")
    if not isinstance(repetition_penalty, (int, float)) or repetition_penalty <= 0 or not math.isfinite(repetition_penalty):
        raise ValueError(f"repetition_penalty 必须是正实数，得到 {repetition_penalty}")
    if not isinstance(num_return_sequences, int) or num_return_sequences < 1:
        raise ValueError(f"num_return_sequences 必须是 >=1 的整数，得到 {num_return_sequences!r}")
    if not isinstance(max_new_tokens, int) or max_new_tokens < 0:
        raise ValueError(f"max_new_tokens 必须是非负整数，得到 {max_new_tokens!r}")

def apply_repetition_penalty(logits, input_ids, penalty: float, window: int):
    if abs(penalty - 1.0) <= 1e-6:
        return logits
    batch_size = logits.shape[0]
    for i in range(batch_size):
        start = max(0, input_ids.size(1) - window)
        seen = torch.unique(input_ids[i, start:])
        score = logits[i, seen]
        logits[i, seen] = torch.where(score > 0, score / penalty, score * penalty)
    return logits

def warp_logits(logits, temperature: float, top_k: int, top_p: float):
    if temperature > 0:
        logits = logits / temperature

    vocab_size = logits.size(-1)
    if top_k > 0:
        k = min(top_k, vocab_size)
        kth_value = torch.topk(logits, k, dim=-1).values[..., -1, None]
        logits = torch.where(
            logits < kth_value,
            torch.full_like(logits, float("-inf")),
            logits,
        )

    probs = torch.softmax(logits, dim=-1)

    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_remove = cumulative_probs > top_p
        sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
        sorted_remove[..., 0] = False
        remove = torch.zeros_like(sorted_remove).scatter_(1, sorted_indices, sorted_remove)
        probs = probs.masked_fill(remove, 0.0)

    if not torch.isfinite(probs).all():
        raise RuntimeError("采样概率出现 NaN/Inf，请检查 logits 或温度设置")
    return probs

def sample_next_token(logits, do_sample: bool, temperature: float, top_k: int,
                      top_p: float, generator=None):
    if (not do_sample) or temperature == 0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    probs = warp_logits(logits, temperature, top_k, top_p)
    return torch.multinomial(probs, num_samples=1, generator=generator)

def normalize_eos_token_id(eos_token_id, device):
    if eos_token_id is None:
        return None
    if torch.is_tensor(eos_token_id):
        eos = eos_token_id.flatten().long()
    elif isinstance(eos_token_id, (list, tuple)):
        eos = torch.tensor(eos_token_id, dtype=torch.long)
    else:
        eos = torch.tensor([eos_token_id], dtype=torch.long)
    return eos.to(device)

# ============================================================
# 配置
# ============================================================

class ChatConfig(PretrainedConfig):
    model_type = "LingLongChat"

    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=8,
        use_moe=False,
        dropout=0.0,
        vocab_size=6400,
        bos_token_id=1,
        eos_token_id=2,
        flash_attn=True,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=None,
        hidden_act="silu",
        intermediate_size=None,
        max_position_embeddings=32768,
        rms_norm_eps=1e-6,
        rope_theta=1e6,
        tie_word_embeddings=True,
        inference_rope_scaling=False,
        rope_scaling=None,
        num_experts=4,
        num_experts_per_tok=1,
        moe_intermediate_size=None,
        norm_topk_prob=True,
        router_aux_loss_coef=5e-4,
        router_z_loss_coef=1e-3,
        **kwargs
    ):
        super().__init__(**kwargs)
        if num_attention_heads % num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads({num_attention_heads}) 必须能被 "
                f"num_key_value_heads({num_key_value_heads}) 整除"
            )

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.use_moe = use_moe
        self.dropout = dropout
        self.vocab_size = vocab_size
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.flash_attn = flash_attn
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim({self.head_dim}) 必须为偶数（RoPE 需要成对的维度）")
        self.hidden_act = hidden_act

        if intermediate_size is None:
            intermediate_size = math.ceil(hidden_size * math.pi / 64) * 64
        self.intermediate_size = intermediate_size

        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.tie_word_embeddings = tie_word_embeddings

        if inference_rope_scaling and rope_scaling is None:
            self.rope_scaling = {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn"
            }
        else:
            self.rope_scaling = rope_scaling

        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        if num_experts_per_tok > num_experts:
            raise ValueError(
                f"num_experts_per_tok({num_experts_per_tok}) 不能超过 num_experts({num_experts})"
            )
        self.moe_intermediate_size = moe_intermediate_size if moe_intermediate_size is not None else self.intermediate_size
        self.norm_topk_prob = norm_topk_prob
        self.router_aux_loss_coef = router_aux_loss_coef
        self.router_z_loss_coef = router_z_loss_coef

# ============================================================
# 基础组件
# ============================================================

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return (self.weight * self.norm(x.float())).type_as(x)

def precompute_freqs_cis(dim: int, end: int, rope_base: float = 1e6, rope_scaling: dict = None):
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    attn_factor = 1.0

    if rope_scaling is not None:
        orig_max = rope_scaling.get("original_max_position_embeddings", 2048)
        factor = rope_scaling.get("factor", 16)
        beta_fast = rope_scaling.get("beta_fast", 32.0)
        beta_slow = rope_scaling.get("beta_slow", 1.0)
        attn_factor = rope_scaling.get("attention_factor", 1.0)

        inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
        low = max(math.floor(inv_dim(beta_fast)), 0)
        high = min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
        if low > high:
            raise ValueError(
                f"YaRN ramp 区间非法：low({low}) > high({high})，请检查 "
                f"beta_fast/beta_slow/original_max_position_embeddings/factor 配置"
            )
        ramp = torch.clamp((torch.arange(dim // 2).float() - low) / max(high - low, 0.001), 0, 1)
        freqs = freqs * (1 - ramp + ramp / factor)

    t = torch.arange(end)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cos, freqs_sin, attn_factor

def _rope_broadcast(t, q):
    if t.dim() == 2:
        return t[None, :, None, :]
    if t.dim() == 3:
        return t[:, :, None, :]
    if t.dim() == 4:
        return t
    raise ValueError(f"RoPE cos/sin 维度必须是 2/3/4，得到 {t.dim()}")

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    cos_q, sin_q = _rope_broadcast(cos, q), _rope_broadcast(sin, q)
    q_embed = ((q * cos_q) + (rotate_half(q) * sin_q)).to(q.dtype)
    k_embed = ((k * cos_q) + (rotate_half(k) * sin_q)).to(k.dtype)
    return q_embed, k_embed

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, num_key_value_heads, n_rep, head_dim)
        .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )

def build_causal_allowed(q_len: int, kv_len: int, past_len: int, attention_mask=None, device=None):
    causal = torch.ones(q_len, kv_len, dtype=torch.bool, device=device).tril(diagonal=past_len)
    allowed = causal.unsqueeze(0).unsqueeze(0)
    if attention_mask is not None:
        key_allowed = attention_mask.bool()[:, None, None, :]
        allowed = allowed & key_allowed

    has_any = allowed.any(dim=-1, keepdim=True)
    if not bool(has_any.all()):
        diag_key_idx = torch.arange(q_len, device=device) + past_len
        in_range = diag_key_idx < kv_len
        diag = torch.zeros(q_len, kv_len, dtype=torch.bool, device=device)
        diag[torch.arange(q_len, device=device)[in_range],
             diag_key_idx[in_range]] = True
        allowed = allowed | (~has_any & diag.unsqueeze(0).unsqueeze(0))
    return allowed

def _read_cache_layer_kv(cache, layer_idx: int):
    try:
        k = cache.get_key(layer_idx)
        v = cache.get_value(layer_idx)
    except (AttributeError, NotImplementedError, IndexError):
        layer = cache.layers[layer_idx]
        k = layer.keys if layer.is_initialized else None
        v = layer.values if layer.is_initialized else None
    if k is None or v is None:
        return None, None
    return k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous()

def unpack_past_cache(past_key_values, num_layers: int):
    if isinstance(past_key_values, Cache):
        if len(past_key_values) == 0:
            return [None] * num_layers, past_key_values
        kv_list = [_read_cache_layer_kv(past_key_values, i) for i in range(num_layers)]
        return kv_list, past_key_values

    if past_key_values is None:
        return [None] * num_layers, None
    if len(past_key_values) == 0:
        return [None] * num_layers, None

    if not isinstance(past_key_values, (list, tuple)):
        raise TypeError(
            f"past_key_values 类型不支持：{type(past_key_values)}，"
            "请传 transformers Cache 或 list[(k,v),...]"
        )
    if len(past_key_values) != num_layers:
        raise ValueError(
            f"past_key_values 层数 {len(past_key_values)} 与模型层数 {num_layers} 不一致"
        )
    normed = []
    for layer_kv in past_key_values:
        if layer_kv is None:
            normed.append(None)
        elif isinstance(layer_kv, (list, tuple)) and len(layer_kv) == 2 and torch.is_tensor(layer_kv[0]):
            normed.append((layer_kv[0], layer_kv[1]))
        else:
            raise ValueError("past_key_values 每层必须是 None 或 (k, v) 二元组")
    return normed, None

# ============================================================
# 注意力 / 前馈 / MoE
# ============================================================

class Attention(nn.Module):
    def __init__(self, config: ChatConfig):
        super().__init__()
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.head_dim
        self.is_causal = True

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.flash = hasattr(F, "scaled_dot_product_attention") and config.flash_attn

    def forward(
        self,
        x,
        position_embeddings,
        past_key_value=None,
        use_cache=False,
        attention_mask=None
    ):
        bsz, seq_len, _ = x.shape
        cos, sin, attn_factor = position_embeddings
        scale = float(attn_factor) / math.sqrt(self.head_dim)
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xq, xk = self.q_norm(xq), self.k_norm(xk)
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        if past_key_value is not None:
            pk, pv = past_key_value
            xk = torch.cat([pk, xk], dim=1)
            xv = torch.cat([pv, xv], dim=1)
        present_kv = (xk, xv) if use_cache else None

        xq = xq.transpose(1, 2)
        xk = repeat_kv(xk, self.n_rep).transpose(1, 2)
        xv = repeat_kv(xv, self.n_rep).transpose(1, 2)

        q_len, kv_len = xq.size(2), xk.size(2)
        past_len = kv_len - q_len
        fast_causal = attention_mask is None and q_len == kv_len

        if self.flash:
            if fast_causal:
                output = F.scaled_dot_product_attention(
                    xq, xk, xv,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=self.is_causal,
                    scale=scale,
                )
            else:
                allowed = build_causal_allowed(q_len, kv_len, past_len, attention_mask, xq.device)
                additive = torch.zeros(allowed.shape, dtype=xq.dtype, device=xq.device)
                additive.masked_fill_(~allowed, torch.finfo(xq.dtype).min)
                output = F.scaled_dot_product_attention(
                    xq, xk, xv,
                    attn_mask=additive,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=False,
                    scale=scale,
                )
        else:
            scores = (xq @ xk.transpose(-2, -1)) * scale
            allowed = build_causal_allowed(q_len, kv_len, past_len, attention_mask, scores.device)
            scores = scores.float().masked_fill(~allowed, float("-inf"))
            attn_weight = torch.softmax(scores, dim=-1)
            attn_weight = torch.nan_to_num(attn_weight, nan=0.0)
            attn_weight = self.attn_dropout(attn_weight).to(xv.dtype)
            output = attn_weight @ xv

        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, present_kv

class FeedForward(nn.Module):
    def __init__(self, config: ChatConfig, intermediate_size: int = None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class MOEFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList([
            FeedForward(config, intermediate_size=config.moe_intermediate_size)
            for _ in range(config.num_experts)
        ])
        self.register_buffer("aux_loss", torch.zeros(()), persistent=False)

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)
        logits = self.gate(x_flat).float()
        scores = F.softmax(logits, dim=-1)

        topk_weight, topk_idx = torch.topk(
            scores, k=self.config.num_experts_per_tok, dim=-1, sorted=False
        )
        if self.config.norm_topk_prob:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)

        y = torch.zeros_like(x_flat)
        for expert_idx, expert in enumerate(self.experts):
            mask = (topk_idx == expert_idx)
            token_mask = mask.any(dim=-1)
            if not token_mask.any():
                continue
            token_idx = token_mask.nonzero(as_tuple=False).flatten()
            mask_per_token = mask[token_idx]
            w_per_token = topk_weight[token_idx]
            w_sum = (w_per_token * mask_per_token).sum(dim=-1)
            expert_out = expert(x_flat[token_idx])
            y.index_add_(0, token_idx, expert_out * w_sum.to(expert_out.dtype).unsqueeze(-1))

        aux_loss = scores.new_zeros(())
        if self.training:
            num_experts = self.config.num_experts
            tokens_per_expert = F.one_hot(topk_idx, num_experts).float().sum(dim=(0, 1))
            expert_fraction = tokens_per_expert / float(topk_idx.numel())
            mean_prob = scores.mean(dim=0)
            balance_loss = (
                (expert_fraction * mean_prob).sum()
                * num_experts
                * self.config.router_aux_loss_coef
            )
            z_loss = (
                torch.mean(torch.logsumexp(logits, dim=-1) ** 2)
                * self.config.router_z_loss_coef
            )
            aux_loss = balance_loss + z_loss

        self.aux_loss = aux_loss.detach()
        return y.view(batch_size, seq_len, hidden_dim), aux_loss

class ChatBlock(nn.Module):
    def __init__(self, layer_id: int, config: ChatConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.config = config
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(
        self,
        hidden_states,
        position_embeddings,
        past_kv=None,
        use_cache=False,
        attention_mask=None
    ):
        residual = hidden_states
        attn_out, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings,
            past_kv,
            use_cache,
            attention_mask
        )
        hidden_states = residual + attn_out

        mlp_in = self.post_attention_layernorm(hidden_states)
        if self.config.use_moe:
            mlp_out, aux_loss = self.mlp(mlp_in)
        else:
            mlp_out = self.mlp(mlp_in)
            aux_loss = mlp_in.new_zeros(())

        hidden_states = hidden_states + mlp_out
        return hidden_states, present_key_value, aux_loss

# ============================================================
# 骨干模型
# ============================================================

class ChatModel(nn.Module):
    def __init__(self, config: ChatConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([
            ChatBlock(l, config) for l in range(self.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        freqs_cos, freqs_sin, attn_factor = precompute_freqs_cis(
            dim=config.head_dim,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)
        self.register_buffer("attn_factor", torch.tensor(attn_factor), persistent=False)
        self._rope_warned = False

    def _reset_rope_buffers(self):
        freqs_cos, freqs_sin, attn_factor = precompute_freqs_cis(
            dim=self.config.head_dim,
            end=self.config.max_position_embeddings,
            rope_base=self.config.rope_theta,
            rope_scaling=self.config.rope_scaling
        )
        device = self.freqs_cos.device
        self.register_buffer("freqs_cos", freqs_cos.to(device), persistent=False)
        self.register_buffer("freqs_sin", freqs_sin.to(device), persistent=False)
        self.register_buffer("attn_factor", torch.tensor(attn_factor, device=device), persistent=False)

    def _ensure_rope(self, target_len: int, device):
        need_rebuild = (
            self.freqs_cos.device != device
            or self.freqs_cos.shape[0] < target_len
        )
        if not need_rebuild:
            return
        end = max(target_len, int(self.freqs_cos.shape[0] * 1.5), self.config.max_position_embeddings)
        if (not self._rope_warned) and self.config.rope_scaling is None and end > self.config.max_position_embeddings:
            warnings.warn(
                f"序列长度 {target_len} 超过 max_position_embeddings="
                f"{self.config.max_position_embeddings}，且未配置 RoPE 缩放；"
                "RoPE 将直接外推，长程位置可能退化（建议启用 YaRN 或调大 max_position_embeddings）",
                RuntimeWarning,
            )
            self._rope_warned = True
        freqs_cos, freqs_sin, attn_factor = precompute_freqs_cis(
            dim=self.config.head_dim,
            end=end,
            rope_base=self.config.rope_theta,
            rope_scaling=self.config.rope_scaling
        )
        self.register_buffer("freqs_cos", freqs_cos.to(device), persistent=False)
        self.register_buffer("freqs_sin", freqs_sin.to(device), persistent=False)
        self.register_buffer("attn_factor", torch.tensor(attn_factor, device=device), persistent=False)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        **kwargs
    ):
        input_ids = validate_input_ids(input_ids, self.config.vocab_size)
        batch_size, seq_length = input_ids.shape

        kv_list, cache_obj = unpack_past_cache(past_key_values, self.num_hidden_layers)

        start_pos = 0
        first_past = kv_list[0]
        if first_past is not None:
            pk, _ = first_past
            if pk is not None:
                start_pos = pk.shape[1]
        kv_len = start_pos + seq_length

        attention_mask = validate_attention_mask(attention_mask, batch_size, kv_len)

        max_pos = start_pos + seq_length
        if position_ids is not None:
            if not torch.is_tensor(position_ids) or tuple(position_ids.shape) != (batch_size, seq_length):
                raise ValueError(
                    f"position_ids 形状须为 ({batch_size}, {seq_length})，"
                    f"实际 {None if position_ids is None else tuple(position_ids.shape)}"
                )
            position_ids = position_ids.long().to(input_ids.device)
            max_pos = max(max_pos, int(position_ids.max().item()) + 1)
        self._ensure_rope(max_pos, input_ids.device)

        if position_ids is None:
            cos = self.freqs_cos[start_pos: start_pos + seq_length]
            sin = self.freqs_sin[start_pos: start_pos + seq_length]
        else:
            flat_pos = position_ids.reshape(-1)
            cos = self.freqs_cos.index_select(0, flat_pos).reshape(batch_size, seq_length, -1)
            sin = self.freqs_sin.index_select(0, flat_pos).reshape(batch_size, seq_length, -1)
        position_embeddings = (cos, sin, self.attn_factor)

        hidden_states = self.dropout(self.embed_tokens(input_ids))

        presents = []
        total_aux_loss = hidden_states.new_zeros(())
        for layer, past_kv in zip(self.layers, kv_list):
            hidden_states, present, aux_loss = layer(
                hidden_states,
                position_embeddings,
                past_kv=past_kv,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            if use_cache:
                presents.append(present)
            total_aux_loss = total_aux_loss + aux_loss

        hidden_states = self.norm(hidden_states)

        if not use_cache:
            presents = None
        elif isinstance(cache_obj, DynamicCache):
            for idx, (k, v) in enumerate(presents):
                if k is not None:
                    cache_obj.update(
                        k[:, start_pos:].transpose(1, 2).contiguous(),
                        v[:, start_pos:].transpose(1, 2).contiguous(),
                        idx,
                    )
            presents = cache_obj
        return hidden_states, presents, total_aux_loss

# ============================================================
# Causal LM 外壳
# ============================================================

class ChatForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = ChatConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: ChatConfig = None):
        self.config = config or ChatConfig()
        super().__init__(self.config)
        self.model = ChatModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.post_init()

    def _init_weights(self, module):
        super()._init_weights(module)
        if module is self.model:
            self.model._reset_rope_buffers()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        logits_to_keep=0,
        labels=None,
        **kwargs
    ):
        if not isinstance(logits_to_keep, int) or logits_to_keep < 0:
            raise ValueError(f"logits_to_keep 必须是非负整数，得到 {logits_to_keep!r}")

        hidden_states, past_key_values, aux_loss = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs
        )

        slice_indices = (
            slice(-logits_to_keep, None)
            if logits_to_keep > 0 else slice(None)
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            if not torch.is_tensor(labels) or labels.dim() != 2 or labels.shape != input_ids.shape:
                raise ValueError(
                    f"labels 形状须与 input_ids {tuple(input_ids.shape)} 一致，"
                    f"实际 {None if labels is None else tuple(labels.shape)}"
                )
            if 0 < logits_to_keep <= 1:
                raise ValueError(
                    "训练（labels 非空）时 logits_to_keep 必须为 0（全量）或 >=2，"
                    "否则 shift 后没有可计算损失的位置"
                )
            labels = labels[:, slice_indices]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
            loss = loss + aux_loss

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states
        )

    @torch.inference_mode()
    def generate(
        self,
        inputs=None,
        attention_mask=None,
        max_new_tokens=8192,
        temperature=0.85,
        top_p=0.85,
        top_k=50,
        eos_token_id=_UNSET,
        streamer=None,
        use_cache=True,
        num_return_sequences=1,
        do_sample=True,
        repetition_penalty=1.0,
        repetition_window: int = 1024,
        generator=None,
        pad_token_id=None,
        position_ids=None,
        **kwargs
    ):
        input_ids = kwargs.pop("input_ids", inputs)
        return_kv = kwargs.pop("return_kv", False)
        past_key_values = kwargs.pop("past_key_values", None)

        if isinstance(input_ids, dict):
            attention_mask = input_ids.get("attention_mask", attention_mask)
            position_ids = input_ids.get("position_ids", position_ids)
            input_ids = input_ids.get("input_ids")

        input_ids = validate_input_ids(input_ids, self.config.vocab_size)
        validate_sampling_params(
            temperature, top_p, top_k, repetition_penalty,
            num_return_sequences, max_new_tokens
        )
        if repetition_window < 0:
            raise ValueError(f"repetition_window 必须非负，得到 {repetition_window}")

        if attention_mask is not None:
            attention_mask = validate_attention_mask(
                attention_mask, input_ids.shape[0], input_ids.shape[1]
            )

            if bool((attention_mask[:, 0] == False).any()):
                raise NotImplementedError(
                    "检测到左 padding（行首 mask=0）；本引擎仅支持右 padding，"
                    "请改用右 padding 或在 tokenizer 侧设置 padding_side='right'"
                )
        if position_ids is not None:
            position_ids = position_ids.long()

        if num_return_sequences != 1:
            input_ids = input_ids.repeat_interleave(num_return_sequences, dim=0)
            if attention_mask is not None:
                attention_mask = attention_mask.repeat_interleave(num_return_sequences, dim=0)
            if position_ids is not None:
                position_ids = position_ids.repeat_interleave(num_return_sequences, dim=0)

        batch_size = input_ids.shape[0]
        device = input_ids.device

        if eos_token_id is _UNSET:
            eos_token_id = getattr(self.config, "eos_token_id", 2)
        eos_token_id = normalize_eos_token_id(eos_token_id, device)

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        if eos_token_id is not None:
            finished |= torch.isin(input_ids[:, -1], eos_token_id)

        if streamer:
            streamer.put(input_ids.cpu())

        if max_new_tokens == 0:
            if streamer:
                streamer.end()
            if return_kv:
                return {"generated_ids": input_ids, "past_kv": past_key_values}
            return input_ids

        first_step = True
        for _ in range(max_new_tokens):
            if finished.all():
                break

            past_len = (
                past_key_values.get_seq_length()
                if isinstance(past_key_values, Cache)
                else (past_key_values[0][0].shape[1] if past_key_values and len(past_key_values) > 0 else 0)
            )
            cur_ids = input_ids[:, past_len:]

            cur_position_ids = None
            if position_ids is not None and first_step:
                cur_position_ids = position_ids[:, past_len: past_len + cur_ids.size(1)]
            elif attention_mask is not None:
                full_pos = torch.cumsum(attention_mask, dim=-1).long() - 1
                full_pos.clamp_min_(0)
                cur_position_ids = full_pos[:, past_len: past_len + cur_ids.size(1)]

            keep = 1 if (first_step and attention_mask is None) else 0

            outputs = self.forward(
                cur_ids,
                attention_mask=attention_mask,
                position_ids=cur_position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                logits_to_keep=keep,
                **kwargs
            )

            if first_step and attention_mask is not None:
                last_valid = attention_mask.sum(dim=-1) - 1 - past_len
                logits = outputs.logits[
                    torch.arange(batch_size, device=device), last_valid.clamp_min(0), :
                ]
            else:
                logits = outputs.logits[:, -1, :]
            first_step = False

            logits = apply_repetition_penalty(
                logits, input_ids, repetition_penalty, repetition_window
            )

            next_token = sample_next_token(
                logits,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                generator=generator,
            )

            if eos_token_id is not None:
                placeholder = torch.full_like(next_token, int(eos_token_id[0].item()))
                next_token = torch.where(finished.unsqueeze(-1), placeholder, next_token)

            if attention_mask is not None:
                new_flag = (~finished).to(attention_mask.dtype).unsqueeze(-1)
                attention_mask = torch.cat([attention_mask, new_flag], dim=-1)

            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values if use_cache else None

            if streamer:
                streamer.put(next_token.cpu())

            if eos_token_id is not None:
                finished |= torch.isin(next_token.squeeze(-1), eos_token_id)
                if finished.all():
                    break

        if streamer:
            streamer.end()

        if return_kv:
            return {"generated_ids": input_ids, "past_kv": past_key_values}
        return input_ids