import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.models.gpt_neox import GPTNeoXConfig


class KVCache:
    def __init__(self):
        # shape [batch_size, num_attention_heads, seq_len, head_size]
        self.key: torch.Tensor = None
        self.value: torch.Tensor = None

    def update(self, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # (TODO) Task 2: Implement the update method for the KVCache
        # Concatenate the key and value tensors along the sequence length dimension
        # If the cache is empty, initialize it with the key and value tensors
        # return the updated key and value tensors

    def past_key_values_length(self) -> int:
        # (TODO) Task 2: return the length of the past key values


@dataclass
class BaseModelOutputWithPast(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    kvcaches: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class CausalLMOutputWithPast(ModelOutput):
    logits: torch.FloatTensor = None
    kvcaches: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim,
                          2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device,
                         dtype=torch.int64).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(
                seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len],
            self.sin_cached[:seq_len],
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "The hidden size is not divisble by the number of attention heads! Make sure to update them"
            )
        self.head_size = self.hidden_size // self.num_attention_heads
        self.rotary_ndims = int(self.head_size * config.rotary_pct)
        self._init_bias(config.max_position_embeddings)

        self.register_buffer(
            "masked_bias", torch.tensor(-1e9), persistent=False)
        self._init_rope()

        self.norm_factor = self.head_size**-0.5
        self.query_key_value = nn.Linear(
            config.hidden_size, 3 * config.hidden_size, bias=config.attention_bias)
        self.dense = nn.Linear(
            config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.is_causal = True

    def _init_bias(self, max_positions, device=None):
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )
        if device is not None:
            self.bias = self.bias.to(device)

    def _init_rope(self):
        self.rotary_emb = RotaryEmbedding(
            self.rotary_ndims, self.config.max_position_embeddings, base=self.config.rotary_emb_base
        )

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        position_ids: torch.LongTensor,
        layer_kvcache: Optional[Tuple[torch.Tensor]] = None,
    ):
        # Apply attention-specific projections and rope
        query, key, value, present = self._attn_projections_and_rope(
            hidden_states=hidden_states, position_ids=position_ids, layer_kvcache=layer_kvcache
        )

        # Compute attention
        attn_output = self._attn(
            query, key, value, attention_mask)
        
        attn_output = self.dense(attn_output)

        return attn_output, present

    def _attn_projections_and_rope(
        self,
        hidden_states: torch.FloatTensor,
        position_ids: torch.LongTensor,
        layer_kvcache: KVCache,
    ):
        # Compute QKV
        # Attention heads [batch, seq_len, hidden_size]
        #   --> [batch, seq_len, (np * 3 * head_size)]
        qkv = self.query_key_value(hidden_states)

        # [batch, seq_len, (num_heads * 3 * head_size)]
        #   --> [batch, seq_len, num_heads, 3 * head_size]
        new_qkv_shape = qkv.size()[:-1] + \
            (self.num_attention_heads, 3 * self.head_size)
        qkv = qkv.view(*new_qkv_shape)

        # [batch, seq_len, num_attention_heads, 3 * head_size] --> 3 [batch, num_attention_heads, seq_len, head_size]
        query = qkv[..., : self.head_size].permute(0, 2, 1, 3)
        key = qkv[..., self.head_size: 2 * self.head_size].permute(0, 2, 1, 3)
        value = qkv[..., 2 * self.head_size:].permute(0, 2, 1, 3)

        # Compute rotary embeddings on rotary_ndims
        query_rot = query[..., : self.rotary_ndims]
        query_pass = query[..., self.rotary_ndims:]
        key_rot = key[..., : self.rotary_ndims]
        key_pass = key[..., self.rotary_ndims:]

        # Compute token offset for rotary embeddings (when decoding)
        seq_len = key.shape[-2]
        seq_len += layer_kvcache.past_key_values_length()
        cos, sin = self.rotary_emb(value, seq_len=seq_len)
        query, key = apply_rotary_pos_emb(
            query_rot, key_rot, cos, sin, position_ids)
        query = torch.cat((query, query_pass), dim=-1)
        key = torch.cat((key, key_pass), dim=-1)

        # Cache QKV values
        key, value = layer_kvcache.update(key, value)

        return query, key, value, layer_kvcache

    def _attn(self, query, key, value, attention_mask=None):
        # (TODO) Task 3: Implement the forward pass for the Attention module
    
        # Input: q, k, v: [bs, num_attention_heads, seq_len, attn_head_size]
        # Output: attn_output: [bs, seq_len, hidden_size]
    
        # Pseudocode:
        # 1. Compute the attention scores using the query and key tensors
        # 2. Apply the causal mask to the attention scores using self.bias
        # 3. Apply the attention mask to the attention scores
        # 4. Apply the softmax activation function to the attention scores
        # 5. Apply the attention dropout to the attention weights
        # 6. Compute the attention output by multiplying the attention weights with the value tensor
        # 7. organize the attention output to [bs, seq_len, hidden_size] and return

        # Hint: use torch.finfo(attn_scores.dtype).min to get the mask value


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense_h_to_4h = nn.Linear(
            config.hidden_size, config.intermediate_size)
        self.dense_4h_to_h = nn.Linear(
            config.intermediate_size, config.hidden_size)

    def _gelu(self, x):
        # (TODO) Task 1: Implement the GELU activation function
        # ref: https://pytorch.org/docs/stable/generated/torch.nn.GELU.html

    def forward(self, hidden_states):
        # (TODO) Task 1: Implement the FeedForward forward pass


class PythiaLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.use_parallel_residual = config.use_parallel_residual
        self.input_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_dropout = nn.Dropout(config.hidden_dropout)
        self.post_mlp_dropout = nn.Dropout(config.hidden_dropout)
        self.attention = Attention(config)
        self.mlp = FeedForward(config)

    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor],
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        layer_kvcache: Optional[Tuple[torch.Tensor]] = None,
    ):
        # pseudocode:
        # x = x + attn(ln1(x)) + mlp(ln2(x))
        attention_layer_outputs = self.attention(
            self.input_layernorm(hidden_states),
            attention_mask=attention_mask,
            position_ids=position_ids,
            layer_kvcache=layer_kvcache,
        )
        # output_attn: attn_output, present, (attn_weights)
        attn_output = attention_layer_outputs[0]
        attn_output = self.post_attention_dropout(attn_output)
        outputs = attention_layer_outputs[1:]

        mlp_output = self.mlp(self.post_attention_layernorm(hidden_states))
        mlp_output = self.post_mlp_dropout(mlp_output)
        hidden_states = mlp_output + attn_output + hidden_states

        return (hidden_states,) + outputs


class PythiaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embed_in = nn.Embedding(config.vocab_size, config.hidden_size)
        self.emb_dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList([PythiaLayer(config)
                                    for _ in range(config.num_hidden_layers)])
        self.final_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)

        self._attn_implementation = config._attn_implementation

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        kvcaches: Optional[Tuple[KVCache]] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        input_shape = input_ids.size()

        batch_size, seq_length = input_shape

        if kvcaches is None:
            past_length = 0
            kvcaches = tuple([KVCache() for _ in range(self.config.num_hidden_layers)])
        else:
            past_length = kvcaches[0].past_key_values_length()

        position_ids = torch.arange(
            past_length, seq_length + past_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0)

        inputs_embeds = self.embed_in(input_ids)

        # Attention mask.
        attention_mask = attention_mask.view(
            batch_size, -1) if attention_mask is not None else None

        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask=attention_mask,
            input_shape=(batch_size, seq_length),
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_length,
        )

        hidden_states = self.emb_dropout(inputs_embeds)

        presents = ()
        all_attentions = None
        all_hidden_states = None
        for i, (layer, layer_kvcache) in enumerate(zip(self.layers, kvcaches)):
            outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                layer_kvcache=layer_kvcache,
            )
            hidden_states = outputs[0]
            presents = presents + (outputs[1],)

        hidden_states = self.final_layer_norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            kvcaches=presents,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class PythiaForCausalLM(PreTrainedModel):
    config_class = GPTNeoXConfig
    base_model_prefix = "gpt_neox"
    _tied_weights_keys = ["embed_out.weight"]

    def __init__(self, config):
        super().__init__(config)

        self.gpt_neox = PythiaModel(config)
        self.embed_out = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        kvcaches: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        outputs = self.gpt_neox(
            input_ids,
            attention_mask=attention_mask,
            kvcaches=kvcaches,
        )

        hidden_states = outputs[0]
        lm_logits = self.embed_out(hidden_states)

        return CausalLMOutputWithPast(
            logits=lm_logits,
            kvcaches=outputs.kvcaches,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
