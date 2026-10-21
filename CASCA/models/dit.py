import math
import typing

import einops
from einops import rearrange
from functools import partial
try:
  import flash_attn
  import flash_attn.layers.rotary
except:
  pass
import huggingface_hub
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
  from torch.nn.attention.flex_attention import flex_attention, create_block_mask
  FLEX_ATTN_AVAILABLE = True
except:
  FLEX_ATTN_AVAILABLE = False

# Flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)

def xmidi_row_adaptive_block_mask(
    b, h, q_idx, kv_idx, block_size=None, n=None,
    num_cols=None, row_unk_mask=None):
  """Memory-efficient V2 block mask with per-sample inactive rows."""
  base = block_diff_mask(
    b, h, q_idx, kv_idx, block_size=block_size, n=n)
  q_row = (q_idx % n) // num_cols
  kv_row = (kv_idx % n) // num_cols
  q_active = ~row_unk_mask[b, q_row]
  kv_active = ~row_unk_mask[b, kv_row]
  # An inactive query gets one safe edge; its output is zeroed afterwards.
  safe_inactive = (~q_active) & (q_idx == kv_idx)
  return (base & q_active & kv_active) | safe_inactive


def block_diff_mask(b, h, q_idx, kv_idx, block_size=None, n=None):
  """
  Constructs the specialized block diffusion attention mask for training
  composed of three masks:
  - **Block Diagonal Mask (M_BD)**: Self-attention within noised blocks
  - **Offset Block Causal Mask (M_OBC)**: Cross-attention for conditional context
  - **Block Causal Mask (M_BC)**: Attention to update x0

  Args:
      b, h: Batch and head indices (ignored for mask logic).
      q_idx, kv_idx: Query and Key indices.
      seq_len: Total sequence length.
      block_size: Defines the block structure.

  Returns:
      A boolean attention mask.
  """

  # Indicate whether token belongs to xt or x0
  x0_flag_q = (q_idx >= n)
  x0_flag_kv = (kv_idx >= n)

  # Compute block indices
  block_q = torch.where(x0_flag_q == 1,
                        (q_idx - n) // block_size,
                        q_idx // block_size)
  block_kv = torch.where(x0_flag_kv == 1,
                        (kv_idx - n) // block_size,
                        kv_idx // block_size)

  # **1. Block Diagonal Mask (M_BD) **
  block_diagonal = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)

  # **2. Offset Block-Causal Mask (M_OBC) **
  offset_block_causal = (
    (block_q > block_kv)
    & (x0_flag_kv == 1)
    & (x0_flag_q == 0)
  )

  # **3. Block-Causal Mask (M_BC) **
  block_causal = (block_q >= block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 1)

  # **4. Combine Masks **
  return block_diagonal | offset_block_causal | block_causal

@torch.compile(fullgraph=True, mode="max-autotune-no-cudagraphs")
def fused_flex_attention(q, k, v, mask=None):
    return flex_attention(q, k, v, block_mask=mask)


def bias_dropout_add_scale(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float,
    training: bool) -> torch.Tensor:
  if bias is not None:
    out = scale * F.dropout(x + bias, p=prob, training=training)
  else:
    out = scale * F.dropout(x, p=prob, training=training)

  if residual is not None:
    out = residual + out
  return out


def get_bias_dropout_add_scale(training):
  def _bias_dropout_add(x, bias, scale, residual, prob):
    return bias_dropout_add_scale(
      x, bias, scale, residual, prob, training)

  return _bias_dropout_add


# function overload
def modulate(x: torch.Tensor,
             shift: torch.Tensor,
             scale: torch.Tensor) -> torch.Tensor:
  return x * (1 + scale) + shift

@torch.jit.script
def bias_dropout_add_scale_fused_train(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float) -> torch.Tensor:
  return bias_dropout_add_scale(
    x, bias, scale, residual, prob, True)

@torch.jit.script
def bias_dropout_add_scale_fused_inference(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float) -> torch.Tensor:
  return bias_dropout_add_scale(
    x, bias, scale, residual, prob, False)

@torch.jit.script
def modulate_fused(x: torch.Tensor,
                   shift: torch.Tensor,
                   scale: torch.Tensor) -> torch.Tensor:
  return modulate(x, shift, scale)


class Rotary(torch.nn.Module):
  def __init__(self, dim, base=10_000):
    super().__init__()
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    self.register_buffer('inv_freq', inv_freq)
    self.seq_len_cached = None
    self.cos_cached = None
    self.sin_cached = None

  def forward(self, x, seq_dim=1):
    seq_len = x.shape[seq_dim]
    if seq_len != self.seq_len_cached:
      self.seq_len_cached = seq_len
      t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
      freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone())
      emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
      # dims are: batch, seq_len, qkv, head, dim
      self.cos_cached = emb.cos()[None, :, None, None, :].repeat(1,1,3,1,1)
      self.sin_cached = emb.sin()[None, :, None, None, :].repeat(1,1,3,1,1)
      # This makes the transformation on v an identity.
      self.cos_cached[:,:,2,:,:].fill_(1.)
      self.sin_cached[:,:,2,:,:].fill_(0.)

    return self.cos_cached, self.sin_cached


def rotate_half(x):
  x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
  return torch.cat((-x2, x1), dim=-1)


def rotate_half_axis_local(x, axis_dims=None):
  """Rotate paired channels independently inside each positional axis."""
  if axis_dims is None:
    return rotate_half(x)
  if sum(axis_dims) != x.shape[-1]:
    raise ValueError(
      f'axis_dims={axis_dims} do not cover head_dim={x.shape[-1]}')
  parts = torch.split(x, tuple(int(dim) for dim in axis_dims), dim=-1)
  if any(part.shape[-1] % 2 for part in parts):
    raise ValueError(f'Every rotary axis must have even width, got {axis_dims}')
  return torch.cat([rotate_half(part) for part in parts], dim=-1)


def unpack_rotary(rotary_cos_sin):
  if len(rotary_cos_sin) == 2:
    cos, sin = rotary_cos_sin
    return cos, sin, None
  if len(rotary_cos_sin) == 3:
    return rotary_cos_sin
  raise ValueError('rotary embedding must return (cos, sin[, axis_dims])')


def split_and_apply_rotary_pos_emb(qkv, rotary_cos_sin):
  with torch.amp.autocast('cuda', enabled=False):
    cos, sin, axis_dims = unpack_rotary(rotary_cos_sin)
    if axis_dims is not None:
      qkv = apply_rotary_pos_emb_torchscript(
        qkv, cos.to(qkv.dtype), sin.to(qkv.dtype), axis_dims)
      return qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
    cos = cos.to(qkv.dtype)
    sin = sin.to(qkv.dtype)
    cos = cos[0,:,0,0,:cos.shape[-1]//2]
    sin = sin[0,:,0,0,:sin.shape[-1]//2]
    q, k, v = qkv.chunk(3, dim=2)
    q = flash_attn.layers.rotary.apply_rotary_emb_torch(
      q.squeeze(dim=2), cos, sin)
    k = flash_attn.layers.rotary.apply_rotary_emb_torch(
      k.squeeze(dim=2), cos, sin)
    v = v.squeeze(dim=2)
  return q, k, v


def apply_rotary_pos_emb_torchscript(qkv, cos, sin, axis_dims=None):
  return (qkv * cos) + (rotate_half_axis_local(qkv, axis_dims) * sin)


def apply_rotary_pos_emb(qkv, cos, sin):
  cos = cos[0,:,0,0,:cos.shape[-1]//2]
  sin = sin[0,:,0,0,:sin.shape[-1]//2]
  return flash_attn.layers.rotary.apply_rotary_emb_qkv_(qkv, cos, sin)


def regular_attention_multi_headed(q, k, v):
  # Assuming qkv is a tensor with shape [batch, seq_len, 3, num_heads, head_dim]
  # where the 3 represents Q, K, V packed in that order
  attention_output = F.scaled_dot_product_attention(
    query=q.transpose(1, 2),
    key=k.transpose(1, 2),
    value=v.transpose(1, 2),
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False)
  # [batch_size, seq_len, num_heads, head_dim]
  attention_output = attention_output.transpose(1, 2)
  return einops.rearrange(attention_output, 'b s h d -> b s (h d)')


#################################################################################
#                                  Layers                                       #
#################################################################################
class LayerNorm(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.weight = nn.Parameter(torch.ones([dim]))
    self.dim = dim
  def forward(self, x):
    with torch.amp.autocast('cuda', enabled=False):
      x = F.layer_norm(x.float(), [self.dim])
    return x * self.weight[None, None, :]


def residual_linear(x, W, x_skip, residual_scale):
  """x_skip + residual_scale * W @ x"""
  dim_out, dim_in = W.shape[0], W.shape[1]
  return torch.addmm(
    x_skip.view(-1, dim_out),
    x.view(-1, dim_in),
    W.T,
    alpha=residual_scale).view(*x.shape[:-1], dim_out)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################
class TimestepEmbedder(nn.Module):
  """
  Embeds scalar timesteps into vector representations.
  """
  def __init__(self, hidden_size, frequency_embedding_size=256):
    super().__init__()
    self.mlp = nn.Sequential(
      nn.Linear(frequency_embedding_size, hidden_size, bias=True),
      nn.SiLU(),
      nn.Linear(hidden_size, hidden_size, bias=True))
    self.frequency_embedding_size = frequency_embedding_size

  @staticmethod
  def timestep_embedding(t, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
    half = dim // 2
    freqs = torch.exp(
      - math.log(max_period)
      * torch.arange(start=0, end=half).to(t.dtype).to(t.device)
      / half)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
      embedding = torch.cat(
        [embedding,
         torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

  def forward(self, t):
    t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
    t_emb = self.mlp(t_freq)
    return t_emb


class LabelEmbedder(nn.Module):
  """Embeds class labels into vector representations.
  
  Also handles label dropout for classifier-free guidance.
  """
  def __init__(self, num_classes, cond_size):
    super().__init__()
    self.embedding_table = nn.Embedding(num_classes + 1, cond_size)
    self.num_classes = num_classes

    # TODO think of initializing with 0.02 std deviation like in original DiT paper

  def forward(self, labels):
    embeddings = self.embedding_table(labels)
    return embeddings
    

#################################################################################
#                                 Core Model                                    #
#################################################################################

class DDiTBlockCausal(nn.Module):
  def __init__(self, n, dim, n_heads, mlp_ratio=4, dropout=0.1, max_batch_size=64, max_seqlen=1024, adaLN=False, cond_dim=None, attn_backend='flash_attn'):
    super().__init__()
    self.n_heads = n_heads
    self.max_seqlen = max_seqlen
    self.n = n

    self.norm1 = LayerNorm(dim)
    self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
    self.attn_out = nn.Linear(dim, dim, bias=False)
    self.dropout1 = nn.Dropout(dropout)

    self.norm2 = LayerNorm(dim)
    self.mlp = nn.Sequential(
      nn.Linear(dim, mlp_ratio * dim, bias=True),
      nn.GELU(approximate='tanh'),
      nn.Linear(mlp_ratio * dim, dim, bias=True))
    self.dropout2 = nn.Dropout(dropout)
    self.dropout = dropout
    self.adaLN = adaLN
    if self.adaLN:
      self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim)
      self.adaLN_modulation.weight.data.zero_()
      self.adaLN_modulation.bias.data.zero_()
    self.attn_backend = attn_backend
    self.kv_cache = None

  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return bias_dropout_add_scale_fused_inference

  def get_qkv(self, x, rotary_cos_sin, store_kv=False):
    # compute qkv (potentially use cache)
    if self.kv_cache is not None:
      new_qkv = self.attn_qkv(x[:, -1:])
      qkv = torch.cat((self.kv_cache, new_qkv), dim=1)
    else:
      qkv = self.attn_qkv(x)
    # store kv cache in a sliding window (can't exceed context len)
    if store_kv:
      self.kv_cache = qkv[:, -(self.max_seqlen-1):].clone()
      
    qkv = einops.rearrange(
      qkv,
      'b s (three h d) -> b s three h d',
      three=3,
      h=self.n_heads)
    with torch.amp.autocast('cuda', enabled=False):
      cos, sin, axis_dims = unpack_rotary(rotary_cos_sin)
      if self.attn_backend == 'flash_attn' and axis_dims is None:
        qkv = apply_rotary_pos_emb(
          qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
      else:
        qkv = apply_rotary_pos_emb_torchscript(
          qkv, cos.to(qkv.dtype), sin.to(qkv.dtype), axis_dims)
          
    return qkv

  def cross_attn(self, qkv, mask=None):
    scale = qkv.shape[-1]
    qkv = qkv.transpose(1, 3)
    mask = mask.bool() if mask is not None else None
    x = F.scaled_dot_product_attention(
      query=qkv[:, :, 0],
      key=qkv[:, :, 1],
      value=qkv[:, :, 2],
      attn_mask=mask,
      is_causal=True,
      scale=1 / math.sqrt(scale))
    x = x.transpose(1, 2)
    x = rearrange(x, 'b s h d -> b s (h d)')
    return x

  def forward(self,
              x,
              rotary_cos_sin,
              c=None,
              causal=True,
              mask=None,
              store_kv=False,
              **kwargs):
    del kwargs
    batch_size, seq_len = x.shape[0], x.shape[1]
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = None, None, None, None, None, None
    bias_dropout_scale_fn = self._get_bias_dropout_scale()
    if c is not None and c.shape[0] == batch_size:
      (shift_msa, scale_msa, gate_msa, shift_mlp,
      scale_mlp, gate_mlp) = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
    elif c is not None:
      (shift_msa, scale_msa, gate_msa, shift_mlp,
      scale_mlp, gate_mlp) = rearrange(
        self.adaLN_modulation(c), '(b h) d -> b h d', b=batch_size
        ).chunk(6, dim=-1)

    # attention operation
    x_skip = x
    if c is not None:
      x = modulate_fused(self.norm1(x), shift_msa, scale_msa)
    else:
      x = self.norm1(x)
    
    qkv = self.get_qkv(x, rotary_cos_sin, store_kv=store_kv)
    if self.attn_backend == 'flash_attn':
      qkv = einops.rearrange(qkv, 'b s ... -> (b s) ...')
      cu_seqlens = torch.arange(
        0, (batch_size + 1) * seq_len,
        step=seq_len, dtype=torch.int32, device=qkv.device)
      x = flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func(
        qkv, cu_seqlens, seq_len, 0.0, causal=True)
      x = einops.rearrange(x, '(b s) h d -> b s (h d)', b=batch_size)
    else:
      x = self.cross_attn(qkv, c)
      
    if c is not None:
      x = bias_dropout_scale_fn(self.attn_out(x),
        None,
        gate_msa,
        x_skip,
        self.dropout)
      # mlp operation
      x = bias_dropout_scale_fn(
        self.mlp(modulate_fused(
          self.norm2(x), shift_mlp, scale_mlp)),
        None, gate_mlp, x, self.dropout)
    else:
      scale = torch.ones(1, device=x.device, dtype=x.dtype)
      x = bias_dropout_scale_fn(
        self.attn_out(x), None, scale, x_skip, self.dropout)
      x = bias_dropout_scale_fn(
        self.mlp(self.norm2(x)), None, scale, x, self.dropout)
    return x


class DDiTBlock(nn.Module):
  def __init__(self, n, dim, n_heads, adaLN,
               latent_dim=None, cond_dim=None,
               latent_conditioning=-1, mlp_ratio=4,
               dropout=0.1, block_size=1,
               max_batch_size=64, max_seqlen=1024, attn_backend='flash_attn'):
    super().__init__()
    self.max_seqlen = max_seqlen
    self.n = n
    self.n_heads = n_heads
    self.adaLN = adaLN
    self.latent_conditioning = latent_conditioning
    self.block_size = block_size

    self.norm1 = LayerNorm(dim)
    self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
    self.attn_out = nn.Linear(dim, dim, bias=False)
    self.dropout1 = nn.Dropout(dropout)

    self.norm2 = LayerNorm(dim)
    self.mlp = nn.Sequential(
      nn.Linear(dim, mlp_ratio * dim, bias=True),
      nn.GELU(approximate='tanh'),
      nn.Linear(mlp_ratio * dim, dim, bias=True))
    self.dropout2 = nn.Dropout(dropout)
    self.dropout = dropout
    self.kv_cache = None
    self.cache_idx = 0

    if self.adaLN:
      self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim)
      self.adaLN_modulation.weight.data.zero_()
      self.adaLN_modulation.bias.data.zero_()
    self.attn_backend = attn_backend

  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return bias_dropout_add_scale_fused_inference

  def get_qkv(self, x, rotary_cos_sin, store_kv=False):
    """Return packed QKV, or current Q with cached-prefix/current K/V."""
    qkv = einops.rearrange(
      self.attn_qkv(x),
      'b s (three h d) -> b s three h d',
      three=3,
      h=self.n_heads)
    cos, sin, axis_dims = unpack_rotary(rotary_cos_sin)

    if self.kv_cache is None:
      with torch.amp.autocast('cuda', enabled=False):
        if self.attn_backend == 'flash_attn' and axis_dims is None:
          return apply_rotary_pos_emb(
            qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
        return apply_rotary_pos_emb_torchscript(
          qkv, cos.to(qkv.dtype), sin.to(qkv.dtype), axis_dims)

    q_len = qkv.shape[1]
    start = self.cache_idx
    end = start + q_len
    if end > self.max_seqlen:
      raise RuntimeError(
        f'KV cache overflow: prefix={start}, q_len={q_len}, '
        f'capacity={self.max_seqlen}')
    if cos.shape[1] < end:
      raise ValueError(
        f'RoPE length {cos.shape[1]} is shorter than cache end {end}')
    current_rotary = (
      cos[:, start:end], sin[:, start:end], axis_dims
    ) if axis_dims is not None else (cos[:, start:end], sin[:, start:end])
    current_qkv = apply_rotary_pos_emb_torchscript(
      qkv,
      current_rotary[0].to(qkv.dtype),
      current_rotary[1].to(qkv.dtype),
      axis_dims)
    current_q = current_qkv[:, :, 0]
    current_kv = current_qkv[:, :, 1:]
    prefix_kv = self.kv_cache[:, :start]
    all_kv = torch.cat((prefix_kv, current_kv), dim=1)

    if store_kv:
      self.kv_cache[:, start:end].copy_(current_kv.detach())
      self.cache_idx = end
    return current_q, all_kv[:, :, 0], all_kv[:, :, 1]
  
  def attn_mlp(self, x, c, gate_msa, gate_mlp, shift_mlp, scale_mlp, x_skip):
    bias_dropout_scale_fn = self._get_bias_dropout_scale()
    if c is not None:
      x = bias_dropout_scale_fn(self.attn_out(x),
        None,
        gate_msa,
        x_skip,
        self.dropout)
      # mlp operation
      x = bias_dropout_scale_fn(
        self.mlp(modulate_fused(
          self.norm2(x), shift_mlp, scale_mlp)),
        None, gate_mlp, x, self.dropout)
    else:
      scale = torch.ones(1, device=x.device, dtype=x.dtype)
      x = bias_dropout_scale_fn(
        self.attn_out(x), None, scale, x_skip, self.dropout)
      x = bias_dropout_scale_fn(
        self.mlp(self.norm2(x)), None, scale, x, self.dropout)
    return x

  def cross_attn(self, qkv, mask=None):
    if isinstance(qkv, tuple):
      q, k, v = qkv
      scale = q.shape[-1]
      q = q.transpose(1, 2)
      k = k.transpose(1, 2)
      v = v.transpose(1, 2)
    else:
      scale = qkv.shape[-1]
      qkv = qkv.transpose(1, 3)
      q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
    mask = mask.bool() if mask is not None else None
    x = F.scaled_dot_product_attention(
      query=q,
      key=k,
      value=v,
      attn_mask=mask,
      is_causal=False,
      scale=1 / math.sqrt(scale))
    x = x.transpose(1, 2)
    x = rearrange(x, 'b s h d -> b s (h d)')
    return x

  def cross_attn_flex(self, qkv, mask=None):
    qkv = rearrange(qkv, 'b s three h d -> b h three s d', h=self.n_heads)
    x = fused_flex_attention(
      qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2], mask=mask)
    x = rearrange(x, 'b h s d -> b s (h d)')
    return x

  def forward(self,
              x,
              rotary_cos_sin,
              c,
              causal=False,
              mask=None,
              sample_mode=False,
              store_kv=False,
              query_active=None):
    batch_size, seq_len = x.shape[0], x.shape[1]

    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = None, None, None, None, None, None
    if c is not None and c.shape[0] == batch_size:
      (shift_msa, scale_msa, gate_msa, shift_mlp,
      scale_mlp, gate_mlp) = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
    elif c is not None:
      (shift_msa, scale_msa, gate_msa, shift_mlp,
      scale_mlp, gate_mlp) = rearrange(
        self.adaLN_modulation(c), '(b h) d -> b h d', b=batch_size
        ).chunk(6, dim=-1)

    x_skip = x
    if c is not None:
      x = modulate_fused(self.norm1(x), shift_msa, scale_msa)
    else:
      x = self.norm1(x)

    # get qkvs
    if mask is not None and not sample_mode:
      qkv_x = self.get_qkv(x[:,:self.n], rotary_cos_sin)
      qkv_x0 = self.get_qkv(x[:,self.n:], rotary_cos_sin)
      qkv = torch.cat((qkv_x, qkv_x0), dim=1)
    else:
      qkv = self.get_qkv(x, rotary_cos_sin, store_kv=store_kv)
      
    # Cached sampling has non-square Q/KV and therefore uses SDPA directly.
    if isinstance(qkv, tuple):
      x = self.cross_attn(qkv, mask=mask)
    elif self.attn_backend == 'flash_attn' and mask is None:
      qkv = einops.rearrange(qkv, 'b s ... -> (b s) ...')
      cu_seqlens = torch.arange(
        0, (batch_size + 1) * seq_len, step=seq_len,
        dtype=torch.int32, device=qkv.device)
      x = flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func(
        qkv, cu_seqlens, seq_len, 0., causal=causal)
      x = rearrange(x, '(b s) h d -> b s (h d)', b=batch_size)     
    elif self.attn_backend == 'flex' and FLEX_ATTN_AVAILABLE:
      x = self.cross_attn_flex(qkv, mask=mask)
    elif self.attn_backend == 'sdpa':
      x = self.cross_attn(qkv, mask=mask)
    else:
      raise ValueError('Unknown attention backend')
    if self.kv_cache is not None:
      x = x[:, -self.block_size:]
    if query_active is not None:
      if query_active.shape != x.shape[:2]:
        raise ValueError(
          f'query_active shape {tuple(query_active.shape)} != {tuple(x.shape[:2])}')
      x = x * query_active.to(dtype=x.dtype).unsqueeze(-1)
    x = self.attn_mlp(x, c, gate_msa, gate_mlp, shift_mlp, scale_mlp, x_skip)
    return x
   
class EmbeddingLayer(nn.Module):
  def __init__(self, dim, vocab_dim):
    super().__init__()
    self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
    torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

  def forward(self, x):
    return self.embedding[x]


class DDiTFinalLayer(nn.Module):
  def __init__(self, hidden_size, out_channels, cond_dim,
               adaLN, tie_word_embeddings=False):
    super().__init__()
    self.norm_final = LayerNorm(hidden_size)
    self.adaLN = adaLN
    self.tie_word_embeddings = bool(tie_word_embeddings)
    self.out_channels = int(out_channels)
    if self.tie_word_embeddings:
      self.linear = None
      self.bias = nn.Parameter(torch.zeros(out_channels))
    else:
      self.linear = nn.Linear(hidden_size, out_channels)
      self.linear.weight.data.zero_()
      self.linear.bias.data.zero_()
      self.register_parameter('bias', None)
    if self.adaLN:
      self.adaLN_modulation = nn.Linear(
        cond_dim, 2 * hidden_size, bias=True)
      self.adaLN_modulation.weight.data.zero_()
      self.adaLN_modulation.bias.data.zero_()

  def prepare_hidden(self, x, c):
    x = self.norm_final(x)
    if c is not None:
      if c.shape[0] == x.shape[0]:
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
      else:
        shift, scale = rearrange(
          self.adaLN_modulation(c),
          '(b h) d -> b h d', b=x.shape[0]).chunk(2, dim=-1)
      x = modulate_fused(x, shift, scale)
    return x

  def _projection(self, x, embedding_weight=None):
    if self.tie_word_embeddings:
      if embedding_weight is None:
        raise ValueError('Tied output projection requires embedding_weight')
      return F.linear(x, embedding_weight, self.bias)
    return self.linear(x)

  def forward(self, x, c, embedding_weight=None):
    return self._projection(
      self.prepare_hidden(x, c), embedding_weight=embedding_weight)

  def selected_log_probs(self, x, c, targets, selection_mask,
                         embedding_weight=None, excluded_token_id=None,
                         chunk_size=64):
    """Compute exact target log-probabilities only at selected positions."""
    if targets.shape != x.shape[:2] or selection_mask.shape != targets.shape:
      raise ValueError(
        f'targets/selection must match hidden prefix {tuple(x.shape[:2])}')
    hidden = self.prepare_hidden(x, c)
    selected = selection_mask.bool().nonzero(as_tuple=False)
    output = hidden.new_zeros(targets.shape)
    if selected.numel() == 0:
      # A stochastic diffusion draw can contain no masked target positions.
      # Return an exact zero that remains connected to every output-path
      # parameter; otherwise Lightning calls backward() on a detached scalar
      # on this rank while the other DDP ranks wait in collective operations.
      dummy_logits = self._projection(
        hidden[:1, :1].reshape(1, -1), embedding_weight)
      return output + dummy_logits.sum() * 0.0
    batch_idx, pos_idx = selected[:, 0], selected[:, 1]
    values = []
    chunk_size = max(1, int(chunk_size))
    for start in range(0, selected.shape[0], chunk_size):
      end = min(start + chunk_size, selected.shape[0])
      b = batch_idx[start:end]
      p = pos_idx[start:end]
      logits = self._projection(hidden[b, p], embedding_weight)
      if excluded_token_id is not None:
        logits[:, excluded_token_id] = torch.finfo(logits.dtype).min
      values.append(F.log_softmax(logits, dim=-1).gather(
        1, targets[b, p, None]).squeeze(1))
    output[batch_idx, pos_idx] = torch.cat(values, dim=0)
    return output


class DIT(nn.Module, huggingface_hub.PyTorchModelHubMixin):
  def __init__(self, config, vocab_size: int):
    super().__init__()
    if type(config) == dict:
      config = omegaconf.OmegaConf.create(config)
    self.causal = getattr(config.model, 'causal_attention', config.algo.parameterization == 'ar')
    self.n = config.model.length
    self.adaLN = not self.causal or getattr(config.model, 'adaln', False)
    self.config = config
    self.vocab_size = vocab_size
    self.block_size = config.block_size
    self.output_chunk_size = int(getattr(config.model, 'output_chunk_size', 64))
    dim = config.model.hidden_size
    cond_dim = config.model.cond_dim
    self.n_heads = config.model.n_heads
    self.vocab_embed = EmbeddingLayer(dim, vocab_size)
    if self.adaLN == True:
      self.sigma_map = TimestepEmbedder(cond_dim)
    if not self.causal:
      self.sigma_map = TimestepEmbedder(cond_dim)
    # XMIDI / XMIDI-GetMusic: swap to 2D (row, col) RoPE so the model can
    # distinguish between rows of the (NUM_ROWS, NUM_COLS) grid
    # (R&D §"相对位置编码"). Falls back to 1D RoPE for all other datasets.
    # NUM_ROWS/NUM_COLS come from the matching dataset package. The V2
    # ``xmidi_getmusic`` protocol is 16x512 (4 source + 12 target rows).
    from xmidi_compat import is_xmidi_like, get_xmidi_module
    if is_xmidi_like(config):
      _xm = get_xmidi_module(config)
      Rotary2D = _xm.Rotary2D
      NUM_ROWS = _xm.NUM_ROWS
      NUM_COLS = _xm.NUM_COLS
      self.rotary_emb = Rotary2D(
        dim // config.model.n_heads,
        num_rows=NUM_ROWS,
        num_cols=NUM_COLS)
      # Stash the xmidi(_getmusic) module so ``_maybe_get_xmidi_mask`` can
      # resolve dataset-specific helpers (build_xmidi_attn_mask /
      # _base_block_diff_mask / NUM_COLS) without re-reading config.
      self._xmidi_module = _xm
    else:
      self.rotary_emb = Rotary(dim // config.model.n_heads)
      self._xmidi_module = None
    self.attn_backend = getattr(config.model, 'attn_backend', 'flash_attn')
    # was hard-coded to 1024 (OWT/LM1B); allow override so that XMIDI
    # variants, including V2 length 8192, train without KV-cache buffer
    # overruns or RoPE length mismatches.
    self.max_seqlen = getattr(config.model, 'max_seqlen', config.model.length)

    blocks = []
    for _ in range(config.model.n_blocks):
      if self.causal:
        block = DDiTBlockCausal(
          n=config.model.length,
          dim=dim,
          n_heads=config.model.n_heads,
          dropout=config.model.dropout,
          max_batch_size=config.loader.eval_batch_size,
          adaLN=self.adaLN,
          cond_dim=cond_dim,
          attn_backend=self.attn_backend)
      else:
        block = DDiTBlock(
          n=config.model.length,
          dim=dim,
          n_heads=config.model.n_heads,
          cond_dim=cond_dim,
          adaLN=self.adaLN,
          dropout=config.model.dropout,
          block_size=self.block_size,
          attn_backend=self.attn_backend,
          max_seqlen=self.max_seqlen)
      blocks.append(block)
    self.blocks = nn.ModuleList(blocks)
    self.output_layer = DDiTFinalLayer(
      hidden_size=dim,
      out_channels=vocab_size,
      cond_dim=cond_dim,
      adaLN=self.adaLN,
      tie_word_embeddings=config.model.tie_word_embeddings)
    if config.algo.cross_attn:
      # KV-cache sampling builds a compact (B,1,16,prefix) mask per block.
      # Avoid materialising the unused 16384x16384 SDPA training mask.
      is_cached_sampling = (
        config.mode == 'sample_eval'
        and bool(config.sampling.kv_cache))
      # V2 training always supplies a per-sample row mask, so its Flex block
      # mask is built lazily on the process-local CUDA device in forward().
      # Eager create_block_mask() defaults to cuda:0 and makes every spawned
      # DDP rank allocate there before Lightning assigns local devices.
      uses_dynamic_xmidi_flex_mask = (
        self._xmidi_module is not None
        and self.attn_backend == 'flex')
      if is_cached_sampling or uses_dynamic_xmidi_flex_mask:
        self.block_diff_mask = torch.empty(0, dtype=torch.bool)
      else:
        self.gen_mask(config.model.length, self.block_size, self.attn_backend)

  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return bias_dropout_add_scale_fused_inference
    
  def gen_mask(self, seqlen, block_size, attn_backend='sdpa'):
    """Genererates attention mask"""
    if attn_backend == 'flex' and FLEX_ATTN_AVAILABLE:
      self.block_diff_mask = create_block_mask(
        partial(block_diff_mask, block_size=block_size, n=seqlen),
        B=None, H=None, Q_LEN=seqlen*2, KV_LEN=seqlen*2)
    elif attn_backend == 'sdpa':
      self.block_diff_mask = block_diff_mask(
        b=None, h=None, q_idx=torch.arange(seqlen*2)[:, None], 
        kv_idx=torch.arange(seqlen*2)[None, :], block_size=block_size, n=seqlen)
    else:
      raise ValueError('Unknown attention backend')
    
  def reset_kv_cache(self, eval_batch_size=None):
    self.reset_sampling_audit()
    if eval_batch_size is None:
      eval_batch_size = self.config.loader.eval_batch_size
    device = self.vocab_embed.embedding.device
    cache_dtype = (
      torch.bfloat16 if device.type == 'cuda'
      else self.vocab_embed.embedding.dtype)
    head_dim = self.config.model.hidden_size // self.n_heads
    for block in self.blocks:
      if isinstance(block, DDiTBlock):
        block.kv_cache = torch.zeros(
          eval_batch_size,
          self.max_seqlen,
          2,
          self.n_heads,
          head_dim,
          device=device,
          dtype=cache_dtype)
        block.cache_idx = 0
      else:
        block.kv_cache = None

  def kv_cache_length(self):
    lengths = [block.cache_idx for block in self.blocks
               if isinstance(block, DDiTBlock)]
    if not lengths:
      return 0
    if len(set(lengths)) != 1:
      raise RuntimeError(f'Inconsistent cache lengths across layers: {lengths}')
    return lengths[0]

  def reset_sampling_audit(self):
    self._sampling_audit = {
      'attention_mask_valid': False,
      'current_query_contract': False,
      'cache_consistent': False,
      'absolute_rope': False,
      'forward_calls': 0,
      'committed_tokens': 0,
    }

  def get_sampling_audit(self):
    if not hasattr(self, '_sampling_audit'):
      self.reset_sampling_audit()
    return dict(self._sampling_audit)

  # ---------------------------------------------------------------------
  # XMIDI hook: per-batch UNK-row mask
  #
  # The base ``block_diff_mask`` (paper Suppl. B.6) is dataset-agnostic and
  # is materialised once in ``gen_mask`` at __init__. For XMIDI we further
  # need to disable attention to/from rows that are entirely <UNK> (R&D
  # §"局部 attention 算法"). The set of such rows is sample-specific, so we
  # accept a (B, num_rows) bool tensor here and rebuild a (B, 1, 2L, 2L)
  # mask on-the-fly inside ``forward``.
  #
  # ``Diffusion._loss`` calls ``set_xmidi_row_unk_mask`` once per training
  # step (and clears it afterwards) so this state is implicitly per-step.
  # When None the original static mask is used (full back-compat with all
  # other datasets).
  # ---------------------------------------------------------------------
  def set_xmidi_row_unk_mask(self, row_unk_mask):
    """Register per-sample ``(B, num_rows)`` inactive-track flags."""
    if row_unk_mask is not None:
      expected_rows = self._xmidi_module.NUM_ROWS
      if row_unk_mask.dim() != 2 or row_unk_mask.shape[1] != expected_rows:
        raise ValueError(
          f'row_unk_mask must be (B, {expected_rows}), got '
          f'{tuple(row_unk_mask.shape)}')
      row_unk_mask = row_unk_mask.bool()
    self._xmidi_row_unk_mask = row_unk_mask

  def _xmidi_training_query_active(self):
    row_unk_mask = getattr(self, '_xmidi_row_unk_mask', None)
    if row_unk_mask is None or self._xmidi_module is None:
      return None
    active = self._xmidi_module.attention_active_mask(row_unk_mask)
    return torch.cat((active, active), dim=1)

  def _maybe_get_xmidi_mask(self, default_mask, x_shape, device):
    """Build a per-sample mask if a row_unk_mask is set, else return default.

    Cached across forward passes for the same (seqlen, block_size, device)
    triplet to avoid the O(L^2) base-mask recompute on every call.
    """
    row_unk_mask = getattr(self, '_xmidi_row_unk_mask', None)
    if row_unk_mask is None:
      if isinstance(default_mask, torch.Tensor) and default_mask.numel() == 0:
        raise RuntimeError(
          'XMIDI FlexAttention requires a per-sample row_unk_mask before '
          'forward; call set_xmidi_row_unk_mask().')
      return default_mask
    # Resolve helpers from the xmidi(_getmusic) module that was selected at
    # __init__ time. Falls back to the original ``xmidi`` package if for
    # some reason ``_xmidi_module`` was not set (e.g. legacy ckpts).
    _xm = getattr(self, '_xmidi_module', None)
    if _xm is None:
      from xmidi import build_xmidi_attn_mask, _base_block_diff_mask, NUM_COLS
    else:
      build_xmidi_attn_mask = _xm.build_xmidi_attn_mask
      _base_block_diff_mask = _xm._base_block_diff_mask
      NUM_COLS = _xm.NUM_COLS
    seqlen = self.n
    if self.attn_backend == 'flex' and FLEX_ATTN_AVAILABLE:
      return create_block_mask(
        partial(
          xmidi_row_adaptive_block_mask,
          block_size=self.block_size,
          n=seqlen,
          num_cols=NUM_COLS,
          row_unk_mask=row_unk_mask),
        B=row_unk_mask.shape[0],
        H=None,
        Q_LEN=2 * seqlen,
        KV_LEN=2 * seqlen,
        device=str(device))
    cache_key = (seqlen, self.block_size, str(device))
    cache = getattr(self, '_xmidi_base_mask_cache', {})
    if cache_key not in cache:
      cache[cache_key] = _base_block_diff_mask(
        seqlen, self.block_size, device=device)
      self._xmidi_base_mask_cache = cache
    return build_xmidi_attn_mask(
      seqlen=seqlen,
      block_size=self.block_size,
      row_unk_mask=row_unk_mask,
      num_cols=NUM_COLS,
      base_mask=cache[cache_key],
    )

  def _build_xmidi_sampling_attn_mask(
      self, q_start, q_len, kv_len, device):
    """Return a safe ``(B,1,Q,KV)`` mask and ``(B,Q)`` query activity."""
    if not hasattr(self, '_sampling_audit'):
      self.reset_sampling_audit()
    row_unk_mask = getattr(self, '_xmidi_row_unk_mask', None)
    if row_unk_mask is None:
      return None, None
    _xm = self._xmidi_module
    q_positions = torch.arange(q_start, q_start + q_len, device=device)
    kv_positions = torch.arange(kv_len, device=device)
    q_active = _xm.query_active_from_rows(
      row_unk_mask, q_positions, num_cols=_xm.NUM_COLS)
    kv_active = _xm.query_active_from_rows(
      row_unk_mask, kv_positions, num_cols=_xm.NUM_COLS)
    mask = q_active[:, None, :, None] & kv_active[:, None, None, :]

    # Numerically safe self edge for an inactive current query. The block
    # subsequently zeros its attention update using q_active.
    fallback = torch.zeros_like(mask)
    local = torch.arange(q_len, device=device)
    fallback[:, 0, local, q_start + local] = ~q_active
    mask = mask | fallback
    forbidden = q_active[:, None, :, None] & (~kv_active[:, None, None, :])
    if torch.any(mask & forbidden):
      self._sampling_audit['attention_mask_valid'] = False
      raise RuntimeError('Active query can attend an inactive key/value')
    if not torch.all(mask.any(dim=-1)):
      raise RuntimeError('Sampling attention contains an all-False query')
    self._sampling_audit['attention_mask_valid'] = True
    self._sampling_audit['forward_calls'] += 1
    return mask, q_active


  def forward(self, indices, sigma, sample_mode=False, store_kv=False,
              target_ids=None, output_positions=None,
              excluded_token_id=None):
    x = self.vocab_embed(indices)
    if sigma is None:
      t_cond = None
    else:
      t_cond = F.silu(self.sigma_map(sigma))

    cross_attn = hasattr(self, 'block_diff_mask')
    query_active = None
    cache_start = None
    cache_q_len = None
    if cross_attn:
      mask = self.block_diff_mask
      # special cases for sampling
      if sample_mode:
        if self.config.sampling.kv_cache:
          q_len = x.shape[1]
          q_start = self.kv_cache_length()
          cache_start = q_start
          cache_q_len = q_len
          kv_len = q_start + q_len
          mask, query_active = self._build_xmidi_sampling_attn_mask(
            q_start=q_start,
            q_len=q_len,
            kv_len=kv_len,
            device=x.device)
          # Build absolute phases through the current block. Each layer slices
          # [q_start:q_start+q_len] while cached K/V retain their old phases.
          x_full = x.new_zeros((x.shape[0], kv_len, x.shape[2]))
          rotary_cos_sin = self.rotary_emb(x_full)
          cos = rotary_cos_sin[0]
          axis_dims = rotary_cos_sin[2] if len(rotary_cos_sin) > 2 else None
          absolute_ok = (
            cos.shape[1] == kv_len
            and q_start + q_len == kv_len
            and axis_dims is not None
            and sum(axis_dims) == self.config.model.hidden_size // self.n_heads)
          self._sampling_audit['absolute_rope'] = bool(absolute_ok)
          if not absolute_ok:
            raise RuntimeError('Invalid absolute factorised 2-D RoPE contract')
        else:
          # index block-causal mask only during sampling
          mask = mask[
            self.n:self.n+x.shape[1], self.n:self.n+x.shape[1]]
          row_unk_mask = getattr(self, '_xmidi_row_unk_mask', None)
          if row_unk_mask is not None and self._xmidi_module is not None:
            q_len = x.shape[1]
            adaptive, query_active = self._build_xmidi_sampling_attn_mask(
              q_start=0, q_len=q_len, kv_len=q_len, device=x.device)
            mask = mask.bool().view(1, 1, q_len, q_len) & adaptive
            # Restore the safe diagonal when the static mask removed it.
            inactive = ~query_active
            diagonal = torch.eye(
              q_len, dtype=torch.bool, device=x.device)[None, None]
            mask = mask | (inactive[:, None, :, None] & diagonal)
          rotary_cos_sin = self.rotary_emb(x)

      else:
        # Training uses a safe self edge for inactive queries and then zeros
        # their attention updates inside every block.
        mask = self._maybe_get_xmidi_mask(mask, x.shape, x.device)
        query_active = self._xmidi_training_query_active()
        rotary_cos_sin = self.rotary_emb(x[:, :self.n])

    else:
      rotary_cos_sin = self.rotary_emb(x)
      mask = None

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
      for i in range(len(self.blocks)):
        x = self.blocks[i](
          x,
          rotary_cos_sin,
          c=t_cond,
          causal=self.causal,
          sample_mode=sample_mode,
          mask=mask,
          store_kv=store_kv,
          query_active=query_active)
      if cache_start is not None:
        observed = self.kv_cache_length()
        expected = cache_start + cache_q_len if store_kv else cache_start
        if observed != expected:
          self._sampling_audit['cache_consistent'] = False
          raise RuntimeError(
            f'KV cache length {observed}, expected {expected}')
        self._sampling_audit['cache_consistent'] = True
        if cache_q_len != indices.shape[1]:
          self._sampling_audit['current_query_contract'] = False
          raise RuntimeError('Cached Q length differs from current input block')
        self._sampling_audit['current_query_contract'] = True
        self._sampling_audit['committed_tokens'] = observed
      if cross_attn and not sample_mode:
        x = x[:, :self.n]
      embedding_weight = (
        self.vocab_embed.embedding
        if self.output_layer.tie_word_embeddings else None)
      if output_positions is not None:
        if target_ids is None:
          raise ValueError('target_ids are required with output_positions')
        x = self.output_layer.selected_log_probs(
          x, t_cond,
          targets=target_ids,
          selection_mask=output_positions,
          embedding_weight=embedding_weight,
          excluded_token_id=excluded_token_id,
          chunk_size=self.output_chunk_size)
      else:
        x = self.output_layer(
          x, t_cond, embedding_weight=embedding_weight)
    return x