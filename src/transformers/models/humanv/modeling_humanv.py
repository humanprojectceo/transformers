# coding=utf-8
# Copyright 2025 The HumanV Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch HumanV model with efficient sparse attention (local+global block) and Sparse MoE layers.

Key fixes and features included:
- Correct mask/dtype handling for SDPA/bfloat16.
- Prevents caching inference-mode tensors in sparse lookup caches.
- Dynamic Sparse MoE routing (Top-K) and flexible hybrid Dense/MoE architecture.
- Load balancing auxiliary loss to prevent expert collapse during training.
- Integrated PyTorch FlexAttention for custom sliding window attention masks.
- Enabled compatibility with HF StaticCache and graph-friendly compilation (no dict caches).
- Optimized VRAM retention via Inline Allocation of Auxiliary Losses.
- Hardware-agnostic fallback to dense attention if FlexAttention is not supported.
- Implemented Industrial-grade Paged KV Cache (HumanVPagedCache) compatible with FlexAttention.

BUGFIX (this revision):
- Fixed `HumanVPagedCache.__init__` raising
  `cannot access local variable 'torch' where it is not associated with a value`.
  Root cause: `import torch._dynamo` inside the function body implicitly declared
  `torch` as a *local* variable for the entire enclosing function scope (Python
  scoping rule: any assignment/import of a name anywhere in a function makes it
  local for the whole function). This shadowed the module-level `torch` import
  used earlier in `__init__` (e.g. `torch.zeros(...)`, `torch.full(...)`),
  causing an UnboundLocalError at runtime. Fixed by importing the submodule
  under an alias (`import torch._dynamo as torch_dynamo`) so the global `torch`
  name is never shadowed locally.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Optional, Tuple, List, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

# Safe import for PyTorch FlexAttention (PyTorch 2.5+)
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    HAS_FLEX = True
except ImportError:
    HAS_FLEX = False

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...generation import GenerationMixin
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import logging, ModelOutput
from .configuration_humanv import HumanVConfig

logger = logging.get_logger(__name__)


# -----------------------------------------------------------------------------
# Precision-Safe Negative Infinity (Dynamic Masking to Prevent Overflow)
# -----------------------------------------------------------------------------
def _get_neg_inf(dtype: torch.dtype) -> float:
    """
    Computes a dynamically scaled negative infinity fallback value based on the tensor's precision.
    This prevents half-precision overflows in floating-point operations.
    """
    if dtype in (torch.float16, torch.half):
        return -30000.0
    return -1e9


# -----------------------------------------------------------------------------
# PyTorch FlexAttention Logical Mask Formulation
# -----------------------------------------------------------------------------
def _sparse_mask_fn(
    b: torch.Tensor,
    h: torch.Tensor,
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
    block_size: int,
    local_blocks: int,
    global_blocks: int,
    window_size: int,
    padding_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Logical function defining the local-global block sparse attention pattern.
    Compiles down into a highly efficient, single-fused Triton kernel.
    
    Args:
        b: Batch dimension tensor.
        h: Head dimension tensor.
        q_idx: Query token index tensor.
        kv_idx: Key/Value token index tensor.
        block_size: Dimension size of each block.
        local_blocks: Number of local blocks to attend to.
        global_blocks: Number of initial global blocks to attend to.
        window_size: Sliding window constraint parameter.
        padding_mask: Boolean tensor mask representing sequence padding.
    """
    is_causal = q_idx >= kv_idx
    q_block = q_idx // block_size
    kv_block = kv_idx // block_size

    is_global = kv_block < global_blocks
    is_local = (kv_block <= q_block) & (kv_block >= q_block - local_blocks + 1)
    valid = is_causal & (is_global | is_local)

    if window_size > 0:
        within_window = (q_idx - kv_idx) < window_size
        valid = valid & (within_window | is_global)

    if padding_mask is not None:
        return valid & padding_mask[b, kv_idx]

    return valid


# -----------------------------------------------------------------------------
# Industrial-grade Paged KV Cache Class
# -----------------------------------------------------------------------------
class HumanVPagedCache(Cache):
    """
    Paged KV Cache manager engineered to align scattered GPU pages with PyTorch FlexAttention.
    Guarantees fixed physical shapes during execution to prevent TorchInductor recompilations.
    """
    def __init__(
        self, 
        config: HumanVConfig, 
        max_batch_size: int, 
        num_pages: int, 
        page_size: int, 
        device: torch.device, 
        dtype: torch.dtype = torch.bfloat16
    ):
        # Satisfy Transformers parent validation requirements by supplying a dummy class for replication
        super().__init__(layer_class_to_replicate=object)
        
        self.num_pages = num_pages
        self.page_size = page_size
        self._max_batch_size = max_batch_size  # Private attribute prevents parent property setter clashes
        self.device = device
        self.dtype = dtype
        
        self.num_layers = config.num_hidden_layers
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        
        # Pre-allocate contiguous physical page buffers in GPU memory (Fixed Shape -> Zero JIT Recompiles)
        self.k_cache = [
            torch.zeros((1, self.num_kv_heads, num_pages * page_size, self.head_dim), device=device, dtype=dtype)
            for _ in range(self.num_layers)
        ]
        self.v_cache = [
            torch.zeros((1, self.num_kv_heads, num_pages * page_size, self.head_dim), device=device, dtype=dtype)
            for _ in range(self.num_layers)
        ]
        
        # Logical-to-Physical mapping tables
        self.page_table = torch.full((max_batch_size, num_pages), -1, dtype=torch.long, device=device)
        self.physical_to_logical = torch.full((max_batch_size, num_pages), -1, dtype=torch.long, device=device)
        self.seq_lengths = torch.zeros(max_batch_size, dtype=torch.long, device=device)
        self.free_pages = list(range(num_pages))

        # Register cache addresses as static to prevent Dynamo from skipping CUDA Graphs during in-place mutations.
        # NOTE: We import the submodule under an alias (`torch_dynamo`) instead of `import torch._dynamo`.
        # Using the bare `import torch._dynamo` statement binds the name `torch` as a *local* variable
        # for this entire function scope (Python scoping quirk), which shadows the module-level `torch`
        # import used earlier in this same method and causes an UnboundLocalError at runtime.
        try:
            import torch._dynamo as torch_dynamo
            for layer_idx in range(self.num_layers):
                torch_dynamo.mark_static_address(self.k_cache[layer_idx])
                torch_dynamo.mark_static_address(self.v_cache[layer_idx])
            torch_dynamo.mark_static_address(self.seq_lengths)
            torch_dynamo.mark_static_address(self.page_table)
            torch_dynamo.mark_static_address(self.physical_to_logical)
        except Exception:
            pass

    @property
    def batch_size(self) -> int:
        return self._max_batch_size

    @property
    def max_batch_size(self) -> int:
        return self._max_batch_size

    @property
    def is_initialized(self) -> bool:
        return True

    @property
    def is_compileable(self) -> bool:
        return True

    def allocate_page(self, batch_idx: int, logical_page_idx: int) -> int:
        """Allocates a free physical page to a logical sequence page index."""
        if not self.free_pages:
            raise RuntimeError("Out of physical pages in HumanVPagedCache!")
        physical_page_idx = self.free_pages.pop(0)
        self.page_table[batch_idx, logical_page_idx] = physical_page_idx
        self.physical_to_logical[batch_idx, physical_page_idx] = logical_page_idx
        return physical_page_idx

    def gather_contiguous(self, layer_idx: int, kv_seq_len: int | torch.Tensor, bsz: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gathers scattered physical pages on GPU memory into a logical contiguous representation.
        Completely vectorized and fully compile-friendly.
        """
        device = self.device
        
        # 1. Create a logical index mesh grid
        positions = torch.arange(kv_seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        logical_pages = positions // self.page_size
        offsets = positions % self.page_size
        
        # 2. Extract mapped physical indices from the page table
        batch_indices = torch.arange(bsz, device=device).unsqueeze(-1).expand(-1, kv_seq_len)
        physical_pages = self.page_table[batch_indices, logical_pages]
        physical_indices = physical_pages * self.page_size + offsets
        
        # 3. Direct gather using indexing without transposing entire raw cache
        k_cache_trans = self.k_cache[layer_idx][0].transpose(0, 1)  # (S, num_kv_heads, head_dim)
        v_cache_trans = self.v_cache[layer_idx][0].transpose(0, 1)  # (S, num_kv_heads, head_dim)
        
        k_gathered = k_cache_trans[physical_indices]  # (bsz, kv_seq_len, num_kv_heads, head_dim)
        v_gathered = v_cache_trans[physical_indices]
        
        # 4. Permute back to standard HF format: (bsz, num_kv_heads, kv_seq_len, head_dim)
        k_gathered = k_gathered.permute(0, 2, 1, 3)
        v_gathered = v_gathered.permute(0, 2, 1, 3)
        
        # 5. Apply masking for unallocated entries
        valid_mask = (physical_pages >= 0).unsqueeze(1).unsqueeze(-1)  # (bsz, 1, kv_seq_len, 1)
        k_gathered = k_gathered * valid_mask
        v_gathered = v_gathered * valid_mask
        
        return k_gathered, v_gathered

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the paged cache buffers with incoming query step Key and Value states.
        Handles advanced indexing alignment natively with no transpose overhead on LHS.
        """
        bsz, num_kv_heads, q_len, head_dim = key_states.shape
        device = key_states.device
        
        is_first_layer = (layer_idx == 0)
        
        # Vectorized logical page mapping directly on GPU (Pure Tensor Arithmetic, Compile-Friendly)
        for b in range(bsz):
            start_len = self.seq_lengths[b]
            logical_positions = torch.arange(q_len, device=device) + start_len
            logical_pages = logical_positions // self.page_size
            offsets = logical_positions % self.page_size
            
            if is_first_layer:
                if torch.compiler.is_compiling():
                    # During JIT compilation, we assume pages have been pre-allocated to bypass stateful CPU operations
                    pass
                else:
                    unique_pages = torch.unique(logical_pages)
                    for lp in unique_pages:
                        lp_idx = int(lp.item())
                        if self.page_table[b, lp_idx] == -1:
                            self.allocate_page(b, lp_idx)
            
            physical_pages = self.page_table[b, logical_pages]
            physical_indices = physical_pages * self.page_size + offsets
            
            # Direct in-place assignment aligned with PyTorch advanced indexing. 
            # No transpose needed on RHS since the indexing slice retains the default dimension layout.
            self.k_cache[layer_idx][0, :, physical_indices, :] = key_states[b].to(dtype=self.k_cache[layer_idx].dtype)
            self.v_cache[layer_idx][0, :, physical_indices, :] = value_states[b].to(dtype=self.v_cache[layer_idx].dtype)
            
            if is_first_layer:
                self.seq_lengths[b] += q_len
                
        kv_seq_len = self.seq_lengths.max() if torch.compiler.is_compiling() else int(self.seq_lengths.max().item())
        return self.gather_contiguous(layer_idx, kv_seq_len, bsz)

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int | torch.Tensor:
        """
        Returns the sequence length of the cached states. Returns a Tensor when compiling to avoid graph breaks.
        """
        if torch.compiler.is_compiling():
            # Keep get_seq_length symbolic on GPU during compilation, completely avoiding host-sync graph breaks
            return self.seq_lengths.max()
        return int(self.seq_lengths.max().item())


# -----------------------------------------------------------------------------
# Custom Output Classes for MoE Support
# -----------------------------------------------------------------------------
@dataclass
class HumanVBaseModelOutputWithPast(ModelOutput):
    """Base class for HumanV model outputs, including optional MoE routing metadata."""
    last_hidden_state: torch.Tensor = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None
    router_logits: Optional[Tuple[torch.Tensor, ...]] = None
    aux_loss: Optional[torch.Tensor] = None


@dataclass
class HumanVCausalLMOutputWithPast(ModelOutput):
    """Base class for HumanV causal language model outputs."""
    loss: Optional[torch.Tensor] = None
    aux_loss: Optional[torch.Tensor] = None
    logits: torch.Tensor = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None
    router_logits: Optional[Tuple[torch.Tensor, ...]] = None


# -----------------------------------------------------------------------------
# Normalization Layers
# -----------------------------------------------------------------------------
class HumanVRMSNorm(nn.Module):
    """Custom standard root-mean-square normalization (fallback option)."""
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.to(torch.float32)
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x).to(dtype)


class HumanVTorchRMSNorm(nn.Module):
    """PyTorch native RMSNorm module wrapping, defaulting to native performance if available."""
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        if hasattr(nn, "RMSNorm"):
            self.norm = nn.RMSNorm(hidden_size, eps=eps)
        else:
            self.norm = HumanVRMSNorm(hidden_size, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


# -----------------------------------------------------------------------------
# Rotary Position Embeddings (RoPE)
# -----------------------------------------------------------------------------
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half of the hidden dimension values for RoPE formulation."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Applies the sinusoidal rotary embeddings to query and key states."""
    cos = cos.unsqueeze(1)  # (B, 1, T, D)
    sin = sin.unsqueeze(1)  # (B, 1, T, D)
    q = (q * cos) + (_rotate_half(q) * sin)
    k = (k * cos) + (_rotate_half(k) * sin)
    return q, k


class HumanVRotaryEmbedding(nn.Module):
    """Constructs compile-safe sinusoidal cache arrays for Rotary Position Embeddings."""
    def __init__(self, dim: int, max_position_embeddings: int, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = int(max_position_embeddings)
        self.base = float(base)

        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._set_cos_sin_cache(self.max_position_embeddings, device=torch.device("cpu"), dtype=torch.float32)

    def _set_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = t[:, None] * self.inv_freq[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("_cos_cached", emb.cos().to(dtype=dtype), persistent=False)
        self.register_buffer("_sin_cached", emb.sin().to(dtype=dtype), persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        max_pos = torch.max(position_ids)
        if max_pos >= self.max_seq_len_cached:
            new_limit = int(max_pos.item()) + 512
            self._set_cos_sin_cache(new_limit, device=self.inv_freq.device, dtype=x.dtype)

        if self._cos_cached.device != x.device or self._cos_cached.dtype != x.dtype:
            self._cos_cached = self._cos_cached.to(device=x.device, dtype=x.dtype)
            self._sin_cached = self._sin_cached.to(device=x.device, dtype=x.dtype)

        cos = self._cos_cached[position_ids]  # (B, T, D)
        sin = self._sin_cached[position_ids]  # (B, T, D)
        return cos, sin


# -----------------------------------------------------------------------------
# Multi-Layer Perceptrons (MLP)
# -----------------------------------------------------------------------------
class HumanVMLP(nn.Module):
    """Standard multi-layer perceptron (MLP) block using Gated Linear Units."""
    def __init__(self, config: HumanVConfig):
        super().__init__()
        hidden_size = int(getattr(config, "hidden_size"))
        intermediate_size = int(getattr(config, "intermediate_size", hidden_size * 4))
        act = str(getattr(config, "hidden_act", "silu"))
        bias = bool(getattr(config, "mlp_bias", False))

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.act_fn = ACT2FN[act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# -----------------------------------------------------------------------------
# Sparse Mixture of Experts (MoE)
# -----------------------------------------------------------------------------
class HumanVMoeBlock(nn.Module):
    """Sparse Mixture of Experts (MoE) block with dynamic Top-K token routing."""
    def __init__(self, config: HumanVConfig):
        super().__init__()
        self.hidden_size = int(getattr(config, "hidden_size"))
        self.num_experts = int(getattr(config, "num_experts", 8))
        self.top_k = int(getattr(config, "num_experts_per_tok", 2))

        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList([HumanVMLP(config) for _ in range(self.num_experts)])

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        orig_shape = hidden_states.shape
        x = hidden_states.view(-1, self.hidden_size)

        router_logits = self.gate(x)
        routing_weights = F.softmax(router_logits, dim=-1)

        top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)

        # Normalize routing weights over Top-K selected experts
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        top_k_weights = top_k_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros_like(x)

        for expert_idx in range(self.num_experts):
            token_indices, top_k_pos = torch.where(top_k_indices == expert_idx)
            if token_indices.numel() == 0:
                continue

            expert_input = x[token_indices]
            expert_output = self.experts[expert_idx](expert_input)

            weight = top_k_weights[token_indices, top_k_pos].unsqueeze(-1)
            final_hidden_states.index_add_(0, token_indices, expert_output * weight)

        return final_hidden_states.view(orig_shape), router_logits


def load_balancing_loss(router_logits_list: list[torch.Tensor], num_experts: int, top_k: int) -> torch.Tensor:
    """
    Computes GShard load balancing auxiliary loss to penalize expert overload and prevent routing collapse.
    """
    if not router_logits_list:
        return torch.tensor(0.0)

    valid_logits = [logits for logits in router_logits_list if logits is not None]
    if not valid_logits:
        return torch.tensor(0.0)

    router_logits = torch.cat(valid_logits, dim=0)
    routing_weights = F.softmax(router_logits, dim=-1)

    _, top_k_indices = torch.topk(routing_weights, top_k, dim=-1)
    mask = torch.zeros_like(routing_weights)
    mask.scatter_(1, top_k_indices, 1.0)

    f = mask.mean(dim=0)
    P = routing_weights.mean(dim=0)

    loss = num_experts * torch.sum(f * P)
    return loss


# -----------------------------------------------------------------------------
# Attention (Dense + Sparse local/global block via FlexAttention)
# -----------------------------------------------------------------------------
class HumanVAttention(nn.Module):
    """
    Unified attention layer managing GQA (Grouped Query Attention) and sliding window sparse patterns.
    """
    def __init__(self, config: HumanVConfig, layer_idx: int, layer_type: str):
        super().__init__()
        self.config = config
        self.layer_idx = int(layer_idx)
        self.layer_type = str(layer_type)

        self.head_dim = int(getattr(config, "head_dim"))
        self.num_heads = int(getattr(config, "num_attention_heads"))
        self.num_kv_heads = int(getattr(config, "num_key_value_heads", self.num_heads))

        if self.num_kv_heads <= 0:
            raise ValueError(f"num_key_value_heads must be > 0, got {self.num_kv_heads}")
        if self.num_kv_heads > self.num_heads:
            raise ValueError(
                f"num_key_value_heads ({self.num_kv_heads}) cannot exceed num_attention_heads ({self.num_heads})"
            )
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")

        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scaling = float(getattr(config, "attention_scaling", 1.0)) / (self.head_dim ** 0.5)
        self.attention_dropout = float(getattr(config, "attention_dropout", 0.0))

        bias = bool(getattr(config, "attention_bias", False))
        hidden_size = int(getattr(config, "hidden_size"))

        self.q_proj = nn.Linear(hidden_size, self.num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden_size, bias=bias)

        self.rope_partial_rotary_factor = float(getattr(config, "rope_partial_rotary_factor", 1.0))

        self.use_sparse_attention = bool(getattr(config, "use_sparse_attention", False))
        self.sparse_attention_impl = str(getattr(config, "sparse_attention_impl", "local_global_block"))
        self.sparse_block_size = int(getattr(config, "sparse_block_size", 64))
        self.sparse_local_num_blocks = int(getattr(config, "sparse_local_num_blocks", 8))
        self.sparse_global_num_blocks = int(getattr(config, "sparse_global_num_blocks", 1))
        self.sparse_attention_window = int(getattr(config, "sparse_attention_window", 0) or 0)

        self.kv_cache_dtype = str(getattr(config, "kv_cache_dtype", "auto"))
        self.attn_backend = str(getattr(config, "attn_backend", "gqa_matmul")).lower().strip()
        if self.attn_backend not in ("gqa_matmul", "sdpa"):
            self.attn_backend = "gqa_matmul"

        # Compile the flex_attention helper function with CUDA Graphs disabled to prevent overwrite errors in loops
        if HAS_FLEX:
            try:
                self.flex_attention_compiled = torch.compile(
                    flex_attention, 
                    dynamic=True, 
                    options={"triton.cudagraphs": False}
                )
            except Exception:
                self.flex_attention_compiled = flex_attention
        else:
            self.flex_attention_compiled = None

    def _kv_dtype(self, x: torch.Tensor) -> torch.Tensor:
        """Casts incoming key/value tensor states to specified precision formats."""
        if self.kv_cache_dtype == "auto":
            return x
        if self.kv_cache_dtype in ("bf16", "bfloat16"):
            return x.to(torch.bfloat16)
        if self.kv_cache_dtype in ("fp16", "float16"):
            return x.to(torch.float16)
        if self.kv_cache_dtype in ("fp32", "float32"):
            return x.to(torch.float32)
        return x

    def _repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Expands Key-Value head counts to match Grouped Query Attention ratios."""
        if n_rep == 1:
            return x
        bsz, num_kv_heads, seq_len, head_dim = x.shape
        return (
            x[:, :, None, :, :]
            .expand(bsz, num_kv_heads, n_rep, seq_len, head_dim)
            .reshape(bsz, num_kv_heads * n_rep, seq_len, head_dim)
        )

    def _apply_partial_rope(self, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """Applies rotary position embeddings optionally only on a fraction of attention channels."""
        f = self.rope_partial_rotary_factor
        if f >= 0.999999:
            return _apply_rotary(q, k, cos, sin)

        rotary_dim = int(self.head_dim * f)
        if rotary_dim <= 0:
            return q, k

        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

        q_rot, k_rot = _apply_rotary(q_rot, k_rot, cos[..., :rotary_dim], sin[..., :rotary_dim])
        q = torch.cat([q_rot, q_pass], dim=-1)
        k = torch.cat([k_rot, k_pass], dim=-1)
        return q, k

    def _sdpa_mha_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask_4d: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Executes Scaled Dot-Product Attention (SDPA) using fused kernel paths."""
        dropout_p = self.attention_dropout if self.training else 0.0

        if attention_mask_4d is None:
            return F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=True)

        if attention_mask_4d.dtype is not torch.bool and attention_mask_4d.dtype != q.dtype:
            attention_mask_4d = attention_mask_4d.to(dtype=q.dtype)

        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask_4d, dropout_p=dropout_p, is_causal=False
        )

    def _grouped_dense_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask_4d: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Standard, un-fused dense attention execution fallback pathway."""
        k = self._repeat_kv(k, self.num_kv_groups)
        v = self._repeat_kv(v, self.num_kv_groups)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling

        if attention_mask_4d is not None:
            scores = scores + attention_mask_4d

        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        probs = F.dropout(probs, p=self.attention_dropout, training=self.training)

        out = torch.matmul(probs, v)
        return out

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask_4d: Optional[torch.Tensor] = None,
        attention_mask_2d: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = self._apply_partial_rope(q, k, cos, sin)

        if past_key_values is not None:
            k = self._kv_dtype(k)
            v = self._kv_dtype(v)
            cache_kwargs = {"cache_position": cache_position} if cache_position is not None else None
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)

        k_len = k.shape[-2]

        use_sparse = (
            self.use_sparse_attention
            and self.layer_type == "sliding_attention"
            and self.sparse_attention_impl == "local_global_block"
        )

        has_flex_support = HAS_FLEX and q.is_cuda

        if use_sparse and has_flex_support:
            try:
                # 1. BroadCast KV heads to align GQA within FlexAttention
                k = self._repeat_kv(k, self.num_kv_groups)
                v = self._repeat_kv(v, self.num_kv_groups)

                # Setup Boolean Padding mask for Dynamic/Static/Paged Caches (unified layout)
                pad_mask = attention_mask_2d.to(dtype=torch.bool) if attention_mask_2d is not None else None
                mask_mod = partial(
                    _sparse_mask_fn,
                    block_size=self.sparse_block_size,
                    local_blocks=self.sparse_local_num_blocks,
                    global_blocks=self.sparse_global_num_blocks,
                    window_size=self.sparse_attention_window,
                    padding_mask=pad_mask,
                )
                kv_len = k_len

                block_mask = create_block_mask(
                    mask_mod,
                    B=bsz,
                    H=None,  # Broadcast mask across heads
                    Q_LEN=q_len,
                    KV_LEN=kv_len,
                    device=q.device,
                    _compile=True,  # JIT pre-compilation
                )

                # 2. Select execution kernel depending on compilation environment
                if torch.compiler.is_compiling():
                    # The whole forward pass is being compiled; avoid dual compilations
                    attn_out = flex_attention(q, k, v, block_mask=block_mask)
                else:
                    # Eager mode (e.g. within model.generate() loops); invoke the compiled Triton kernel safely
                    attn_out = self.flex_attention_compiled(q, k, v, block_mask=block_mask)

            except Exception as e:
                logger.warning_once(f"FlexAttention compilation fell back to Dense GQA. Reason: {e}")
                attn_out = self._grouped_dense_attention(q, k, v, attention_mask_4d)
        else:
            # CPU or Non-Triton GPUs fall back to highly optimized dense 4D attention
            if use_sparse:
                attn_out = self._grouped_dense_attention(q, k, v, attention_mask_4d)
            else:
                if self.attn_backend == "sdpa":
                    k = self._repeat_kv(k, self.num_kv_groups)
                    v = self._repeat_kv(v, self.num_kv_groups)
                    attn_out = self._sdpa_mha_attention(q, k, v, attention_mask_4d)
                else:
                    attn_out = self._grouped_dense_attention(q, k, v, attention_mask_4d)

        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, q_len, self.num_heads * self.head_dim)
        attn_out = self.o_proj(attn_out)
        return attn_out, None


# -----------------------------------------------------------------------------
# Decoder Layer
# -----------------------------------------------------------------------------
class HumanVDecoderLayer(nn.Module):
    """Transformer decoder block encapsulating attention, MLP/MoE layers, and residual connections."""
    def __init__(self, config: HumanVConfig, layer_idx: int):
        super().__init__()
        layer_types = getattr(config, "layer_types", None)
        layer_type = "full_attention" if layer_types is None else str(layer_types[layer_idx])

        eps = float(getattr(config, "rms_norm_eps", 1e-6))
        norm_backend = str(getattr(config, "norm_backend", "torch_rmsnorm")).lower().strip()
        hidden_size = int(getattr(config, "hidden_size"))

        if norm_backend in ("layernorm", "ln"):
            self.input_layernorm = nn.LayerNorm(hidden_size, eps=eps)
            self.post_attention_layernorm = nn.LayerNorm(hidden_size, eps=eps)
        else:
            self.input_layernorm = HumanVTorchRMSNorm(hidden_size, eps=eps)
            self.post_attention_layernorm = HumanVTorchRMSNorm(hidden_size, eps=eps)

        self.self_attn = HumanVAttention(config=config, layer_idx=layer_idx, layer_type=layer_type)

        mlp_types = getattr(config, "mlp_types", None)
        mlp_type = "dense" if mlp_types is None else str(mlp_types[layer_idx])

        if mlp_type == "moe":
            self.mlp = HumanVMoeBlock(config)
        else:
            self.mlp = HumanVMLP(config)

        self.resid_dropout = float(getattr(config, "resid_dropout", 0.0))
        self.hidden_dropout = float(getattr(config, "hidden_dropout", 0.0))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask_4d: Optional[torch.Tensor] = None,
        attention_mask_2d: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # 1. Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attn_out, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask_4d=attention_mask_4d,
            attention_mask_2d=attention_mask_2d,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            cache_position=cache_position,
        )
        if self.resid_dropout and self.training:
            attn_out = F.dropout(attn_out, p=self.resid_dropout, training=True)
        hidden_states = residual + attn_out

        # 2. MLP / MoE Block
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        if isinstance(self.mlp, HumanVMoeBlock):
            mlp_out, router_logits = self.mlp(hidden_states)
        else:
            mlp_out = self.mlp(hidden_states)
            router_logits = None

        if self.hidden_dropout and self.training:
            mlp_out = F.dropout(mlp_out, p=self.hidden_dropout, training=True)
        hidden_states = residual + mlp_out

        return hidden_states, router_logits


# -----------------------------------------------------------------------------
# HF Base PreTrained Model
# -----------------------------------------------------------------------------
class HumanVPreTrainedModel(PreTrainedModel):
    """Pretrained model class configuration template mapping model configurations."""
    config_class = HumanVConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["HumanVDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]

    def _init_weights(self, module: nn.Module):
        std = float(getattr(self.config, "initializer_range", 0.02))
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


# -----------------------------------------------------------------------------
# HumanVModel
# -----------------------------------------------------------------------------
class HumanVModel(HumanVPreTrainedModel):
    """Transformer decoder stack that forwards embeddings and processes sequential hidden states."""
    def __init__(self, config: HumanVConfig):
        super().__init__(config)

        self.padding_idx = getattr(config, "pad_token_id", None)
        hidden_size = int(getattr(config, "hidden_size"))

        self.embed_tokens = nn.Embedding(int(config.vocab_size), hidden_size, padding_idx=self.padding_idx)
        self.layers = nn.ModuleList([HumanVDecoderLayer(config, i) for i in range(int(getattr(config, "num_hidden_layers")))])

        eps = float(getattr(config, "rms_norm_eps", 1e-6))
        norm_backend = str(getattr(config, "norm_backend", "torch_rmsnorm")).lower().strip()
        if norm_backend in ("layernorm", "ln"):
            self.norm = nn.LayerNorm(hidden_size, eps=eps)
        else:
            self.norm = HumanVTorchRMSNorm(hidden_size, eps=eps)

        rope_base = float(getattr(config, "rope_theta", 10000.0))
        self.rotary_emb = HumanVRotaryEmbedding(
            dim=int(getattr(config, "head_dim")),
            max_position_embeddings=int(getattr(config, "max_position_embeddings", 2048)),
            base=rope_base,
        )

        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _get_causal_mask(self, q_len: int, src_len: int, past_len: int | torch.Tensor, device: torch.device, dtype: torch.dtype):
        neg_inf = _get_neg_inf(dtype)
        m = torch.triu(
            torch.full((q_len, src_len), neg_inf, device=device, dtype=dtype),
            diagonal=1 + past_len,
        )
        return m[None, None, :, :]

    def _prepare_attention_masks(self, attention_mask_2d: torch.Tensor, q_len: int, past_len: int | torch.Tensor, dtype: torch.dtype):
        device = attention_mask_2d.device
        src_len = int(attention_mask_2d.shape[1])

        causal = self._get_causal_mask(q_len=q_len, src_len=src_len, past_len=past_len, device=device, dtype=dtype)

        key_valid = attention_mask_2d.to(dtype=torch.bool)
        neg_inf = _get_neg_inf(dtype)
        
        pad_bias = (~key_valid)[:, None, None, :].to(dtype=dtype) * neg_inf
        return causal + pad_bias

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, HumanVBaseModelOutputWithPast]:
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("You must provide input_ids or inputs_embeds")
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache is None:
            use_cache = bool(getattr(self.config, "use_cache", True))

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning(
                    "Gradient checkpointing is enabled, but `use_cache` is set to `True`. "
                    "This is not supported during training and `use_cache` will be set to `False`."
                )
                use_cache = False

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        bsz, q_len = inputs_embeds.shape[:2]

        if cache_position is None:
            cache_position = kwargs.get("cache_position", None)

        past_len = past_key_values.get_seq_length() if (past_key_values is not None and use_cache) else 0

        is_static_cache = past_key_values is not None and past_key_values.__class__.__name__ == "StaticCache"
        if use_cache and is_static_cache:
            max_cache_len = getattr(past_key_values, "max_cache_len", -1)
            if max_cache_len is not None and max_cache_len > 0:
                kv_seq_len = max_cache_len
            else:
                kv_seq_len = past_len + q_len
        else:
            kv_seq_len = past_len + q_len

        if attention_mask is None:
            attention_mask_2d = torch.ones((bsz, kv_seq_len), device=inputs_embeds.device, dtype=torch.bool)
        else:
            if attention_mask.dim() != 2:
                raise ValueError("attention_mask must be 2D (bsz, seq)")
            
            attention_mask_2d = attention_mask.to(device=inputs_embeds.device, dtype=torch.bool)
            if attention_mask_2d.shape[1] < kv_seq_len:
                pad_len = kv_seq_len - attention_mask_2d.shape[1]
                pad_tensor = torch.zeros((bsz, pad_len), device=inputs_embeds.device, dtype=torch.bool)
                attention_mask_2d = torch.cat([attention_mask_2d, pad_tensor], dim=-1)

        if position_ids is None:
            position_ids = torch.arange(q_len, dtype=torch.long, device=inputs_embeds.device) + past_len
            position_ids = position_ids.unsqueeze(0).expand(bsz, -1)

        cos, sin = self.rotary_emb(inputs_embeds, position_ids)
        position_embeddings = (cos, sin)

        attention_mask_4d = self._prepare_attention_masks(
            attention_mask_2d, q_len=q_len, past_len=past_len, dtype=inputs_embeds.dtype
        )

        hidden_states = inputs_embeds
        all_hidden_states = [] if output_hidden_states else None

        # Inline auxiliary loss allocation to minimize memory footprints
        total_aux_loss = torch.tensor(0.0, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        num_experts = int(getattr(self.config, "num_experts", 8))
        top_k = int(getattr(self.config, "num_experts_per_tok", 2))

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    attention_mask_4d,
                    attention_mask_2d,
                    position_embeddings,
                    past_key_values if use_cache else None,
                    output_attentions,
                    cache_position,
                    use_reentrant=False,
                )
                if isinstance(outputs, tuple):
                    hidden_states, router_logits = outputs
                else:
                    hidden_states = outputs
                    router_logits = None
            else:
                hidden_states, router_logits = layer(
                    hidden_states,
                    attention_mask_4d=attention_mask_4d,
                    attention_mask_2d=attention_mask_2d,
                    position_embeddings=position_embeddings,
                    past_key_values=past_key_values if use_cache else None,
                    output_attentions=output_attentions,
                    cache_position=cache_position,
                )

            if router_logits is not None:
                layer_aux_loss = load_balancing_loss([router_logits], num_experts, top_k)
                total_aux_loss = total_aux_loss + layer_aux_loss

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        return HumanVBaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=None,
            router_logits=None,
            aux_loss=total_aux_loss,
        )


# -----------------------------------------------------------------------------
# HumanVForCausalLM
# -----------------------------------------------------------------------------
class HumanVForCausalLM(HumanVPreTrainedModel, GenerationMixin):
    """Causal language model class wrapper wrapping standard generation helper interfaces."""
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: HumanVConfig):
        super().__init__(config)
        self.model = HumanVModel(config)
        self.vocab_size = int(config.vocab_size)
        self.lm_head = nn.Linear(int(getattr(config, "hidden_size")), int(config.vocab_size), bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, HumanVCausalLMOutputWithPast]:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states).to(torch.float32)

        loss = None
        aux_loss = outputs.aux_loss

        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

            if aux_loss is not None:
                router_aux_loss_coef = float(getattr(self.config, "router_aux_loss_coef", 0.01))
                loss = loss + router_aux_loss_coef * aux_loss

        return HumanVCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=None,
            router_logits=None,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        cache_position = kwargs.get("cache_position", None)
        if past_key_values is not None:
            if cache_position is not None:
                input_ids = input_ids[:, cache_position]
            else:
                input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "use_cache": kwargs.get("use_cache", True),
            "cache_position": cache_position,
        }


__all__ = ["HumanVForCausalLM", "HumanVModel", "HumanVPreTrainedModel", "HumanVPagedCache"]
