# SPDX-License-Identifier: Apache-2.0
"""
LLSA (Log-Linear Sparse Attention) backend integration for FastVideo.

External kernels live in the 'llsa' package:
  - llsa.kernel.torch_op.flash_sparse_attention_res_1_varlen.llsa_l1_varlen
  - llsa.kernel.torch_op.flash_sparse_attention_res_2_varlen.llsa_l2_varlen

Installation (non-commercial S-Lab License 1.0):
  pip install -e git+https://github.com/SingleZombie/LLSA.git

Notes
-----
- This backend is optional. FastVideo will import and run without 'llsa'
  installed unless FASTVIDEO_ATTENTION_BACKEND=LLSA_ATTN is explicitly
  selected.
- Kernels are non-causal (bidirectional). If a causal path is requested,
  this backend raises NotImplementedError.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable

import torch

from fastvideo.attention.backends.abstract import (AttentionBackend, AttentionImpl, AttentionMetadata,
                                                   AttentionMetadataBuilder)
from fastvideo.logger import init_logger

logger = init_logger(__name__)

# Heuristic switch between L1 and L2 kernels by sequence length.
_LLSA_L2_THRESHOLD: int = 16384
_DEFAULT_BLOCK_SIZE: int = 16
_DEFAULT_TOPK: int = 64


def _import_llsa_kernels() -> tuple[Callable[..., torch.Tensor], Callable[..., torch.Tensor]]:
    try:
        from llsa.kernel.torch_op.flash_sparse_attention_res_1_varlen import llsa_l1_varlen
        from llsa.kernel.torch_op.flash_sparse_attention_res_2_varlen import llsa_l2_varlen
        return llsa_l1_varlen, llsa_l2_varlen
    except ImportError as e:
        raise ImportError(
            "LLSA backend requires the external 'llsa' package. "
            "Install with: pip install -e git+https://github.com/SingleZombie/LLSA.git "
            "(S-Lab License 1.0, non-commercial)."
        ) from e


class LLSAAttentionBackend(AttentionBackend):

    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        return "LLSA_ATTN"

    @staticmethod
    def get_impl_cls() -> type["LLSAAttentionImpl"]:
        return LLSAAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        raise NotImplementedError

    @staticmethod
    def get_builder_cls() -> type["AttentionMetadataBuilder"]:
        raise NotImplementedError


@dataclass
class _LLSAMetadata(AttentionMetadata):
    current_timestep: int = 0


class _LLSACallBuilder:
    """Adapts to minor API variations across llsa varlen kernels."""

    def __init__(self, kernel: Callable[..., torch.Tensor]) -> None:
        self.kernel = kernel
        self.params = list(inspect.signature(kernel).parameters.keys())

    def call(
        self,
        q_flat: torch.Tensor,
        k_flat: torch.Tensor,
        v_flat: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        *,
        is_causal: bool,
        scale: float | None,
        block_size: int,
        topk: int,
        dropout_p: float = 0.0,
    ) -> torch.Tensor:
        # Positional QKV first (shared across signatures)
        args: list[Any] = [q_flat, k_flat, v_flat]
        kwargs: dict[str, Any] = {}

        # Varlen seqlens/maxlen (q-only or q/k)
        if "cu_seqlens" in self.params:
            kwargs["cu_seqlens"] = cu_seqlens
        else:
            if "cu_seqlens_q" in self.params:
                kwargs["cu_seqlens_q"] = cu_seqlens
            if "cu_seqlens_k" in self.params:
                kwargs["cu_seqlens_k"] = cu_seqlens
        if "max_seqlen" in self.params:
            kwargs["max_seqlen"] = max_seqlen
        else:
            if "max_seqlen_q" in self.params:
                kwargs["max_seqlen_q"] = max_seqlen
            if "max_seqlen_k" in self.params:
                kwargs["max_seqlen_k"] = max_seqlen

        # Causality (documented unsupported; pass through when present)
        if "is_causal" in self.params:
            kwargs["is_causal"] = is_causal
        elif "causal" in self.params:
            kwargs["causal"] = is_causal

        # Scale
        if scale is not None:
            if "softmax_scale" in self.params:
                kwargs["softmax_scale"] = scale
            elif "scale" in self.params:
                kwargs["scale"] = scale

        # Block size / topk defaults
        if "block_size" in self.params:
            kwargs["block_size"] = block_size
        if "topk" in self.params:
            kwargs["topk"] = topk
        elif "top_k" in self.params:
            kwargs["top_k"] = topk
        elif "k_topk" in self.params:
            kwargs["k_topk"] = topk
        elif "v_topk" in self.params:
            kwargs["v_topk"] = topk

        # Dropout (train-only; keep 0.0)
        if "dropout_p" in self.params:
            kwargs["dropout_p"] = dropout_p
        elif "p_dropout" in self.params:
            kwargs["p_dropout"] = dropout_p

        try:
            return self.kernel(*args, **kwargs)
        except TypeError:
            # Fallback: minimal varlen signature
            return self.kernel(q_flat, k_flat, v_flat, cu_seqlens, max_seqlen)


class LLSAAttentionImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout = float(extra_impl_args.get("dropout_p", 0.0) or 0.0)
        if self.causal:
            # LLSA kernels are bidirectional; do not silently run causal.
            raise NotImplementedError("LLSA backend does not support causal attention.")
        # Lazy import; errors surface only when the backend is actively used.
        self._l1, self._l2 = _import_llsa_kernels()

    def preprocess_qkv(
        self,
        qkv: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        # [B, L, H, D] -> [B, H, L, D]
        return qkv.permute(0, 2, 1, 3).contiguous()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        """
        Inputs are [B, H, L, D] and contiguous (preprocess_qkv applied).
        Returns [B, H, L, D].
        """
        if query.dim() != 4:
            raise ValueError(f"Expected query with 4 dims [B,H,L,D], got {query.shape}")
        B, H, L, D = query.shape

        # Choose kernel by sequence length
        kernel = self._l1 if L <= _LLSA_L2_THRESHOLD else self._l2
        call = _LLSACallBuilder(kernel)

        # Flatten across batch for varlen API: [(B*L), H, D]
        q_bld = query.permute(0, 2, 1, 3).contiguous()  # [B, L, H, D]
        k_bld = key.permute(0, 2, 1, 3).contiguous()
        v_bld = value.permute(0, 2, 1, 3).contiguous()
        q_flat = q_bld.reshape(B * L, H, D)
        k_flat = k_bld.reshape(B * L, H, D)
        v_flat = v_bld.reshape(B * L, H, D)

        # Varlen offsets: uniform lengths across batch
        cu_seqlens = torch.arange(0, (B + 1) * L, L, device=query.device, dtype=torch.int32)
        max_seqlen = int(L)

        out_flat = call.call(q_flat,
                             k_flat,
                             v_flat,
                             cu_seqlens,
                             max_seqlen,
                             is_causal=self.causal,
                             scale=self.softmax_scale,
                             block_size=_DEFAULT_BLOCK_SIZE,
                             topk=min(_DEFAULT_TOPK, L),
                             dropout_p=self.dropout)
        # Restore to [B, H, L, D]
        out_bld = out_flat.reshape(B, L, H, D)
        return out_bld.permute(0, 2, 1, 3).contiguous()

    def postprocess_output(
        self,
        output: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        # [B, H, L, D] -> [B, L, H, D]
        return output.permute(0, 2, 1, 3).contiguous()

