# SPDX-License-Identifier: Apache-2.0
"""LLSA backend smoke tests.

Skips when CUDA is unavailable. When the external `llsa` package is missing,
verifies that selecting LLSA_ATTN raises a clear ImportError. When installed,
runs a tiny forward shape check.
"""

import importlib.util
import os

import pytest
import torch

from fastvideo.attention.layer import LocalAttention
from fastvideo.attention.selector import get_attn_backend
from fastvideo.forward_context import set_forward_context
from fastvideo.platforms.interface import AttentionBackendEnum


def _has_llsa() -> bool:
    return importlib.util.find_spec("llsa") is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for LLSA backend tests")
def test_llsa_backend_uninstalled_raises(monkeypatch):
    """Selecting LLSA without the package installed raises an informative error."""
    if _has_llsa():
        pytest.skip("llsa is installed; run the forward-shape check instead")
    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", "LLSA_ATTN")
    with pytest.raises(ImportError) as excinfo:
        _ = get_attn_backend(
            head_size=64,
            dtype=torch.float16,
            supported_attention_backends=(AttentionBackendEnum.LLSA_ATTN, ),
        )
    msg = str(excinfo.value)
    assert "llsa" in msg and "pip install -e git+https://github.com/SingleZombie/LLSA.git" in msg


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for LLSA backend tests")
def test_llsa_backend_forward_shape(monkeypatch):
    """Tiny forward smoke when `llsa` is present."""
    if not _has_llsa():
        pytest.skip("llsa not installed — skipping forward-shape check")
    device = torch.device("cuda")
    dtype = torch.float16
    B, L, H, D = 1, 128, 2, 64
    q = torch.randn(B, L, H, D, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    # Force-select the backend and restrict the layer support set.
    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", "LLSA_ATTN")
    attn = LocalAttention(
        num_heads=H,
        head_size=D,
        causal=False,
        supported_attention_backends=(AttentionBackendEnum.LLSA_ATTN, ),
    ).to(device)
    with torch.inference_mode(), set_forward_context(current_timestep=0, attn_metadata=None):
        out = attn(q, k, v)
    assert out.shape == (B, L, H, D)

