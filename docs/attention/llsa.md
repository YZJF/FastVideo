# LLSA (Log-Linear Sparse Attention)

LLSA is an optional attention backend for experimentation. It wraps the external
`llsa` package by SingleZombie and adds it to FastVideo’s attention selector.

## Install

```bash
pip install -e git+https://github.com/SingleZombie/LLSA.git
```

Then select it at runtime:

```bash
export FASTVIDEO_ATTENTION_BACKEND=LLSA_ATTN
```

The backend integrates with DiT-based models that already support FlashAttention
and SDPA. It auto-selects between L1/L2 varlen kernels based on sequence length
and uses a default `block_size=16`.

## License

LLSA is released under the S-Lab License 1.0 (non-commercial). The kernels are
not vendored into this repository. You must install the external package
separately and respect its license.

