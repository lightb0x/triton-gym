# GEMM-FP8
Integrated quantize-GEMM module, "hybrid" variant

Implements MX-like matrix multiplication:
- row-block-wise times col-block-wise

where block can be
- 32 (MX specification)
- 64 (most TFLOPS on 4090)
- 128 (DeepSeek-V3 style)

In contrast, [`torch._scaled_mm` supports](https://gist.github.com/drisspg/783616821043ab4594b9784f556c6714)
- tensor-wise times tensor-wise
- row-wise times col-wise

## Blueprint
This module integrates quantize-GEMM as `Linear`, custom PyTorch module.

Assume matrix multiplication (same as [PyTorch `Linear`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.linear.html))
```
# forward
input   [*, in_f]
weight  [out_f, in_f]  --> weight.t  [in_f, out_f]
bias    [out_f]
output  [*, out_f]

# backward (param update)
grad_out  [*, out_f]   --> grad_out.t  [out_f, *]
input     [*, in_f]
grad_w    [out_f, in_f]

# backward (backprop, compute as `(weight.t @ grad_out.t).t`)
grad_out  [*, out_f]
weight    [out_f, in_f]
grad_i    [*, in_f]
```
where quantization granularity is:
- `input` and `grad_out` : `block_size` aligned with reduction dimension
- `weight` : `(block_size, block_size)` square-tile

Computation graph as follows: (from high-level to lower level)
- Forward
  - matmul : `A @ W.t`
    - quantize_vec_lhs `q(A)`
    - quantize_block-transpose `q(W).t`
  - `save_for_backward`
    - quantize_vec_rhs `q(A)`
    - (reuse `q(W).t`)
- Backward
  - backprop `grad_out @ W === (W.t @ grad_out.t).t`
    - quantize_vec_lhs-transpose `grad_out.t`
    - (reuse `q(W).t`)
  - param update `grad_out.t @ A`
    - quantize_vec_rhs-transpose `grad_out.t`
    - (reuse `q(A)`)


## Outline
- `linear.py` : PyTorch module
- `scaled_matmul.py` : matmul function
- `quantize_fp8.py` : quantizer function
  - Supports both quantization: given `(BLOCK_SIZE, BLOCK_SIZE)` input,
    - Square-tile based quantization : `(1, 1)` scale
    - Vector-wise quantization : `(BLOCK_SIZE, 1)` (rhs) scale and/or `(1, BLOCK_SIZE)` (lhs) scale
  - Unified interface
    - Both quantization returns `(BLOCK_SIZE, 1)` (rhs) scale and `(1, BLOCK_SIZE)` (lhs) scale
    - To reduce GMEM communication
    - To maximize performance of `matmul_scaled`: input to be fed directly to Tensor core as-is
  - See upper section "Blueprint" for detail
- `config.py` : quantization configuration
  - Block size of 64 turns out to be optimum in Ada (AD102)
- `test_and_bench.py` : test and benchmark "hybrid" FP8 `Linear`

## how to run
```bash
python3 test_and_bench.py  # `Linear` module
python3 matmul_scaled.py   # `matmul_scaled` function
python3 quantize_fp8.py    # `quantize_fp8` function
```

## Take-home message
- High performance of `matmul` can be bottlenecked easily:
  - [CUDA core] block-scaling accumulation occurs in CUDA core
  - [Memory access pattern] cache hit ratio must be high
  - input of `matmul` should be streamed as-is to tensor core
- FP8 performance of Ada comes with caveat
  - 300 TFLOPS achieved only with tensor-wise quantization
  - With proper scaling, performance drops: barely beats BF16 baseline by ~20%
