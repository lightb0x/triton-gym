# Quantizer module
Quantizes into *tensor-wise* FP8_e5m2

## Blueprint
Assume matrix multiplication (same as [PyTorch `Linear`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.linear.html))
```
# forward
input   [*, in_f]
weight  [out_f, in_f]
bias    [out_f]
output  [*, out_f]

# backward (param update)
grad_out.t  [out_f, *]
input       [*, in_f]
grad_w      [out_f, in_f]

# backward (backprop)
grad_out  [*, out_f]
weight    [out_f, in_f]
grad_i    [*, in_f]
```
This module should do its work preparing gemm input as FP8 *(both forward and backward)*

## Outline
- `quantize.py` : PyTorch module
- `quantize_fwd.py` : Triton kernel & wrapper
- `test_and_bench.py`

## How to run
```bash
python3 test_and_bench.py
```

## Take-home message
- Triton performance boost from GMEM communication reduction
- Something off with `tl.cast`
