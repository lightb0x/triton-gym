# Triton gym
## Outline
- `1_vector_addition` : [tutorial] 1D kernel
- `2_softmax` : [tutorial] 2D kernel
- `3_add` : 1D adder module
- `4_quantize_fp8` : 2D module (tensorwise & square-blockwise quantization)
- `5_matmul` : [tutorial] matrix multiplication

where
- tutorial : not wrapped into PyTorch layer
- others : wrapped into PyTorch layer, with forward and backward
