import torch
from quantize_fwd import quantize_fp8_square_tile_func, quantize_fp8_tensor_func
from utils import is_power_of_two


class QuantizeFP8TensorFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, max_exp):
        x_fp8, scale = quantize_fp8_tensor_func(x, max_exp)
        return x_fp8, scale

    @staticmethod
    def backward(ctx, grad_output):
        # assume GEMM handles divide-by-scale in advance
        return grad_output, None


def gen_sqtile_func(block_size):
    class QuantizeFP8SquareTileFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, max_exp):
            x_fp8, scale = quantize_fp8_square_tile_func(x, block_size, max_exp)
            return x_fp8, scale

        @staticmethod
        def backward(ctx, grad_output):
            # assume GEMM handles divide-by-scale in advance
            return grad_output, None, None

    return QuantizeFP8SquareTileFunction


class Quantize(torch.nn.Module):
    def __init__(self, max_exp: int = 1, block_size: int = 128, scale_unit="Tile"):
        """
        Args:
            max_to (float)
                Map `max(|input|)` to FP exponent range power-of-two `[max_to // 2, max_to)`
                For example,
                ```python3
                input = [0.5, 1, 32, -64]
                absmax = input.abs().max()  # 64
                absmax_exp = ceil(log2(absmax))  # 6

                max_to = 1  # max_exp = 0

                scale = 2**(absmax_exp - max_exp)  # 64
                input_fp8 = input * 2**(max_exp - absmax_exp)  # [2**-7, 2**-6, 0.5, -1]
                ```
        """
        super().__init__()
        assert max_exp < 16, "fp8_e5m2 maximum value < 2**16 = 65,536"
        assert max_exp > 0, (
            "max_exp should be positive to leverage dynamic range of fp8_e5m2"
        )

        self.max_exp = max_exp
        assert scale_unit in ["Tile", "Tensor"], (
            f"scale unit must be 'Tile' or 'Tensor: got {scale_unit}"
        )
        assert is_power_of_two(block_size)
        self.block_size = block_size
        self.scale_unit = scale_unit
        if self.scale_unit == "Tile":
            self.quantize_func = gen_sqtile_func(block_size)
        else:
            assert self.scale_unit == "Tensor"
            self.quantize_func = QuantizeFP8TensorFunction

    def forward(self, x):
        return self.quantize_func.apply(x, self.max_exp)
