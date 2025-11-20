import torch
import triton
from matmul_scaled import matmul_scaled
from quantize_fp8 import quantize_fp8

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def gen_linear_func(max_exp, use_bias):
    class LinearFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, act, wgt, bias):
            assert len(wgt.shape) == 2
            assert act.shape[-1] == wgt.shape[1]
            # act : (M, K)
            # wgt : (N, K)

            orig_act_shape = act.shape
            orig_wgt_shape = wgt.shape

            act = act.view(-1, act.shape[-1])

            act_fp8_bwd, act_scale_bwd, act_fp8_fwd, act_scale_fwd = (
                quantize_fp8(act, max_exp, is_sqtile=False)
            )
            # wgt.t : (K, N)
            wgt_fp8, wgt_scale_r, _, wgt_scale_l = quantize_fp8(wgt, max_exp, is_sqtile=True, out_trans=True)

            ctx.save_for_backward(act_fp8_bwd, act_scale_bwd, wgt_fp8, wgt_scale_l)

            # (M, K) @ (K, N) --> (M, N)
            result = matmul_scaled(
                lhs=act_fp8_fwd,
                lhs_scale=act_scale_fwd,
                rhs=wgt_fp8,
                rhs_scale=wgt_scale_r,
                bias=bias,
            )
            return result.view([*orig_act_shape[:-1], orig_wgt_shape[0]])

        @staticmethod
        def backward(ctx, grad_output):
            act_fp8_bwd, act_scale_bwd, wgt_fp8, wgt_scale = ctx.saved_tensors

            # grad_out.t : (N, M)
            grad_out_fp8_r, grad_out_scale_r, grad_out_fp8_l, grad_out_scale_l = (
                quantize_fp8(grad_output.contiguous(), max_exp, is_sqtile=False, out_trans=True)
            )

            # grad_out.t @ act : (N, M) @ (M, K) --> (N, K)
            grad_wgt = matmul_scaled(
                lhs=grad_out_fp8_l,
                lhs_scale=grad_out_scale_l,
                rhs=act_fp8_bwd,
                rhs_scale=act_scale_bwd,
            )

            # wgt.t @ grad_out.t : (K, N) @ (N, M) --> (K, M)
            # then transposed output : (M, K)
            grad_act = matmul_scaled(
                lhs=wgt_fp8,
                lhs_scale=wgt_scale,
                rhs=grad_out_fp8_r,
                rhs_scale=grad_out_scale_r,
                out_transpose=True,
            )

            if use_bias:
                return grad_act, grad_wgt, grad_output.view(-1, grad_output.shape[-1]).sum(dim=0)
            else:
                return grad_act, grad_wgt, None

    return LinearFunction


class Linear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        device=DEVICE,
        max_exp: int = 1,
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features), device=device)
        )
        self.bias = (
            torch.nn.Parameter(torch.zeros((out_features), device=device))
            if bias
            else None
        )

        assert max_exp < 16, "fp8_e5m2 maximum value == 57,344 (1.75 * 2**15) < 2**16"
        assert max_exp > 0, (
            "max_exp should be positive to leverage dynamic range of fp8_e5m2"
        )
        self.max_exp = max_exp

        self.linear_function = gen_linear_func(self.max_exp, bias)

    def forward(self, x):
        return self.linear_function.apply(x, self.weight, self.bias)
