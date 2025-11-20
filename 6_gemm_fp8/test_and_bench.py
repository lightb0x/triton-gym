import torch
import triton
from linear import Linear
from quantize_fp8 import quantize_fp8
from utils import generate_range, reconstruct

DEVICE = triton.runtime.driver.active.get_active_torch_device()

class Q_fp8(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, max_exp, is_sqtile, is_left):
        x_fp8_r, x_scale_r, x_fp8_l, x_scale_l = quantize_fp8(x, max_exp=max_exp, is_sqtile=is_sqtile)
        if is_left:
            return reconstruct(x_fp8_l, x_scale_l)
        else:
            return reconstruct(x_fp8_r, x_scale_r)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None

class LinearGM(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        device=DEVICE,
        max_exp: int = 1,
        block_size: int = 128,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features), device=DEVICE)
        )
        self.bias = (
            torch.nn.Parameter(torch.zeros((out_features), device=DEVICE))
            if bias
            else None
        )
        self.max_exp = max_exp

    def forward(self, x):
        x_fp8 = Q_fp8.apply(x, self.max_exp, False, True)
        w_fp8 = Q_fp8.apply(self.weight, self.max_exp, True, False)
        dot = torch.nn.functional.linear(x_fp8, w_fp8)
        if self.bias is None:
            result = dot
        else:
            result = dot + self.bias
        return result


_CONFIGS = [
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=generate_range(128, 8, 16384+1),
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "Torch"],
        styles=[("green", "-"), ("blue", "-")],
        ylabel="TFLOPS",
        plot_name="-".join(["linear", f"bias{use_bias}", "fwd" + "bwd" if include_backward else ""]),
        args={"include_backward": include_backward, "use_bias": use_bias},
    ) for (include_backward, use_bias) in [(True, True), (True, False), (False, True), (False, False)]
]


@triton.testing.perf_report(_CONFIGS)
def benchmark(N, provider, include_backward, use_bias):
    init_param = torch.randn((N, N), device=DEVICE, dtype=torch.float32)
    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)
    if provider == "torch":
        module = LinearGM(N, N, bias=use_bias)
    else:
        assert provider == "triton"
        module = Linear(N, N, bias=use_bias)
    module.weight.data = init_param

    input_x = torch.randn_like(init_param)

    if include_backward:
        def bench_run():
            module(input_x).sum().backward()

        ms = triton.testing.do_bench(bench_run)
    else:
        def bench_run():
            module(input_x)

        ms = triton.testing.do_bench(bench_run)

    num_matmul = 3 if include_backward else 1
    tflops = lambda ms: num_matmul * 2 * N * N * N * 1e-12 / (ms * 1e-3)
    return tflops(ms)



if __name__ == "__main__":
    seed = 42
    max_exp = 12

    for M in generate_range(128, 4, 4096+1):
        N = M
        for K in generate_range(128, 4, 4096+1):
            for use_bias in [True, False]:
                print(f"(M, N, K): ({M}, {N}, {K}), bias={use_bias}", end="\t")
                mod_gm = LinearGM(K, N, bias=use_bias, max_exp=max_exp)
                mod_ours = Linear(K, N, bias=use_bias, max_exp=max_exp)

                gen = torch.Generator(device=DEVICE).manual_seed(seed)
                param = torch.randn((N, K), device=DEVICE, generator=gen)

                mod_gm.weight.data = param
                mod_ours.weight.data = param

                t1 = torch.randn((M, K), device=DEVICE, generator=gen)
                t1 = torch.nn.Parameter(t1, requires_grad=True)
                t1_ = t1.clone().detach()
                t1_ = torch.nn.Parameter(t1_, requires_grad=True)

                result_gm = mod_gm(t1)
                result_ours = mod_ours(t1_)
                loss_gm = result_gm.sum()
                loss_ours = result_ours.sum()
                loss_gm.backward()
                loss_ours.backward()

                errors = []

                fwd_error = torch.nn.functional.mse_loss(result_ours, result_gm)
                errors.append(fwd_error.item())

                wgt_grad_err = torch.nn.functional.mse_loss(mod_gm.weight.grad, mod_ours.weight.grad)
                errors.append(wgt_grad_err.item())

                act_grad_err = torch.nn.functional.mse_loss(t1.grad, t1_.grad)
                errors.append(act_grad_err.item())

                if use_bias:
                    bias_grad_error = torch.nn.functional.mse_loss(mod_gm.bias.grad, mod_ours.bias.grad)
                    errors.append(bias_grad_error.item())

                print(errors)

    print("running benchmark ...")
    benchmark.run(print_data=True, save_path="./")
