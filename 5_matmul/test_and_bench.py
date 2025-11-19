import torch
import triton
from matmul_func import matmul
from utils import generate_range

DEVICE = triton.runtime.driver.active.get_active_torch_device()


_CONFIGS = [
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=generate_range(128, 8, 16384 + 1),
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "Torch"],
        styles=[("green", "-"), ("blue", "-")],
        ylabel="TFLOPS",
        plot_name="-".join(
            ["benchmark", f"bias{is_bias}", "fp8" if is_fp8 else "bf16"]
        ),
        args={"is_bias": is_bias, "is_fp8": is_fp8},
    )
    for (is_bias, is_fp8) in [(True, False), (False, False)]
]
_CONFIGS += [
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=generate_range(128, 8, 16384 + 1),
        line_arg="provider",
        line_vals=["triton"],
        line_names=["Triton"],
        styles=[("green", "-")],
        ylabel="TFLOPS",
        plot_name="-".join(
            ["benchmark", f"bias{is_bias}", "fp8" if is_fp8 else "bf16"]
        ),
        args={"is_bias": is_bias, "is_fp8": is_fp8},
    )
    for (is_bias, is_fp8) in [(True, True), (False, True)]
]


@triton.testing.perf_report(_CONFIGS)
def benchmark(N, provider, is_bias, is_fp8):
    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)
    if provider == "torch":
        func = torch.matmul
    else:
        assert provider == "triton"
        func = matmul

    dtype = torch.float8_e5m2 if is_fp8 else torch.bfloat16
    lhs = torch.randn((N, N), device=DEVICE).to(dtype)
    rhs = torch.randn((N, N), device=DEVICE).to(dtype)

    if is_bias:
        bias = torch.randn((N), device=DEVICE)
    else:
        bias = None

    def bench_run():
        if provider == "torch":
            result = func(lhs, rhs)
            if bias is not None:
                result = result + bias.view(-1, N).broadcast_to((N, N))
        else:
            func(lhs, rhs, bias)

    ms = triton.testing.do_bench(bench_run)
    tflops = lambda ms: 2 * N * N * N * 1e-12 / (ms * 1e-3)
    return tflops(ms)


if __name__ == "__main__":
    seed = 42

    for dim in generate_range(128, 8, 16384 + 1):
        for is_bias in [True, False]:
            for is_fp8 in [True, False]:
                dtype = torch.float8_e5m2 if is_fp8 else torch.bfloat16
                print(f"dim: {dim}x{dim}", end="\t")
                print(f"bias {is_bias}", end="\t")
                print(f"dtype {dtype}", end="\t")

                gen = torch.Generator(device=DEVICE).manual_seed(seed)
                lhs = torch.randn((dim, dim), device=DEVICE, generator=gen).to(dtype)
                rhs = torch.randn((dim, dim), device=DEVICE, generator=gen).to(dtype)

                if is_bias:
                    bias = torch.randn((dim), device=DEVICE, generator=gen) * 32
                else:
                    bias = None

                # `torch.matmul`
                if is_fp8:
                    result_gm = torch.matmul(lhs.to(torch.float), rhs.to(torch.float))
                else:
                    result_gm = torch.matmul(lhs, rhs)

                if bias is not None:
                    result_gm += bias.view(-1, dim).broadcast_to(dim, dim)

                # # `torch.nn.functional.linear`
                # - does not support FP8 input
                # - requires datatypes of all inputs to be the same
                # from torch.nn import functional as F
                # rhs_gm = rhs.to(torch.float).t().to(dtype)
                # result_gm = F.linear(lhs, rhs_gm, bias)

                result_ours = matmul(lhs, rhs, bias=bias)

                loss_gm = result_gm.sum()
                loss_ours = result_ours.sum()

                error = torch.nn.functional.mse_loss(result_ours, result_gm)
                print(error.item())
                if error.item() > 1:
                    print(result_gm)
                    print(result_ours)
                    offset = (result_ours - result_gm).abs()
                    print(offset.max())
                    print(offset.argmax())
                    exit()

    print("running benchmark ...")
    benchmark.run(print_data=True, save_path="./")
