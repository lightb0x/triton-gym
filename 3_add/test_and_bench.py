import torch
import torch.nn.functional as F
import triton
from add import Add


class AddReluGM(torch.nn.Module):
    def __init__(self, init_param, apply_relu=False):
        super().__init__()
        self.bias = torch.nn.Parameter(init_param, requires_grad=True)
        self.apply_relu = apply_relu

    def forward(self, x):
        if self.apply_relu:
            return F.relu(self.bias + x)
        else:
            return self.bias + x


def generate_range(start, num_steps, until):
    """
    generates floating-point-like range
    """
    current = start
    step = start
    arr = []
    while current < until:
        arr.append(current)
        current += step
        if current == step * num_steps:
            step *= 2

    return arr


_CONFIGS = [
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[i for i in generate_range(128, 4, 16384 + 1)],
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "Torch"],
        styles=[("green", "-"), ("blue", "-")],
        ylabel="Gelem/s",
        plot_name="-".join(["benchmark", f"relu{apply_relu}", f"bwd{include_bwd}"]),
        args={"apply_relu": apply_relu, "include_bwd": include_bwd},
    )
    for apply_relu in [True, False]
    for include_bwd in [True, False]
]


@triton.testing.perf_report(_CONFIGS)
def benchmark(N, provider, apply_relu, include_bwd):
    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)

    init_param = torch.randn(size=(N, N), device=DEVICE, dtype=torch.float32)
    if provider == "torch":
        module = AddReluGM(init_param, apply_relu)
    else:
        assert provider == "triton"
        module = Add(init_param, apply_relu)

    input_x = torch.randn_like(init_param)
    if include_bwd:
        ms = triton.testing.do_bench(lambda: module(input_x).sum().backward())
    else:
        ms = triton.testing.do_bench(lambda: module(input_x))
    gbps = lambda ms: N * N * 1e-9 / (ms * 1e-3)
    return gbps(ms)


if __name__ == "__main__":
    seed = 42
    DEVICE = triton.runtime.driver.active.get_active_torch_device()

    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    for i in range(4, 17):
        for apply_relu in [True, False]:
            size = 2**i
            t1 = torch.randn(size, generator=gen, device=DEVICE)
            t2 = torch.randn(size, generator=gen, device=DEVICE)

            t1_ = t1.clone().detach()
            t2_ = t2.clone().detach()

            baseline_module = AddReluGM(t1, apply_relu)
            our_module = Add(t1_, apply_relu)

            baseline_module.zero_grad()
            our_module.zero_grad()

            baseline_loss = baseline_module(t2).sum()
            our_loss = our_module(t2_).sum()

            baseline_loss.backward()
            our_loss.backward()

            is_intact = torch.allclose(baseline_module.bias.grad, our_module.bias.grad)

            print(f"settings: dim={size}, apply_relu={apply_relu}", end="\t")
            if is_intact:
                print("Passed!")
            else:
                print(f"loss : {baseline_loss}, {our_loss}")
                print(f"grad : {baseline_module.bias.grad}, {our_module.bias.grad}")

    print("running benchmark ...")
    benchmark.run(print_data=True, save_path="./")
