import torch
import triton
from quantize import Quantize


class QuantizeGM(torch.nn.Module):
    def __init__(self, max_exp: int = 1, block_size: int = 128, scale_unit="Tile"):
        super().__init__()

        self.max_exp = max_exp
        self.block_size = block_size
        self.scale_unit = scale_unit

    def forward(self, x):
        if self.scale_unit == "Tensor":
            x_maxexp = torch.log2(x.abs().max()).floor()
            x_fp8 = (x * 2 ** (self.max_exp - x_maxexp)).to(torch.float8_e5m2)
            x_scale = 2 ** (x_maxexp - self.max_exp)
        else:
            orig_shape = x.shape
            x = x.view(-1, x.shape[-1])
            M, N = x.shape
            assert M % self.block_size == 0, (
                f"{M} must be multiple of block size {self.block_size}"
            )
            assert N % self.block_size == 0, (
                f"{N} must be multiple of block size {self.block_size}"
            )
            x_block = x.view(
                M // self.block_size,
                self.block_size,
                N // self.block_size,
                self.block_size,
            )
            x_block_tr = x_block.transpose(1, 2)
            x_block_tr = x_block_tr.reshape(
                M // self.block_size, N // self.block_size, -1
            )
            x_maxexp = torch.log2(x_block_tr.abs().max(dim=-1).values).floor()
            x_fp8 = (
                (
                    x_block_tr
                    * (2 ** (self.max_exp - x_maxexp)).view([*x_maxexp.shape, 1])
                )
                .view(
                    M // self.block_size,
                    N // self.block_size,
                    self.block_size,
                    self.block_size,
                )
                .transpose(1, 2)
                .reshape(orig_shape)
                .to(torch.float8_e5m2)
            )
            x_scale = 2 ** (x_maxexp - self.max_exp)
        return x_fp8, x_scale


def reconstruct(x_fp8, x_scale, block_size=None):
    if block_size is None:
        return x_fp8.to(torch.float) * x_scale
    else:
        x_scale_br = x_scale.reshape((*x_scale.shape, 1, 1)).broadcast_to(
            (
                *x_scale.shape,
                block_size,
                block_size,
            )
        )
        x_scale_brt = x_scale_br.transpose(1, 2).reshape(x_fp8.shape)
        return x_fp8.to(torch.float) * x_scale_brt


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
        plot_name="-".join(["benchmark", scale_unit, f"block{block_size}"]),
        args={"scale_unit": scale_unit, "block_size": block_size},
    )
    for (scale_unit, block_size) in [("Tile", 32), ("Tile", 128), ("Tensor", 128)]
]


@triton.testing.perf_report(_CONFIGS)
def benchmark(N, provider, scale_unit, block_size):
    init_param = torch.randn((N, N), device=DEVICE, dtype=torch.float32)
    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)
    if provider == "torch":
        module = QuantizeGM(scale_unit=scale_unit, block_size=min(N, block_size))
    else:
        assert provider == "triton"
        module = Quantize(scale_unit=scale_unit, block_size=min(N, block_size))

    input_x = torch.randn_like(init_param)
    ms = triton.testing.do_bench(lambda: module(input_x))
    gbps = lambda ms: N * N * 1e-9 / (ms * 1e-3)
    return gbps(ms)


if __name__ == "__main__":
    DEVICE = triton.runtime.driver.active.get_active_torch_device()
    seed = 42
    block_size = 128

    # test
    for shape in [(2**i, 2**i) for i in range(4, 15)]:
        for scale_unit in ["Tile", "Tensor"]:
            gen = torch.Generator(device=DEVICE).manual_seed(seed)
            t1 = torch.randn(shape, generator=gen, device=DEVICE)

            our_module = Quantize(
                scale_unit=scale_unit, block_size=min(shape[0], shape[1], block_size)
            )
            gm_module = QuantizeGM(
                scale_unit=scale_unit, block_size=min(shape[0], shape[1], block_size)
            )

            our_fp8, our_scale = our_module(t1)
            gm_fp8, gm_scale = gm_module(t1)

            our_recon = reconstruct(
                our_fp8,
                our_scale,
                block_size=min(shape[0], shape[1], block_size)
                if scale_unit == "Tile"
                else None,
            )
            gm_recon = reconstruct(
                gm_fp8,
                gm_scale,
                block_size=min(shape[0], shape[1], block_size)
                if scale_unit == "Tile"
                else None,
            )

            print(f"shape: {shape}, scale_unit={scale_unit}", end="\t")
            if not torch.allclose(our_recon, gm_recon):
                error = torch.nn.functional.mse_loss(our_recon, gm_recon)
                print(f"L2 error = {error}")
            else:
                print()

    print("running benchmark ...")
    benchmark.run(print_data=True, save_path="./")
