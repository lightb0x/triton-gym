import torch
import triton
import triton.language as tl

_CONFIGS = [
    triton.Config({"BN": bn}, num_stages=ns)
    for (bn, ns) in [
        (4096, 3),
        (4096, 4),
        (8192, 3),
        (8192, 4),
    ]
]


@triton.autotune(
    configs=_CONFIGS,
    key=["N"],
)
@triton.jit
def mask_kernel(
    # pointer i/o
    g_x,
    g_xmask,
    g_result,
    # pointer shape
    N: tl.constexpr,
    # hyperparameter
    BN: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    offs = pid * BN + tl.arange(0, BN)
    mask = offs < N

    g_x_ptrs = g_x + offs
    xdata = tl.load(g_x_ptrs, mask=mask)

    g_mask_ptrs = g_xmask + offs

    xmask = tl.load(g_mask_ptrs, mask=mask)

    # result = tl.cast(xmask * 0.5, xdata.dtype) * xdata
    result = xdata * 0.5 * xmask

    g_result_ptrs = g_result + offs
    tl.store(g_result_ptrs, result, mask=mask)

    return


def mask_func(x, x_mask):
    assert x.numel() == x_mask.numel()
    assert x_mask.dtype == torch.uint8

    orig_shape = x.shape

    # NOTE when `x` is gradient, we need copy of it
    # one of these passes:
    x = x.contiguous()
    # x = x.clone().view(-1)

    # while one of these fails:
    # x = x.reshape(-1)
    # x = x.view(-1)

    N = x.numel()

    result = torch.empty_like(x)

    def grid(meta):
        return (triton.cdiv(N, meta["BN"]),)

    mask_kernel[grid](x, x_mask, result, N)

    return result.view(orig_shape)


if __name__ == "__main__":
    from add_relu_fwd import add_relu_fwd_func

    # test drive
    DEVICE = triton.runtime.driver.active.get_active_torch_device()

    for i in range(4, 14):
        shape = (2**i,)

        gen = torch.Generator(DEVICE).manual_seed(42)
        t1 = torch.randn(shape, generator=gen, device=DEVICE)
        t2 = torch.randn(shape, generator=gen, device=DEVICE)
        t1[3] = 0
        t2[3] = 0
        result, mask = add_relu_fwd_func(t1, t2)

        added_gm = t1 + t2
        mask_gm = torch.where(added_gm > 0, 1.0, torch.where(added_gm == 0, 0.5, 0.0))
        # sim_gradient = torch.ones(shape, device=DEVICE)
        sim_gradient = torch.randn(shape, generator=gen, device=DEVICE) * 128

        our_gradient = mask_func(sim_gradient, mask)
        baseline_gradient = sim_gradient * mask_gm
        if not all(torch.isclose(baseline_gradient, our_gradient)):
            print(f"{2**i} grad : {baseline_gradient}, {our_gradient}")
