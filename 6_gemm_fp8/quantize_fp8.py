import torch
import triton
import triton.language as tl
from config import BLOCK_SIZE
from utils import bf16_maxexp, fp32_maxexp

_CONFIGS = [
    triton.Config({"BM": bm, "BN": bn}, num_warps=nw, num_stages=ns)
    for (bm, bn, nw, ns) in [
        (1, 1, 4, 2),
        (1, 1, 8, 2),
        (1, 1, 4, 3),
        (1, 1, 8, 3),
        (1, 1, 4, 4),
        (1, 1, 8, 4),
    ]
]


@triton.jit
def quantize_bf16_to_fp8_e8_func(x, max_exp, axis):
    x_floorexp = bf16_maxexp(x, axis, keep_dims=True)
    x_safe_exp = tl.where(x_floorexp < max_exp - 127, max_exp - 127, x_floorexp)
    x_safe_exp = tl.where(x_floorexp > max_exp + 63, max_exp + 63, x_safe_exp)

    inv_scale = tl.cast(
        ((max_exp - x_safe_exp + 127) & 0x0000_00FF) << 23,
        tl.float32,
        bitcast=True,
    )
    x_scale = tl.cast(
        ((x_safe_exp - max_exp + 127) & 0x0000_00FF) << 23,
        tl.float32,
        bitcast=True,
    )
    x_fp8 = tl.cast(x * inv_scale, dtype=tl.float8e5)
    return x_fp8, x_scale


@triton.jit
def quantize_fp32_to_fp8_e8_func(x, max_exp, axis):
    """
    [NaN condition]
    e_bit=8 yields exponent range [2**-126, 2**127] --> NaN if x >= 2**128
    e_bit=5 yields exponent range [2**-14, 2**15]   --> NaN if x >= 2**16

    if `x` is all zero or subnormal:
        x_floorexp = -127
        inv_scale = 2**(max_exp + 127) --> NaN if max_exp >= 1
        x_scale = 2**(-max_exp - 127)
    elif x_floorexp = -126:
        inv_scale = 2**(max_exp + 126) --> NaN if max_exp >= 2
        x_scale = 2**(-max_exp - 126)

    [fix]
    ideal conditions are
    1. `(max_exp - x_safe_exp) <= 127` to prevent NaN scale
        i.e., x_safe_exp >= max_exp - 127
    2. `x_floorexp + max_exp - x_safe_exp <= 15` to prevent NaN after fp8 casting
        i.e., x_safe_exp >= max_exp - 15 + x_floorexp
        RHS range [max_exp - 142, max_exp + 112]
    3. `(x_safe_exp - max_exp) <= 63` to prevent NaN scale during `matmul_scaled`
        i.e., x_safe_exp <= max_exp + 63
    (no need to consider condition 2)
    therefore,
        x_safe_exp = clamp(x_floorexp, min=max_exp-127, max=max_exp+63)
    """
    x_floorexp = fp32_maxexp(x, axis, keep_dims=True)
    x_safe_exp = tl.where(x_floorexp < max_exp - 127, max_exp - 127, x_floorexp)
    x_safe_exp = tl.where(x_floorexp > max_exp + 63, max_exp + 63, x_safe_exp)

    inv_scale = tl.cast(
        ((max_exp - x_safe_exp + 127) & 0x0000_00FF) << 23,
        tl.float32,
        bitcast=True,
    )
    x_scale = tl.cast(
        ((x_safe_exp - max_exp + 127) & 0x0000_00FF) << 23,
        tl.float32,
        bitcast=True,
    )
    x_fp8 = tl.cast(x * inv_scale, dtype=tl.float8e5)
    return x_fp8, x_scale


@triton.autotune(
    configs=_CONFIGS, key=["M", "N", "block_size", "is_sqtile", "out_trans"]
)
@triton.jit
def quantize_fp8_kernel(
    # pointer i/o
    g_x,
    g_x_fp8_r,
    g_x_scale_r,
    g_x_fp8_l,
    g_x_scale_l,
    # pointer shape
    M,
    N,
    # scalar param
    block_size: tl.constexpr,
    max_exp: tl.constexpr,
    is_sqtile: tl.constexpr,
    out_trans: tl.constexpr,
    is_bf16: tl.constexpr,
    # hyperparam
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    pid_m = pid // tl.cdiv(N, BN * block_size)
    pid_n = pid % tl.cdiv(N, BN * block_size)

    stride_m = N
    stride_n = 1

    SM = tl.cdiv(M, block_size)
    SN = tl.cdiv(N, block_size)

    if out_trans:
        stride_out_n = M
        stride_out_m = 1
        stride_out_sn = SM
        stride_out_sm = 1
    else:
        stride_out_m = N
        stride_out_n = 1
        stride_out_sm = SN
        stride_out_sn = 1

    for i_m in tl.static_range(0, BM):
        for i_n in tl.static_range(0, BN):
            offs_m = (
                pid_m * BM * block_size + i_m * block_size + tl.arange(0, block_size)
            )
            offs_n = (
                pid_n * BN * block_size + i_n * block_size + tl.arange(0, block_size)
            )

            g_x_ptrs = g_x + (offs_m[:, None] * stride_m + offs_n[None, :] * stride_n)
            mask_m = offs_m[:, None] < M
            mask_n = offs_n[None, :] < N
            x = tl.load(g_x_ptrs, mask=mask_m & mask_n)

            if out_trans:
                mask_out_n = offs_n[:, None] < N
                mask_out_m = offs_m[None, :] < M
            else:
                mask_out_m = offs_m[:, None] < M
                mask_out_n = offs_n[None, :] < N

            if is_sqtile:
                if is_bf16:
                    x_fp8, x_scale = quantize_bf16_to_fp8_e8_func(x, max_exp, axis=None)
                else:
                    x_fp8, x_scale = quantize_fp32_to_fp8_e8_func(x, max_exp, axis=None)
                if out_trans:
                    g_x_fp8_ptrs = g_x_fp8_r + (
                        offs_n[:, None] * stride_out_n + offs_m[None, :] * stride_out_m
                    )
                    x_fp8 = tl.trans(x_fp8, (1, 0))
                else:
                    g_x_fp8_ptrs = g_x_fp8_r + (
                        offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
                    )
                tl.store(g_x_fp8_ptrs, x_fp8, mask=mask_out_m & mask_out_n)

                # (1, 1) --> (1, BLOCK_SIZE)
                x_scale_r = tl.broadcast_to(x_scale, (1, block_size))

                # (1, 1) --> (BLOCK_SIZE, 1)
                x_scale_l = tl.broadcast_to(x_scale, (block_size, 1))

            else:
                if is_bf16:
                    x_fp8_r, x_scale_r = quantize_bf16_to_fp8_e8_func(x, max_exp, axis=0)
                    x_fp8_l, x_scale_l = quantize_bf16_to_fp8_e8_func(x, max_exp, axis=1)
                else:
                    x_fp8_r, x_scale_r = quantize_fp32_to_fp8_e8_func(x, max_exp, axis=0)
                    x_fp8_l, x_scale_l = quantize_fp32_to_fp8_e8_func(x, max_exp, axis=1)

                if out_trans:
                    # (block_size, block_size) out of (N, M)
                    # value fp8_l.t to GMEM fp8_r
                    g_x_fp8_r_ptrs = g_x_fp8_r + (
                        offs_n[:, None] * stride_out_n + offs_m[None, :] * stride_out_m
                    )
                    x_fp8_l = tl.trans(x_fp8_l, (1, 0))
                    tl.store(g_x_fp8_r_ptrs, x_fp8_l, mask=mask_out_n & mask_out_m)

                    # value fp8_r.t to GMEM fp8_l
                    g_x_fp8_l_ptrs = g_x_fp8_l + (
                        offs_n[:, None] * stride_out_n + offs_m[None, :] * stride_out_m
                    )
                    x_fp8_r = tl.trans(x_fp8_r, (1, 0))
                    tl.store(g_x_fp8_l_ptrs, x_fp8_r, mask=mask_out_n & mask_out_m)
                else:
                    # (block_size, block_size) out of (M, N)
                    # value fp8_r to GMEM fp8_r
                    g_x_fp8_r_ptrs = g_x_fp8_r + (
                        offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
                    )
                    tl.store(g_x_fp8_r_ptrs, x_fp8_r, mask=mask_out_m & mask_out_n)

                    # value fp8_l to GMEM fp8_l
                    g_x_fp8_l_ptrs = g_x_fp8_l + (
                        offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
                    )
                    tl.store(g_x_fp8_l_ptrs, x_fp8_l, mask=mask_out_m & mask_out_n)

            offs_sm = pid_m * BM + i_m + tl.arange(0, 1)
            offs_sn = pid_n * BN + i_n + tl.arange(0, 1)

            if out_trans:
                # value scale_r.t to GMEM scale_l
                # (BLOCK_SIZE, 1) out of (N, cdiv(M, BLOCK_SIZE))
                g_x_scale_l_ptrs = g_x_scale_l + (
                    offs_n[:, None] * stride_out_sn + offs_sm[None, :] * stride_out_m
                )
                mask_out_sm = offs_sm[None, :] < SM
                tl.store(
                    g_x_scale_l_ptrs,
                    tl.trans(x_scale_r, (1, 0)),
                    mask=mask_out_n & mask_out_sm,
                )

                # value scale_l.t to GMEM scale_r
                # (1, BLOCK_SIZE) out of (cdiv(N, BLOCK_SIZE), M)
                g_x_scale_r_ptrs = g_x_scale_r + (
                    offs_sn[:, None] * stride_out_n + offs_m[None, :] * stride_out_sm
                )
                mask_out_sn = offs_sn[:, None] < SN
                tl.store(
                    g_x_scale_r_ptrs,
                    tl.trans(x_scale_l, (1, 0)),
                    mask=mask_out_sn & mask_out_m,
                )

            else:
                # (1, BLOCK_SIZE) out of (cdiv(M, BLOCK_SIZE), N)
                offs_sm = pid_m * BM + i_m + tl.arange(0, 1)
                g_x_scale_r_ptrs = g_x_scale_r + (
                    offs_sm[:, None] * stride_out_m + offs_n[None, :] * stride_out_sn
                )
                mask_out_sm = offs_sm[:, None] < SM
                tl.store(g_x_scale_r_ptrs, x_scale_r, mask=mask_out_sm & mask_out_n)

                # (BLOCK_SIZE, 1) out of (M, cdiv(N, BLOCK_SIZE))
                offs_sn = pid_n * BN + i_n + tl.arange(0, 1)
                g_x_scale_l_ptrs = g_x_scale_l + (
                    offs_m[:, None] * stride_out_sm + offs_sn[None, :] * stride_out_n
                )
                mask_out_sn = offs_sn[None, :] < SN
                tl.store(g_x_scale_l_ptrs, x_scale_l, mask=mask_out_m & mask_out_sn)

    return


def quantize_fp8(x, max_exp: int, is_sqtile: bool = False, out_trans: bool = False):
    assert len(x.shape) == 2
    block_size = BLOCK_SIZE

    M, N = x.shape

    def grid(meta):
        """
        Each subprocess handles (BM * block_size, BN * block_size)
        """
        return (
            triton.cdiv(M, meta["BM"] * block_size)
            * triton.cdiv(N, meta["BN"] * block_size),
        )

    out_shape = (N, M) if out_trans else (M, N)
    scale_r_shape = (triton.cdiv(out_shape[0], block_size), out_shape[1])
    scale_l_shape = (out_shape[0], triton.cdiv(out_shape[1], block_size))

    x_fp8_r = torch.empty(out_shape, device=x.device, dtype=torch.float8_e5m2)
    if is_sqtile:
        x_fp8_l = None
    else:
        x_fp8_l = torch.empty(out_shape, device=x.device, dtype=torch.float8_e5m2)
    x_scale_r = torch.empty(scale_r_shape, device=x.device, dtype=torch.float32)
    x_scale_l = torch.empty(scale_l_shape, device=x.device, dtype=torch.float32)

    quantize_fp8_kernel[grid](
        x,
        x_fp8_r,
        x_scale_r,
        x_fp8_l,
        x_scale_l,
        M,
        N,
        block_size,
        max_exp,
        is_sqtile,
        out_trans,
        is_bf16=x.dtype==torch.bfloat16,
    )

    return x_fp8_r, x_scale_r, x_fp8_l, x_scale_l


if __name__ == "__main__":
    from utils import generate_range, reconstruct

    # test drive
    DEVICE = triton.runtime.driver.active.get_active_torch_device()
    seed = 42
    block_size = 4

    def quantize_fp8_baseline(
        x, max_exp: int, is_sqtile: bool = False, out_trans: bool = False
    ):
        assert len(x.shape) == 2
        block_size = BLOCK_SIZE
        M, N = x.shape

        assert M % block_size == 0
        assert N % block_size == 0

        if is_sqtile:
            x_block = (
                x.view(M // block_size, block_size, N // block_size, block_size)
                .transpose(1, 2)
                .reshape(M // block_size, N // block_size, -1)
            )
            x_maxexp = torch.log2(
                x_block.abs().max(dim=-1, keepdims=True).values
            ).floor()

            x_maxexp_bc = (
                x_maxexp.view(M // block_size, N // block_size, 1, 1)
                .broadcast_to(M // block_size, N // block_size, block_size, block_size)
                .transpose(1, 2)
                .reshape(M, N)
            )
            x_fp8 = (x * 2 ** (max_exp - x_maxexp_bc)).to(torch.float8_e5m2)

            x_scale_r = 2 ** (
                x_maxexp.view(M // block_size, N // block_size, 1)
                .broadcast_to(M // block_size, N // block_size, block_size)
                .reshape(M // block_size, N)
                - max_exp
            )
            x_scale_l = 2 ** (
                x_maxexp.view(M // block_size, 1, N // block_size)
                .broadcast_to(M // block_size, block_size, N // block_size)
                .reshape(M, N // block_size)
                - max_exp
            )

            if out_trans:
                return (
                    x_fp8.t().contiguous(),
                    x_scale_l.t().contiguous(),
                    None,
                    x_scale_r.t().contiguous(),
                )
            else:
                return x_fp8, x_scale_r, None, x_scale_l

        else:
            # left
            x_l = x.view(M, N // block_size, block_size)
            x_maxexp = torch.log2(x_l.abs().max(dim=-1, keepdims=True).values).floor()
            x_scale_l = 2 ** (x_maxexp.view(M, N // block_size) - max_exp)

            x_maxexp_bc = x_maxexp.broadcast_to(M, N // block_size, block_size).reshape(
                M, N
            )
            x_fp8_l = (x * 2 ** (max_exp - x_maxexp_bc)).to(torch.float8_e5m2)

            # right
            x_r = x.view(M // block_size, block_size, N)
            x_maxexp = torch.log2(x_r.abs().max(dim=1, keepdims=True).values).floor()
            x_scale_r = 2 ** (x_maxexp.view(M // block_size, N) - max_exp)

            x_maxexp_bc = x_maxexp.broadcast_to(M // block_size, block_size, N).reshape(
                M, N
            )
            x_fp8_r = (x * 2 ** (max_exp - x_maxexp_bc)).to(torch.float8_e5m2)

            if out_trans:
                return (
                    x_fp8_l.t().contiguous(),
                    x_scale_l.t().contiguous(),
                    x_fp8_r.t().contiguous(),
                    x_scale_r.t().contiguous(),
                )
            else:
                return x_fp8_r, x_scale_r, x_fp8_l, x_scale_l

    for shape in [
        (BLOCK_SIZE, BLOCK_SIZE),
        (BLOCK_SIZE, 2 * BLOCK_SIZE),
        (2 * BLOCK_SIZE, BLOCK_SIZE),
        (2 * BLOCK_SIZE, 2 * BLOCK_SIZE),
    ]:
        for is_sqtile in [True, False]:
            for out_trans in [True, False]:
                print(
                    f"shape : {shape}, sqtile: {is_sqtile}, trans: {out_trans}",
                    end="\t",
                )
                gen = torch.Generator(device=DEVICE).manual_seed(seed)
                t1 = torch.randn(shape, device=DEVICE, generator=gen)

                # row-wise scaling
                for i in range(shape[0]):
                    t1[i, :] *= 2 ** (i / 8)

                print(f"absmax after scaling: {t1.abs().max()}")

                ours_fp8_r, ours_scale_r, ours_fp8_l, ours_scale_l = quantize_fp8(
                    t1, max_exp=5, is_sqtile=is_sqtile, out_trans=out_trans
                )

                gm_fp8_r, gm_scale_r, gm_fp8_l, gm_scale_l = quantize_fp8_baseline(
                    t1, max_exp=5, is_sqtile=is_sqtile, out_trans=out_trans
                )

                if out_trans:
                    t1 = t1.t().contiguous()

                    if not is_sqtile:
                        right_error = torch.nn.functional.mse_loss(
                            gm_fp8_r.to(torch.float), ours_fp8_r.to(torch.float)
                        )
                        left_error = torch.nn.functional.mse_loss(
                            gm_fp8_l.to(torch.float), ours_fp8_l.to(torch.float)
                        )
                        print(f"right tensor error : {right_error}")
                        print(f"left tensor error : {left_error}")
                        if right_error > 1e-1:
                            print(gm_fp8_r)
                            print(ours_fp8_r)
                            exit()
                        if left_error > 1e-1:
                            print(gm_fp8_l)
                            print(ours_fp8_l)
                            exit()

                    print(
                        f"right scale error : {torch.nn.functional.mse_loss(gm_scale_r, ours_scale_r)}"
                    )
                    print(
                        f"left scale error : {torch.nn.functional.mse_loss(gm_scale_l, ours_scale_l)}"
                    )

                assert ours_fp8_r.shape == gm_fp8_r.shape
                assert ours_scale_r.shape == gm_scale_r.shape, (
                    f"scale mismatch: lhs {ours_scale_r.shape} vs. rhs {gm_scale_r.shape}"
                )
                assert ours_scale_l.shape == gm_scale_l.shape

                if is_sqtile:
                    assert ours_fp8_l is None
                    assert gm_fp8_l is None
                else:
                    assert ours_fp8_l.shape == gm_fp8_l.shape

                ours_r_recon = reconstruct(ours_fp8_r, ours_scale_r)
                gm_r_recon = reconstruct(gm_fp8_r, gm_scale_r)

                if is_sqtile:
                    ours_l_recon = reconstruct(ours_fp8_r, ours_scale_l)
                    gm_l_recon = reconstruct(gm_fp8_r, gm_scale_l)
                else:
                    ours_l_recon = reconstruct(ours_fp8_l, ours_scale_l)
                    gm_l_recon = reconstruct(gm_fp8_l, gm_scale_l)

                right = torch.allclose(gm_r_recon, ours_r_recon, rtol=0.25)
                if is_sqtile:
                    print(f"test: ours compared to gm --> block={right}")
                else:
                    left = torch.allclose(gm_l_recon, ours_l_recon, rtol=0.25)
                    print(f"test: ours compared to gm --> left={left}, right={right}")

                if is_sqtile:
                    right = torch.allclose(t1, gm_r_recon, rtol=0.25)
                    print(f"validate: orig compared to gm --> block={right}")
                else:
                    left = torch.allclose(t1, gm_l_recon, rtol=0.25)
                    right = torch.allclose(t1, gm_r_recon, rtol=0.25)
                    print(
                        f"validate: orig compared to gm --> left={left}, right={right}"
                    )

                if is_sqtile:
                    right = torch.allclose(t1, ours_r_recon, rtol=0.25)
                    print(f"validate: orig compared to ours --> block={right}")
                else:
                    left = torch.allclose(t1, ours_l_recon, rtol=0.25)
                    right = torch.allclose(t1, ours_r_recon, rtol=0.25)
                    print(
                        f"validate: orig compared to ours --> left={left}, right={right}"
                    )

    # benchmark
    _CONFIGS = [
        triton.testing.Benchmark(
            x_names=["N"],
            x_vals=generate_range(128, 8, 16384 + 1),
            line_arg="provider",
            line_vals=["triton", "torch"],
            line_names=["Triton", "Torch"],
            styles=[("green", "-"), ("blue", "-")],
            ylabel="Gelem/s",
            plot_name="-".join(
                ["quantize_fp8", f"sqtile{is_sqtile}", f"trans{out_trans}"]
            ),
            args={"is_sqtile": is_sqtile, "out_trans": out_trans},
        )
        for (is_sqtile, out_trans) in [
            (True, True),
            (True, False),
            (False, True),
            (False, False),
        ]
    ]

    @triton.testing.perf_report(_CONFIGS)
    def benchmark(N, provider, is_sqtile, out_trans):
        stream = getattr(torch, DEVICE.type).Stream()
        getattr(torch, DEVICE.type).set_stream(stream)
        if provider == "torch":
            func = quantize_fp8_baseline
        else:
            assert provider == "triton"
            func = quantize_fp8

        x = torch.randn((N, N), device=DEVICE)

        def bench_run():
            func(x, max_exp=9, is_sqtile=is_sqtile, out_trans=out_trans)

        ms = triton.testing.do_bench(bench_run)
        gelems = lambda ms: N * N * 1e-9 / (ms * 1e-3)
        return gelems(ms)

    print("running benchmark ...")
    benchmark.run(print_data=True, save_path="./")
