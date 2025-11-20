"""
Scaled matrix multiplication where quantization granularity is one of:
- (vector, vector)
- (vector, tile)
- (tile, vector)
- (tile, tile)
and output can be transposed
"""

import torch
import triton
import triton.language as tl
from config import BLOCK_SIZE, MATMUL_CONFIGS
from triton.language.extra import libdevice


@triton.autotune(configs=MATMUL_CONFIGS, key=["M", "N", "K", "out_transpose"])
@triton.jit
def matmul_scaled_kernel(
    # pointer i/o
    g_lhs,
    g_lhs_scale,
    g_rhs,
    g_rhs_scale,
    g_bias,
    g_result,
    # pointer shape
    M,
    N,
    K,
    # scalar param
    is_bias: tl.constexpr,
    out_transpose: tl.constexpr,
    # hyperparam
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    # fixed GM
    # GM: tl.constexpr = 8

    # Adaptive GM
    # # consistently outperforms fixed GM=8 on average
    # # BF16 bias=False --> 0.0426197 %
    # # BF16 bias=True  --> 0.671729  %
    # # FP8  bias=False --> 0.2751545 %
    # # FP8  bias=True  --> 2.2620753 %
    #
    # # GM = sqrt(N / BM) and floor-power-of-two
    # # GM = 1 if N / BM in [1, 4)    ..0000_0000_0001  0
    # # GM = 2 if N / BM in [4, 16)   ..0000_0000_0100  2
    # # GM = 4 if N / BM in [16, 64)  ..0000_0001_0000  4
    # # GM = 8 if N / BM in [64, 256) ..0000_0100_0000  8
    GM: tl.constexpr = 1 << (max(0, 31 - libdevice.clz(N // BM)) >> 1)

    pid = tl.program_id(axis=0)
    pid_per_m = tl.cdiv(M, BM)
    pid_per_n = tl.cdiv(N, BN)

    # 1 block group === (BM * block_size * GM, N)
    # pid per group === BM * block_size * GM * N / (BM * block_size * BN * block_size)
    #               === BM * block_size * GM * pid_per_n / (BM * block_size)
    #               === GM * pid_per_n
    pid_per_group = pid_per_n * GM

    group_id = pid // pid_per_group
    group_base_m = group_id * GM
    actual_bgroup = min(GM, pid_per_m - group_base_m)

    pid_g = pid % pid_per_group
    pid_m = group_base_m + pid_g % actual_bgroup
    pid_n = pid_g // actual_bgroup

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)

    lhs_stride_m = K
    lhs_stride_k = 1

    rhs_stride_k = N
    rhs_stride_n = 1

    SK = tl.cdiv(K, BK)
    lhs_stride_sm = SK
    lhs_stride_sk = 1
    rhs_stride_sk = N
    rhs_stride_sn = 1

    accum = tl.zeros((BM, BN), dtype=tl.float32)

    for k in tl.range(tl.cdiv(K, BK)):
        offs_k = k * BK + tl.arange(0, BK)

        # load lhs (BM, BK) out of (M, K)
        g_lhs_ptrs = (
            g_lhs + offs_m[:, None] * lhs_stride_m + offs_k[None, :] * lhs_stride_k
        )
        lhs_mask_m = offs_m[:, None] < M
        lhs_mask_k = offs_k[None, :] < K
        lhs = tl.load(g_lhs_ptrs, mask=lhs_mask_m & lhs_mask_k)

        # load lhs_scale (BM, 1) out of (M, cdiv(K, BK(==block_size)))
        offs_sk = k + tl.arange(0, 1)
        g_lhs_scale_ptrs = (
            g_lhs_scale
            + offs_m[:, None] * lhs_stride_sm
            + offs_sk[None, :] * lhs_stride_sk
        )
        lhs_mask_sk = offs_sk[None, :] < SK
        lhs_scale = tl.load(g_lhs_scale_ptrs, mask=lhs_mask_m & lhs_mask_sk)

        # load rhs (BK, BN)
        g_rhs_ptrs = (
            g_rhs + offs_k[:, None] * rhs_stride_k + offs_n[None, :] * rhs_stride_n
        )
        rhs_mask_k = offs_k[:, None] < K
        rhs_mask_n = offs_n[None, :] < N
        rhs = tl.load(g_rhs_ptrs, mask=rhs_mask_k & rhs_mask_n)

        # load rhs_scale (1, BN) out of (cdiv(K, BK(==block_size)), N)
        g_rhs_scale_ptrs = (
            g_rhs_scale
            + offs_sk[:, None] * rhs_stride_sk
            + offs_n[None, :] * rhs_stride_sn
        )
        rhs_mask_sk = offs_sk[:, None] < SK
        rhs_scale = tl.load(g_rhs_scale_ptrs, mask=rhs_mask_sk & rhs_mask_n)

        # compute fp8 dot & scaled-accumulate
        accum += tl.dot(lhs, rhs) * (lhs_scale * rhs_scale)

    # handle bias
    if is_bias:
        # load bias
        g_bias_ptrs = g_bias + offs_n
        bias_mask = offs_n < N
        bias = tl.load(g_bias_ptrs, mask=bias_mask)

        accum += bias.reshape((1, BN)).broadcast_to((BM, BN))

    # store result
    if out_transpose:
        r_stride_n = M
        r_stride_m = 1
        g_result_ptrs = (
            g_result + offs_n[:, None] * r_stride_n + offs_m[None, :] * r_stride_m
        )
        r_mask_n = offs_n[:, None] < N
        r_mask_m = offs_m[None, :] < M

        tl.store(g_result_ptrs, tl.trans(accum, (1, 0)), mask=r_mask_n & r_mask_m)
    else:
        r_stride_m = N
        r_stride_n = 1
        g_result_ptrs = (
            g_result + offs_m[:, None] * r_stride_m + offs_n[None, :] * r_stride_n
        )
        r_mask_m = offs_m[:, None] < M
        r_mask_n = offs_n[None, :] < N

        tl.store(g_result_ptrs, accum, mask=r_mask_m & r_mask_n)

    return


def matmul_scaled(lhs, lhs_scale, rhs, rhs_scale, bias=None, out_transpose=False):
    assert len(lhs.shape) == 2
    assert len(rhs.shape) == 2
    assert lhs.shape[-1] == rhs.shape[0]

    assert triton.cdiv(lhs.shape[-1], BLOCK_SIZE) == lhs_scale.shape[-1]
    assert triton.cdiv(rhs.shape[0], BLOCK_SIZE) == rhs_scale.shape[0]
    if bias is not None:
        assert len(bias.shape) == 1
        assert bias.shape[0] == rhs.shape[-1]

    M, K = lhs.shape
    _, N = rhs.shape

    if out_transpose:
        result = torch.empty((N, M), device=lhs.device)
    else:
        result = torch.empty((M, N), device=lhs.device)

    def grid(meta):
        """
        Each subprocess handles [BM, BN] of resulting matrix
        """
        return (triton.cdiv(M, meta["BM"]) * triton.cdiv(N, meta["BN"]),)

    matmul_scaled_kernel[grid](
        lhs,
        lhs_scale,
        rhs,
        rhs_scale,
        bias,
        result,
        M,
        N,
        K,
        bias is not None,
        out_transpose,
    )

    return result


if __name__ == "__main__":
    from quantize_fp8 import quantize_fp8
    from utils import generate_range, reconstruct

    # test drive
    DEVICE = triton.runtime.driver.active.get_active_torch_device()
    seed = 42
    max_exp = 9

    for M, N, K in [
        (32, 32, BLOCK_SIZE),
        (64, 64, BLOCK_SIZE),
        (128, 128, BLOCK_SIZE),
        (32, 32, 2 * BLOCK_SIZE),
        (64, 64, 2 * BLOCK_SIZE),
        (128, 128, 2 * BLOCK_SIZE),
        (32, 32, BLOCK_SIZE + 16),
        (64, 64, BLOCK_SIZE + 16),
        (128, 128, BLOCK_SIZE + 16),
    ]:
        print(f"(M, N, K) = ({M}, {N}, {K})", end="\t")
        gen = torch.Generator(device=DEVICE).manual_seed(seed)

        t1 = torch.randn((M, K), generator=gen, device=DEVICE)
        # row-wise scale
        for i in range(M):
            t1[i, :] *= 2 ** (i / 16)
        t2 = torch.randn((K, N), generator=gen, device=DEVICE)

        # Quantize vector-wise, aligned to reduction dimension
        _, _, t1_fp8_l, t1_scale_l = quantize_fp8(t1, max_exp, is_sqtile=False)
        t2_fp8_r, t2_scale_r, _, _ = quantize_fp8(t2, max_exp, is_sqtile=False)

        assert t1_fp8_l.shape == (M, K)
        assert t2_fp8_r.shape == (K, N)
        assert t1_scale_l.shape == (M, triton.cdiv(K, BLOCK_SIZE))
        assert t2_scale_r.shape == (triton.cdiv(K, BLOCK_SIZE), N)

        t1_recon_l = reconstruct(t1_fp8_l, t1_scale_l)
        t2_recon_r = reconstruct(t2_fp8_r, t2_scale_r)

        result_gm = torch.matmul(t1_recon_l, t2_recon_r)
        result_ours = matmul_scaled(t1_fp8_l, t1_scale_l, t2_fp8_r, t2_scale_r)

        result_ours_t = matmul_scaled(
            t1_fp8_l, t1_scale_l, t2_fp8_r, t2_scale_r, out_transpose=True
        ).t()

        print(
            f"check transpose: {torch.allclose(result_ours, result_ours_t)}", end="\t"
        )

        error = torch.nn.functional.mse_loss(result_gm, result_ours)
        print(f"t1 @ t2 --> error={error}")
        if error > 1:
            print(result_gm)
            print(result_ours)

    # benchmark
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
                [
                    "matmul",
                    f"block{BLOCK_SIZE}",
                    f"bias{is_bias}",
                    f"out_t{out_transpose}",
                ]
            ),
            args={"is_bias": is_bias, "out_transpose": out_transpose},
        )
        for (is_bias, out_transpose) in [
            (True, False),
            (False, False),
            (True, True),
            (False, True),
        ]
    ]

    def matmul_scaled_gm(
        lhs_fp8, lhs_scale, rhs_fp8, rhs_scale, bias=None, out_transpose=False
    ):
        lhs = reconstruct(lhs_fp8, lhs_scale)
        rhs = reconstruct(rhs_fp8, rhs_scale)
        result = torch.matmul(lhs, rhs)
        if bias is not None:
            result += bias
        if out_transpose:
            result = result.t()
        return result

    @triton.testing.perf_report(_CONFIGS)
    def benchmark(N, provider, is_bias, out_transpose):
        stream = getattr(torch, DEVICE.type).Stream()
        getattr(torch, DEVICE.type).set_stream(stream)
        if provider == "torch":
            func = matmul_scaled_gm
        else:
            assert provider == "triton"
            func = matmul_scaled

        lhs = torch.randn((N, N), device=DEVICE)
        rhs = torch.randn((N, N), device=DEVICE)

        lhs_fp8_r, lhs_scale_r, lhs_fp8_l, lhs_scale_l = quantize_fp8(
            lhs, max_exp=max_exp, is_sqtile=False
        )
        rhs_fp8, rhs_scale_r, _, rhs_scale_l = quantize_fp8(rhs, max_exp=max_exp, is_sqtile=True)

        if is_bias:
            bias = torch.randn((N), device=DEVICE)
        else:
            bias = None

        def bench_run():
            func(lhs_fp8_l, lhs_scale_l, rhs_fp8, rhs_scale_r, bias, out_transpose)

        ms = triton.testing.do_bench(bench_run)
        tflops = lambda ms: 2 * N * N * N * 1e-12 / (ms * 1e-3)
        return tflops(ms)

    print("running benchmark ...")
    benchmark.run(print_data=True, save_path="./")
