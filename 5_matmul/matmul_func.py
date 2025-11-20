import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

_CONFIGS = [
    triton.Config({"BM": bm, "BN": bn, "BK": bk}, num_stages=ns, num_warps=nw)
    for (bm, bn, bk, gm, ns, nw) in [
        (128, 256, 64, 8, 3, 8),
        (64, 256, 32, 8, 4, 4),
        (128, 128, 32, 8, 4, 4),
        (128, 64, 32, 8, 4, 4),
        (64, 128, 32, 8, 4, 4),
        (128, 32, 32, 8, 4, 4),
        (64, 32, 32, 8, 5, 2),
        (32, 64, 32, 8, 5, 2),
        # fp8
        (128, 256, 128, 8, 3, 8),
        (256, 128, 128, 8, 3, 8),
        (256, 64, 128, 8, 4, 4),
        (64, 256, 128, 8, 4, 4),
        (128, 128, 128, 8, 4, 4),
        (128, 64, 64, 8, 4, 4),
        (64, 128, 64, 8, 4, 4),
        (128, 32, 64, 8, 4, 4),
    ]
]


@triton.autotune(configs=_CONFIGS, key=["M", "N", "K"])
@triton.jit
def matmul_kernel(
    # pointer i/o
    g_lhs,
    g_rhs,
    g_bias,
    g_result,
    # pointer shape
    M,
    N,
    K,
    # scalar param
    is_bias: tl.constexpr,
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
    # GM = sqrt(N / BM) and floor-power-of-two
    # GM = 1 if N / BM in [1, 4)    ..0000_0000_0001  0
    # GM = 2 if N / BM in [4, 16)   ..0000_0000_0100  2
    # GM = 4 if N / BM in [16, 64)  ..0000_0001_0000  4
    # GM = 8 if N / BM in [64, 256) ..0000_0100_0000  8
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

    accum = tl.zeros((BM, BN), dtype=tl.float32)

    for k in tl.range(tl.cdiv(K, BK)):
        offs_k = k * BK + tl.arange(0, BK)

        # load lhs
        g_lhs_ptrs = (
            g_lhs + offs_m[:, None] * lhs_stride_m + offs_k[None, :] * lhs_stride_k
        )
        lhs_mask_m = offs_m[:, None] < M
        lhs_mask_k = offs_k[None, :] < K
        lhs = tl.load(g_lhs_ptrs, mask=lhs_mask_m & lhs_mask_k)

        # load rhs
        g_rhs_ptrs = (
            g_rhs + offs_k[:, None] * rhs_stride_k + offs_n[None, :] * rhs_stride_n
        )
        rhs_mask_k = offs_k[:, None] < K
        rhs_mask_n = offs_n[None, :] < N
        rhs = tl.load(g_rhs_ptrs, mask=rhs_mask_k & rhs_mask_n)

        # compute dot & accumulate
        accum += tl.dot(lhs, rhs)

    # handle bias
    if is_bias:
        # load bias
        g_bias_ptrs = g_bias + offs_n
        bias_mask = offs_n < N
        bias = tl.load(g_bias_ptrs, mask=bias_mask)

        accum += bias.reshape((1, BN)).broadcast_to((BM, BN))

    # store result
    r_stride_m = N
    r_stride_n = 1
    g_result_ptrs = (
        g_result + offs_m[:, None] * r_stride_m + offs_n[None, :] * r_stride_n
    )
    r_mask_m = offs_m[:, None] < M
    r_mask_n = offs_n[None, :] < N
    tl.store(g_result_ptrs, accum, mask=r_mask_m & r_mask_n)

    return


def matmul(lhs, rhs, bias=None):
    assert len(rhs.shape) == 2
    assert lhs.shape[-1] == rhs.shape[0]
    if bias is not None:
        assert len(bias.shape) == 1
        assert bias.shape[0] == rhs.shape[-1]

    lhs_orig_shape = lhs.shape
    lhs = lhs.view(-1, lhs.shape[-1])
    M, K = lhs.shape
    _, N = rhs.shape

    result = torch.empty((M, N), device=lhs.device)

    def grid(meta):
        """
        Each subprocess handles [BM, BN] of resulting matrix
        """
        return (triton.cdiv(M, meta["BM"]) * triton.cdiv(N, meta["BN"]),)

    matmul_kernel[grid](lhs, rhs, bias, result, M, N, K, bias is not None)

    return result.view([*lhs_orig_shape[:-1], N])


if __name__ == "__main__":
    # test drive
    raise NotImplementedError
