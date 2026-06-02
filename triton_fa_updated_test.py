from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl


@triton.jit
def _attn_fwd(
    Q,  # [B, H, S, D]
    K,  # [B, H, S, D]
    V,  # [B, H, S, D]
    softmax_scale,
    M,  # [B, H, S], logsumexp
    O,  # [B, H, S, D]
    stride_Q_batch,
    stride_Q_head,
    stride_Q_seq,
    stride_Q_dim,
    stride_K_batch,
    stride_K_head,
    stride_K_seq,
    stride_K_dim,
    stride_V_batch,
    stride_V_head,
    stride_V_seq,
    stride_V_dim,
    stride_O_batch,
    stride_O_head,
    stride_O_seq,
    stride_O_dim,
    NUM_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    block_q = tl.program_id(0)
    batch_head = tl.program_id(1)

    batch = batch_head // NUM_HEADS
    head = batch_head % NUM_HEADS

    q_offset = batch * stride_Q_batch + head * stride_Q_head
    k_offset = batch * stride_K_batch + head * stride_K_head
    v_offset = batch * stride_V_batch + head * stride_V_head
    o_offset = batch * stride_O_batch + head * stride_O_head

    offs_q = block_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    offs_kv_base = tl.arange(0, BLOCK_SIZE_KV)
    offs_d = tl.arange(0, HEAD_DIM)

    valid_q = offs_q < SEQ_LEN

    Q_block = tl.load(
        Q + q_offset + offs_q[:, None] * stride_Q_seq + offs_d[None, :] * stride_Q_dim,
        mask=valid_q[:, None],
        other=0.0,
    )

    m_i = tl.full((BLOCK_SIZE_Q,), -1.0e20, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_SIZE_Q,), dtype=tl.float32)
    O_block = tl.zeros((BLOCK_SIZE_Q, HEAD_DIM), dtype=tl.float32)

    for start_kv in range(0, SEQ_LEN, BLOCK_SIZE_KV):
        offs_kv = start_kv + offs_kv_base
        valid_kv = offs_kv < SEQ_LEN

        K_T_block = tl.load(
            K
            + k_offset
            + offs_d[:, None] * stride_K_dim
            + offs_kv[None, :] * stride_K_seq,
            mask=valid_kv[None, :],
            other=0.0,
        )

        V_block = tl.load(
            V
            + v_offset
            + offs_kv[:, None] * stride_V_seq
            + offs_d[None, :] * stride_V_dim,
            mask=valid_kv[:, None],
            other=0.0,
        )

        QK_block = tl.dot(Q_block, K_T_block, input_precision="ieee") * softmax_scale

        keep = valid_q[:, None] & valid_kv[None, :]

        if CAUSAL:
            keep = keep & (offs_q[:, None] >= offs_kv[None, :])

        QK_block = tl.where(keep, QK_block, -1.0e20)

        m_ij = tl.maximum(m_i, tl.max(QK_block, axis=1))
        P_block = tl.exp(QK_block - m_ij[:, None])

        alpha = tl.exp(m_i - m_ij)
        l_ij = tl.sum(P_block, axis=1)

        O_block = O_block * alpha[:, None] + tl.dot(P_block, V_block, input_precision="ieee")
        l_i = l_i * alpha + l_ij
        m_i = m_ij

    O_block = O_block / l_i[:, None]
    M_block = m_i + tl.log(l_i)

    tl.store(
        M + batch_head * SEQ_LEN + offs_q,
        M_block,
        mask=valid_q,
    )

    tl.store(
        O + o_offset + offs_q[:, None] * stride_O_seq + offs_d[None, :] * stride_O_dim,
        O_block,
        mask=valid_q[:, None],
    )


@triton.jit
def _attn_bwd_preprocess(
    O,
    dO,
    D,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
):
    block_q = tl.program_id(0)
    batch_head = tl.program_id(1)

    offs_q = block_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    offs_d = tl.arange(0, HEAD_DIM)

    valid_q = offs_q < SEQ_LEN

    O_block = tl.load(
        O + batch_head * SEQ_LEN * HEAD_DIM + offs_q[:, None] * HEAD_DIM + offs_d[None, :],
        mask=valid_q[:, None],
        other=0.0,
    )

    dO_block = tl.load(
        dO + batch_head * SEQ_LEN * HEAD_DIM + offs_q[:, None] * HEAD_DIM + offs_d[None, :],
        mask=valid_q[:, None],
        other=0.0,
    ).to(tl.float32)

    D_block = tl.sum(O_block * dO_block, axis=1)

    tl.store(
        D + batch_head * SEQ_LEN + offs_q,
        D_block,
        mask=valid_q,
    )


@triton.jit
def _attn_bwd_dq(
    Q,
    K,
    V,
    softmax_scale,
    dO,
    dQ,
    M,
    D,
    stride_batch,
    stride_head,
    stride_seq,
    stride_dim,
    NUM_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    block_q = tl.program_id(0)
    batch_head = tl.program_id(1)

    batch = batch_head // NUM_HEADS
    head = batch_head % NUM_HEADS

    offset = batch * stride_batch + head * stride_head
    offset_lse = batch_head * SEQ_LEN

    offs_q = block_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_kv_base = tl.arange(0, BLOCK_KV)
    offs_d = tl.arange(0, HEAD_DIM)

    valid_q = offs_q < SEQ_LEN

    Q_block = tl.load(
        Q + offset + offs_q[:, None] * stride_seq + offs_d[None, :] * stride_dim,
        mask=valid_q[:, None],
        other=0.0,
    )

    dO_block = tl.load(
        dO + offset + offs_q[:, None] * stride_seq + offs_d[None, :] * stride_dim,
        mask=valid_q[:, None],
        other=0.0,
    )

    M_block = tl.load(
        M + offset_lse + offs_q,
        mask=valid_q,
        other=0.0,
    )[:, None]

    D_block = tl.load(
        D + offset_lse + offs_q,
        mask=valid_q,
        other=0.0,
    )

    dQ_block = tl.zeros((BLOCK_Q, HEAD_DIM), dtype=tl.float32)

    for start_kv in range(0, SEQ_LEN, BLOCK_KV):
        offs_kv = start_kv + offs_kv_base
        valid_kv = offs_kv < SEQ_LEN

        K_T_block = tl.load(
            K + offset + offs_d[:, None] * stride_dim + offs_kv[None, :] * stride_seq,
            mask=valid_kv[None, :],
            other=0.0,
        )

        V_T_block = tl.load(
            V + offset + offs_d[:, None] * stride_dim + offs_kv[None, :] * stride_seq,
            mask=valid_kv[None, :],
            other=0.0,
        )

        QK_block = tl.dot(Q_block, K_T_block, input_precision="ieee") * softmax_scale


        keep = valid_q[:, None] & valid_kv[None, :]

        if CAUSAL:
            keep = keep & (offs_q[:, None] >= offs_kv[None, :])

        QK_block = tl.where(keep, QK_block, -1.0e20)
        P_block = tl.exp(QK_block - M_block)
        P_block = tl.where(keep, P_block, 0.0)

        dP_block = tl.dot(dO_block, V_T_block, input_precision="ieee")
        dS_block = P_block * (dP_block - D_block[:, None])

        dQ_block += softmax_scale * tl.dot(dS_block, tl.trans(K_T_block),input_precision="ieee")

    tl.store(
        dQ + offset + offs_q[:, None] * stride_seq + offs_d[None, :] * stride_dim,
        dQ_block,
        mask=valid_q[:, None],
    )


@triton.jit
def _attn_bwd_dk_dv(
    Q,
    K,
    V,
    softmax_scale,
    dO,
    dK,
    dV,
    M,
    D,
    stride_batch,
    stride_head,
    stride_seq,
    stride_dim,
    NUM_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    block_kv = tl.program_id(0)
    batch_head = tl.program_id(1)

    batch = batch_head // NUM_HEADS
    head = batch_head % NUM_HEADS

    offset = batch * stride_batch + head * stride_head
    offset_lse = batch_head * SEQ_LEN

    offs_kv = block_kv * BLOCK_KV + tl.arange(0, BLOCK_KV)
    offs_q_base = tl.arange(0, BLOCK_Q)
    offs_d = tl.arange(0, HEAD_DIM)

    valid_kv = offs_kv < SEQ_LEN

    K_block = tl.load(
        K + offset + offs_kv[:, None] * stride_seq + offs_d[None, :] * stride_dim,
        mask=valid_kv[:, None],
        other=0.0,
    )

    V_block = tl.load(
        V + offset + offs_kv[:, None] * stride_seq + offs_d[None, :] * stride_dim,
        mask=valid_kv[:, None],
        other=0.0,
    )

    dK_block = tl.zeros((BLOCK_KV, HEAD_DIM), dtype=tl.float32)
    dV_block = tl.zeros((BLOCK_KV, HEAD_DIM), dtype=tl.float32)

    for start_q in range(0, SEQ_LEN, BLOCK_Q):
        offs_q = start_q + offs_q_base
        valid_q = offs_q < SEQ_LEN

        Q_T_block = tl.load(
            Q + offset + offs_d[:, None] * stride_dim + offs_q[None, :] * stride_seq,
            mask=valid_q[None, :],
            other=0.0,
        )

        dO_block = tl.load(
            dO + offset + offs_q[:, None] * stride_seq + offs_d[None, :] * stride_dim,
            mask=valid_q[:, None],
            other=0.0,
        )

        M_block = tl.load(
            M + offset_lse + offs_q,
            mask=valid_q,
            other=0.0,
        )

        D_block = tl.load(
            D + offset_lse + offs_q,
            mask=valid_q,
            other=0.0,
        )

        QK_T_block = tl.dot(K_block, Q_T_block, input_precision="ieee") * softmax_scale

        keep = valid_kv[:, None] & valid_q[None, :]

        if CAUSAL:
            # P_T_block is [key, query], allowed iff key <= query.
            keep = keep & (offs_q[None, :] >= offs_kv[:, None])

        QK_T_block = tl.where(keep, QK_T_block, -1.0e20)
        P_T_block = tl.exp(QK_T_block - M_block[None, :])

        P_T_block = tl.where(keep, P_T_block, 0.0)

        dV_block += tl.dot(P_T_block, dO_block, input_precision="ieee")

        dP_T_block = tl.dot(V_block, tl.trans(dO_block), input_precision="ieee")
        dS_T_block = P_T_block * (dP_T_block - D_block[None, :])

        dK_block += softmax_scale * tl.dot(dS_T_block.to(tl.float32), tl.trans(Q_T_block), input_precision="ieee")

    tl.store(
        dV + offset + offs_kv[:, None] * stride_seq + offs_d[None, :] * stride_dim,
        dV_block,
        mask=valid_kv[:, None],
    )

    tl.store(
        dK + offset + offs_kv[:, None] * stride_seq + offs_d[None, :] * stride_dim,
        dK_block,
        mask=valid_kv[:, None],
    )


class TritonAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        causal: bool,
        softmax_scale: float,
    ) -> torch.Tensor:
        assert Q.is_cuda and K.is_cuda and V.is_cuda
        assert Q.shape == K.shape == V.shape
        assert Q.is_contiguous()
        assert K.is_contiguous()
        assert V.is_contiguous()

        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape

        O = torch.empty_like(Q)
        M = torch.empty((BATCH_SIZE, NUM_HEADS, SEQ_LEN), device=Q.device, dtype=torch.float32)

        BLOCK_SIZE_Q = 64
        BLOCK_SIZE_KV = 64

        grid = (
            triton.cdiv(SEQ_LEN, BLOCK_SIZE_Q),
            BATCH_SIZE * NUM_HEADS,
        )

        _attn_fwd[grid](
            Q,
            K,
            V,
            softmax_scale,
            M,
            O,
            Q.stride(0),
            Q.stride(1),
            Q.stride(2),
            Q.stride(3),
            K.stride(0),
            K.stride(1),
            K.stride(2),
            K.stride(3),
            V.stride(0),
            V.stride(1),
            V.stride(2),
            V.stride(3),
            O.stride(0),
            O.stride(1),
            O.stride(2),
            O.stride(3),
            NUM_HEADS=NUM_HEADS,
            SEQ_LEN=SEQ_LEN,
            HEAD_DIM=HEAD_DIM,
            BLOCK_SIZE_Q=BLOCK_SIZE_Q,
            BLOCK_SIZE_KV=BLOCK_SIZE_KV,
            CAUSAL=causal,
            num_warps=4,
            num_stages=3,
        )

        ctx.save_for_backward(Q, K, V, O, M)
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        return O

    @staticmethod
    def backward(ctx: Any, dO: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None]:
        Q, K, V, O, M = ctx.saved_tensors

        dO = dO.contiguous()

        assert Q.is_contiguous()
        assert K.is_contiguous()
        assert V.is_contiguous()
        assert O.is_contiguous()
        assert dO.is_contiguous()

        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape

        dQ = torch.empty_like(Q)
        dK = torch.empty_like(K)
        dV = torch.empty_like(V)

        D = torch.empty_like(M)

        BLOCK_Q = 64
        BLOCK_KV = 64

        preprocess_grid = (
            triton.cdiv(SEQ_LEN, BLOCK_Q),
            BATCH_SIZE * NUM_HEADS,
        )

        _attn_bwd_preprocess[preprocess_grid](
            O,
            dO,
            D,
            SEQ_LEN=SEQ_LEN,
            HEAD_DIM=HEAD_DIM,
            BLOCK_SIZE_Q=BLOCK_Q,
            num_warps=4,
            num_stages=3,
        )

        grid_q = (
            triton.cdiv(SEQ_LEN, BLOCK_Q),
            BATCH_SIZE * NUM_HEADS,
        )

        _attn_bwd_dq[grid_q](
            Q,
            K,
            V,
            ctx.softmax_scale,
            dO,
            dQ,
            M,
            D,
            Q.stride(0),
            Q.stride(1),
            Q.stride(2),
            Q.stride(3),
            NUM_HEADS=NUM_HEADS,
            SEQ_LEN=SEQ_LEN,
            HEAD_DIM=HEAD_DIM,
            BLOCK_Q=BLOCK_Q,
            BLOCK_KV=BLOCK_KV,
            CAUSAL=ctx.causal,
            num_warps=4,
            num_stages=3,
        )

        grid_kv = (
            triton.cdiv(SEQ_LEN, BLOCK_KV),
            BATCH_SIZE * NUM_HEADS,
        )

        _attn_bwd_dk_dv[grid_kv](
            Q,
            K,
            V,
            ctx.softmax_scale,
            dO,
            dK,
            dV,
            M,
            D,
            Q.stride(0),
            Q.stride(1),
            Q.stride(2),
            Q.stride(3),
            NUM_HEADS=NUM_HEADS,
            SEQ_LEN=SEQ_LEN,
            HEAD_DIM=HEAD_DIM,
            BLOCK_Q=BLOCK_Q,
            BLOCK_KV=BLOCK_KV,
            CAUSAL=ctx.causal,
            num_warps=4,
            num_stages=3,
        )

        return dQ, dK, dV, None, None


def torch_attention_ref(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    causal: bool,
    softmax_scale: float,
) -> torch.Tensor:
    P = torch.matmul(Q, K.transpose(-2, -1)) * softmax_scale

    if causal:
        seq_len = Q.shape[-2]
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=Q.device, dtype=torch.bool))
        P = P.masked_fill(~causal_mask, float("-inf"))

    P = torch.softmax(P.float(), dim=-1).to(Q.dtype)
    return torch.matmul(P, V)


def test_op(
    BATCH_SIZE: int,
    NUM_HEADS: int,
    SEQ_LEN: int,
    HEAD_DIM: int,
    causal: bool,
    dtype: torch.dtype = torch.float16,
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> None:
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    Q = torch.empty(
        (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM),
        dtype=dtype,
        device="cuda",
    ).normal_(mean=0.0, std=0.5).requires_grad_()

    K = torch.empty_like(Q).normal_(mean=0.0, std=0.5).requires_grad_()
    V = torch.empty_like(Q).normal_(mean=0.0, std=0.5).requires_grad_()

    dO = torch.randn_like(Q)
    softmax_scale = HEAD_DIM ** -0.5

    ref_O = torch_attention_ref(Q, K, V, causal, softmax_scale)
    ref_O.backward(dO)

    ref_dQ = Q.grad.detach().clone()
    ref_dK = K.grad.detach().clone()
    ref_dV = V.grad.detach().clone()

    Q.grad = None
    K.grad = None
    V.grad = None

    tri_O = TritonAttention.apply(Q, K, V, causal, softmax_scale)
    tri_O.backward(dO)

    tri_dQ = Q.grad.detach().clone()
    tri_dK = K.grad.detach().clone()
    tri_dV = V.grad.detach().clone()

    torch.cuda.synchronize()

    print(
        f"test B={BATCH_SIZE}, H={NUM_HEADS}, S={SEQ_LEN}, D={HEAD_DIM}, "
        f"causal={causal}, dtype={dtype}"
    )

    print("O max abs diff: ", (ref_O - tri_O).abs().max().item())
    print("dQ max abs diff:", (ref_dQ - tri_dQ).abs().max().item())
    print("dK max abs diff:", (ref_dK - tri_dK).abs().max().item())
    print("dV max abs diff:", (ref_dV - tri_dV).abs().max().item())

    assert torch.allclose(ref_O, tri_O, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dQ, tri_dQ, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dK, tri_dK, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dV, tri_dV, atol=atol, rtol=rtol)


def benchmark_forward(
    BATCH_SIZE: int,
    NUM_HEADS: int,
    SEQ_LEN: int,
    HEAD_DIM: int,
    causal: bool,
    dtype: torch.dtype = torch.float16,
    warmup: int = 25,
    iters: int = 100,
) -> None:
    Q = torch.randn((BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), device="cuda", dtype=dtype)
    K = torch.randn_like(Q)
    V = torch.randn_like(Q)
    softmax_scale = HEAD_DIM ** -0.5

    with torch.inference_mode():
        for _ in range(warmup):
            _ = TritonAttention.apply(Q, K, V, causal, softmax_scale)

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    with torch.inference_mode():
        start.record()
        for _ in range(iters):
            _ = TritonAttention.apply(Q, K, V, causal, softmax_scale)
        end.record()

    torch.cuda.synchronize()

    print(
        f"forward benchmark B={BATCH_SIZE}, H={NUM_HEADS}, S={SEQ_LEN}, D={HEAD_DIM}, "
        f"causal={causal}, dtype={dtype}: {start.elapsed_time(end) / iters:.3f} ms"
    )


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    # Small tests first.
    test_op(BATCH_SIZE=2, NUM_HEADS=3, SEQ_LEN=343, HEAD_DIM=16, causal=False, dtype=torch.float32)
    test_op(BATCH_SIZE=2, NUM_HEADS=3, SEQ_LEN=343, HEAD_DIM=16, causal=True, dtype=torch.float32)


    # Your actual dimensions.
    test_op(BATCH_SIZE=1176, NUM_HEADS=3, SEQ_LEN=343, HEAD_DIM=16, causal=False, dtype=torch.float32)

    # Optional forward benchmark.
    benchmark_forward(BATCH_SIZE=1176, NUM_HEADS=3, SEQ_LEN=343, HEAD_DIM=16, causal=False, dtype=torch.float32)

    print("PASSED")