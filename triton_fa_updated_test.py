from __future__ import annotations

import triton_shifted_window_fa
from math_utills import *
import torch
from triton_shifted_window_fa import TritonAttention

def torch_attention_ref(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    RPB: torch.Tensor,
    MASK: torch.Tensor,
    softmax_scale: float,
    num_heads: int,
    n: int
) -> torch.Tensor:
    b = Q.shape[0]
    P = torch.matmul(Q, K.transpose(-2, -1)) * softmax_scale + RPB.to(Q.dtype)
    nw = MASK.shape[0]
    P = P.view(b // nw, nw, num_heads, n, n) + MASK.unsqueeze(1).unsqueeze(0)
    P = P.view(-1, num_heads, n, n)


    P = torch.softmax(P.float(), dim=-1).to(Q.dtype)
    return torch.matmul(P, V)


def test_op(
    BATCH_SIZE: int,
    NUM_HEADS: int,
    SEQ_LEN: int,
    HEAD_DIM: int,
    dtype: torch.dtype = torch.float16,
    atol: float = 5e-7,
    rtol: float = 0,
    window_size: int = 7,
) -> None:
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    rpb = generate_rpb(n=SEQ_LEN, num_heads=NUM_HEADS)
    mask = generate_mask(BATCH_SIZE, SEQ_LEN, num_windows=8)
    Q = torch.empty(
        (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM),
        dtype=dtype,
        device="cuda",
    ).normal_(mean=0.0, std=0.5).requires_grad_()

    K = torch.empty_like(Q).normal_(mean=0.0, std=0.5).requires_grad_()
    V = torch.empty_like(Q).normal_(mean=0.0, std=0.5).requires_grad_()

    dO = torch.randn_like(Q)
    softmax_scale = HEAD_DIM ** -0.5

    ref_O = torch_attention_ref(Q, K, V, rpb, mask, softmax_scale, num_heads=NUM_HEADS, n=SEQ_LEN)
    ref_O.backward(dO)

    ref_dQ = Q.grad.detach().clone()
    ref_dK = K.grad.detach().clone()
    ref_dV = V.grad.detach().clone()

    Q.grad = None
    K.grad = None
    V.grad = None

    tri_O = TritonAttention.apply(Q, K, V, rpb, mask, softmax_scale)
    tri_O.backward(dO)

    tri_dQ = Q.grad.detach().clone()
    tri_dK = K.grad.detach().clone()
    tri_dV = V.grad.detach().clone()

    torch.cuda.synchronize()

    print(
        f"test B={BATCH_SIZE}, H={NUM_HEADS}, S={SEQ_LEN}, D={HEAD_DIM}, "
        f"dtype={dtype}"
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
    dtype: torch.dtype = torch.float16,
    warmup: int = 25,
    iters: int = 50,
) -> None:
    Q = torch.randn((BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), device="cuda", dtype=dtype)
    K = torch.randn_like(Q)
    V = torch.randn_like(Q)
    softmax_scale = HEAD_DIM ** -0.5
    rpb = generate_rpb(n=SEQ_LEN, num_heads=NUM_HEADS)
    mask = generate_mask(BATCH_SIZE, SEQ_LEN, num_windows=8)

    with torch.inference_mode():
        for _ in range(warmup):
            _ = TritonAttention.apply(Q, K, V, rpb, mask, softmax_scale)

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    with torch.inference_mode():
        start.record()
        for _ in range(iters):
            _ = TritonAttention.apply(Q, K, V, rpb, mask, softmax_scale)
        end.record()

    torch.cuda.synchronize()
    print(triton_shifted_window_fa._attn_fwd.best_config)
    print(triton_shifted_window_fa._attn_bwd_dq.best_config)
    print(triton_shifted_window_fa._attn_bwd_dk_dv.best_config)
    print(triton_shifted_window_fa._attn_bwd_preprocess.best_config)
    print(
        f"forward benchmark B={BATCH_SIZE}, H={NUM_HEADS}, S={SEQ_LEN}, D={HEAD_DIM}, "
        f"dtype={dtype}: {start.elapsed_time(end) / iters:.3f} ms"
    )

def generate_rpb(n: int, window_size= 7, num_heads = 3) -> torch.Tensor:
    relative_position_bias_table =  torch.zeros(
            (2 * window_size - 1) * (2 * window_size - 1) * (2 * window_size - 1),
            num_heads,
        )
    mesh_args = torch.meshgrid.__kwdefaults__
    coords_d = torch.arange(window_size)
    coords_h = torch.arange(window_size)
    coords_w = torch.arange(window_size)
    if mesh_args is not None:
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij"))
    else:
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))
    coords_flatten = torch.flatten(coords, 1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += window_size - 1
    relative_coords[:, :, 1] += window_size - 1
    relative_coords[:, :, 2] += window_size - 1
    relative_coords[:, :, 0] *= (2 * window_size - 1) * (2 * window_size - 1)
    relative_coords[:, :, 1] *= 2 * window_size - 1


    relative_position_index = relative_coords.sum(-1)
    trunc_normal_(relative_position_bias_table, std=0.02)


    relative_position_bias = relative_position_bias_table[
        relative_position_index.clone()[:n, :n].reshape(-1)
    ].reshape(n, n, -1).permute(2, 0, 1).contiguous()
    return relative_position_bias.clone().to('cuda')

def generate_mask(b: int, n: int, num_windows: int = 8, dtype=torch.float32) -> torch.Tensor:
    shape = (b // num_windows, n, n)
    mask = -100 * torch.randint(low = 0, high= 2, size=shape, dtype= dtype, device= torch.device("cuda"))
    return mask

if __name__ == "__main__":
    # x0 = torch.load('sw_input0.pt')
    # x1 = torch.load('sw_input1.pt')
    # m1 = torch.load('sw_input1_mask.pt')
    # x0 = torch.load('input_tensor.pt')
    # m0 = torch.load('mask_tensor.pt')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    # Small tests first.
    test_op(BATCH_SIZE=1176, NUM_HEADS=3, SEQ_LEN=343, HEAD_DIM=16, dtype=torch.float32)



    # Your actual dimensions.
    test_op(BATCH_SIZE=1176, NUM_HEADS=3, SEQ_LEN=343, HEAD_DIM=16, dtype=torch.float32)
    # Optional forward benchmark.
    benchmark_forward(BATCH_SIZE=1176, NUM_HEADS=3, SEQ_LEN=343, HEAD_DIM=16, dtype=torch.float32)

    print("PASSED")