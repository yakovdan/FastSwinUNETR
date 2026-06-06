from __future__ import annotations
from SwinAttention import FastWindowAttention
from monai.networks.nets.swin_unetr import WindowAttention as OrigWindowAttention
import numpy as np
import random
from profiling import *
SEED = 1234

def restore_rng_state() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

def abs_error(ref: torch.Tensor, test: torch.Tensor) -> torch.Tensor:
    return (ref - test).abs().max().detach()

def rel_error(ref: torch.Tensor, test: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    diff_tensor = torch.abs(ref - test) + eps
    return (diff_tensor / (ref.abs() + eps)).max().detach()





if __name__ == "__main__":

    # ----------------------------
    # Reproducibility
    # ----------------------------
    dtype = torch.float32
    x = torch.load('input_tensor.pt').to("cuda").to(dtype)
    mask = torch.load('mask_tensor.pt').to("cuda").to(dtype)

    restore_rng_state()
    fast_attn = FastWindowAttention(48, 3, (7, 7, 7), True, 0.25, 0.25).to("cuda").to(dtype)
    fast_attn.profile_sections = True

    restore_rng_state()
    orig_attn = OrigWindowAttention(48, 3, (7, 7, 7), True, 0.25, 0.25).to("cuda").to(dtype)

    fast_attn.eval()
    orig_attn.eval()
    # ----------------------------
    # Correctness check
    # ----------------------------
    fast_attn.profile_sections = False


    restore_rng_state()
    ref_output = orig_attn(x, mask)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    restore_rng_state()
    test_output = fast_attn(x, mask)

    ref_O = ref_output
    dO = torch.randn_like(ref_O)

    ref_O.backward(dO)

    ref_dQ = orig_attn.ref_q.grad.detach().clone()
    ref_dK = orig_attn.ref_k.grad.detach().clone()
    ref_dV = orig_attn.ref_v.grad.detach().clone()
    ref_rpb = orig_attn.ref_rpb.grad.detach().clone()

    refs = [ref_O, ref_dQ, ref_dK, ref_dV, ref_rpb]

    orig_attn.ref_q.grad = None
    orig_attn.ref_k.grad = None
    orig_attn.ref_v.grad = None
    orig_attn.ref_rpb.grad = None

    tri_O, Q, K, V, RPB = test_output
    Q.retain_grad(), K.retain_grad(), V.retain_grad(), RPB.retain_grad()
    tri_O.backward(dO)

    tri_dQ = Q.grad.detach().clone()
    tri_dK = K.grad.detach().clone()
    tri_dV = V.grad.detach().clone()
    tri_dRPB = RPB.grad.detach().clone()
    tris = [tri_O, tri_dQ, tri_dK, tri_dV, tri_dRPB]
    torch.cuda.synchronize()

    for r, t in zip(refs, tris):
        print(abs_error(r, t))
        print(rel_error(r, t))


    section_times = profile_window_attention_sections(
        fast_attn,
        x,
        mask,
        warmup_iters=25,
        profile_iters=50,
    )

    total_ms = sum(section_times.values())

    for section_name, elapsed_ms in section_times.items():
        percentage = elapsed_ms / total_ms * 100.0
        print(f"{section_name}: {elapsed_ms:.3f} ms ({percentage:.1f}%)")

    print(f"Total: {total_ms:.3f} ms")

    section_times = profile_window_attention_backward(
        fast_attn,
        x,
        mask,
        warmup_iters=25,
        profile_iters=50,
    )

    total_ms = sum(section_times.values())

    for section_name, elapsed_ms in section_times.items():
        percentage = elapsed_ms / total_ms * 100.0
        print(f"{section_name}: {elapsed_ms:.3f} ms ({percentage:.1f}%)")

    print(f"Total backward: {total_ms:.3f} ms")


    section_times = profile_original_window_attention_forward(
        orig_attn,
        x,
        mask,
        warmup_iters=25,
        profile_iters=50,
    )


    print(f"Total: {section_times}")

    section_times = profile_original_window_attention_backward(
        orig_attn,
        x,
        mask,
        warmup_iters=25,
        profile_iters=50,
    )

    print(f"Total: {section_times}")

    orig_fwd_mem = profile_forward_peak_memory(orig_attn, x, mask)
    fast_fwd_mem = profile_forward_peak_memory(fast_attn, x, mask)

    orig_train_mem = profile_forward_backward_peak_memory(orig_attn, x, mask)
    fast_train_mem = profile_forward_backward_peak_memory(fast_attn, x, mask)

    print("Original forward:", orig_fwd_mem)
    print("Fast forward:", fast_fwd_mem)
    print("Original fwd+bwd:", orig_train_mem)
    print("Fast fwd+bwd:", fast_train_mem)