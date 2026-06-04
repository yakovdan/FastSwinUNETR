from __future__ import annotations
import torch
from torch import nn
from collections import defaultdict
from SwinAttention import FastWindowAttention
from monai.networks.nets.swin_unetr import WindowAttention as OrigWindowAttention
import numpy as np
import random

SEED = 1234

def restore_rng_state() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)



def profile_window_attention_sections(
    module: FastWindowAttention,
    x: torch.Tensor,
    mask: torch.Tensor | None,
    warmup_iters: int = 25,
    profile_iters: int = 50,
) -> dict[str, float]:
    module.eval()

    # Warmup without section profiling.
    module.profile_sections = False

    with torch.inference_mode():
        for _ in range(warmup_iters):
            _ = module(x, mask)

    torch.cuda.synchronize(x.device)

    accumulated_ms: dict[str, float] = defaultdict(float)

    module.profile_sections = True

    with torch.inference_mode():
        for _ in range(profile_iters):
            _ = module(x, mask)

            for section_name, elapsed_ms in module.last_section_times_ms.items():
                accumulated_ms[section_name] += elapsed_ms

    torch.cuda.synchronize(x.device)

    module.profile_sections = False

    return {
        section_name: total_ms / profile_iters
        for section_name, total_ms in accumulated_ms.items()
    }

if __name__ == "__main__":

    # ----------------------------
    # Reproducibility
    # ----------------------------
    dtype = torch.float32
    x = torch.load('input_tensor.pt').to("cuda").to(dtype)
    mask = torch.load('mask_tensor.pt').to("cuda").to(dtype)

    restore_rng_state()
    win0 = FastWindowAttention(48, 3, (7, 7, 7), True, 0.25, 0.25).to("cuda").to(dtype)
    win0.profile_sections = True

    restore_rng_state()
    win1 = OrigWindowAttention(48, 3, (7, 7, 7), True, 0.25, 0.25).to("cuda").to(dtype)

    win0.eval()
    win1.eval()
    # ----------------------------
    # Correctness check
    # ----------------------------
    win0.profile_sections = False


    restore_rng_state()
    ref_output = win1(x, mask)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    restore_rng_state()
    test_output = win0(x, mask)

    O_GT = ref_output.reshape((1176, 343, 3, 16)).permute(0, 2, 1, 3)
    dO = torch.randn_like(win1.ref_q)

    O_GT.backward(dO)

    ref_dQ = win1.ref_q.grad.detach().clone()
    ref_dK = win1.ref_k.grad.detach().clone()
    ref_dV = win1.ref_v.grad.detach().clone()

    win1.ref_q.grad = None
    win1.ref_k.grad = None
    win1.ref_v.grad = None

    O, Q, K, V, RPB = test_output
    O.backward(dO)

    tri_dQ = Q.grad.detach().clone()
    tri_dK = K.grad.detach().clone()
    tri_dV = V.grad.detach().clone()

    torch.cuda.synchronize()


    print(torch.allclose(ref_output, test_output[0], atol=1e-5, rtol=0))

    section_times = profile_window_attention_sections(
        win0,
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
