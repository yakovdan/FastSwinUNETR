from collections import defaultdict
from typing import Any
import torch


def profile_module_forward(
    module: Any,
    x: tuple[torch.Tensor, ...],
    warmup_iters: int = 50,
    profile_iters: int = 100,
) -> dict[str, float]:
    module.eval()
    device = x[0].device
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Warmup.
    with torch.inference_mode():
        for _ in range(warmup_iters):
            _ = module(*x)

    torch.cuda.synchronize(device)

    accumulated_ms: dict[str, float] = defaultdict(float)

    # Profile.
    with torch.inference_mode():
        for _ in range(profile_iters):
            start_event.record()
            _ = module(*x)
            end_event.record()

            torch.cuda.synchronize(device)
            accumulated_ms["forward"] += start_event.elapsed_time(end_event)
            for section_name, elapsed_ms in module.last_section_times_ms.items():
                accumulated_ms[section_name] += elapsed_ms

    return {
        section_name: total_ms / profile_iters
        for section_name, total_ms in accumulated_ms.items() if section_name != "forward"
    }


def profile_module_torch_profiler(
    module: Any,
    inputs: tuple[torch.Tensor, ...],
    *,
    train: bool = False,
    warmup_iters: int = 10,
    profile_iters: int = 8,
    trace_path: str = "trace.json",
    loss_fn=None,
    sort_by: str = "self_cuda_time_total",
    row_limit: int = 30,
    record_shapes: bool = True,
    with_stack: bool = False,
) -> Any:
    """Kernel-level profile of ``module(*inputs)`` via ``torch.profiler``.

    Captures CPU+CUDA activities, attributes kernels to the ``record_function``
    section ranges emitted inside the model, prints an aggregated op table plus
    a shape-grouped table, and writes a Chrome/Perfetto trace to ``trace_path``.

    train=False profiles the inference forward (eval + inference_mode).
    train=True profiles a forward+backward step (train mode, grads enabled, so
    gradient checkpointing actually recomputes in the backward).
    """
    from torch.profiler import profile, ProfilerActivity

    device = inputs[0].device

    if loss_fn is None:
        def loss_fn(out):
            out = out[0] if isinstance(out, (list, tuple)) else out
            return out.float().pow(2).mean()

    if train:
        module.train()
        # Detached leaves so we don't accumulate grad on the caller's tensors;
        # a grad-requiring input is also what makes checkpoint recompute.
        base = [t.detach() if torch.is_tensor(t) else t for t in inputs]

        def step():
            module.zero_grad(set_to_none=True)
            iter_inputs = tuple(
                t.clone().requires_grad_(True)
                if (torch.is_tensor(t) and t.is_floating_point()) else t
                for t in base
            )
            with torch.enable_grad():
                loss_fn(module(*iter_inputs)).backward()
    else:
        module.eval()

        def step():
            with torch.inference_mode():
                module(*inputs)

    # Warmup: lets cudnn.benchmark pick algos and Triton autotune/compile.
    for _ in range(warmup_iters):
        step()
    torch.cuda.synchronize(device)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=record_shapes,
        profile_memory=True,
        with_stack=with_stack,
    ) as prof:
        for _ in range(profile_iters):
            step()
            torch.cuda.synchronize(device)

    prof.export_chrome_trace(trace_path)

    mode = "train (fwd+bwd)" if train else "inference (fwd)"
    print(f"\n=== top ops by {sort_by} -- {mode} ===")
    print(prof.key_averages().table(sort_by=sort_by, row_limit=row_limit))
    if record_shapes:
        print("\n=== top ops grouped by input shape ===")
        print(prof.key_averages(group_by_input_shape=True).table(
            sort_by=sort_by, row_limit=row_limit))
    print(f"\nChrome/Perfetto trace written to {trace_path} "
          f"(open in chrome://tracing or https://ui.perfetto.dev)")
    return prof


def profile_original_window_attention_forward(
    module: Any,
    x: torch.Tensor,
    mask: torch.Tensor | None,
    warmup_iters: int = 25,
    profile_iters: int = 50,
) -> dict[str, float]:
    module.eval()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Warmup.
    with torch.inference_mode():
        for _ in range(warmup_iters):
            _ = module(x, mask)

    torch.cuda.synchronize(x.device)

    accumulated_ms: dict[str, float] = defaultdict(float)

    # Profile.
    with torch.inference_mode():
        for _ in range(profile_iters):
            start_event.record()
            _ = module(x, mask)
            end_event.record()

            torch.cuda.synchronize(x.device)
            accumulated_ms["forward"] += start_event.elapsed_time(end_event)

    return {
        section_name: total_ms / profile_iters
        for section_name, total_ms in accumulated_ms.items()
    }


def profile_original_window_attention_backward(
    module: Any,
    x: torch.Tensor,
    mask: torch.Tensor | None,
    warmup_iters: int = 25,
    profile_iters: int = 50,
) -> dict[str, float]:
    module.eval()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Use a detached leaf tensor so we do not mutate the caller's x.grad.
    x_base = x.detach()

    # Create a stable upstream gradient once.
    with torch.enable_grad():
        x_tmp = x_base.clone().requires_grad_(True)
        out_tmp = module(x_tmp, mask)
        if isinstance(out_tmp, tuple):
            out_tmp = out_tmp[0]
        d_out = torch.randn_like(out_tmp)

    torch.cuda.synchronize(x.device)

    # Warmup backward.
    for _ in range(warmup_iters):
        module.zero_grad(set_to_none=True)

        with torch.enable_grad():
            x_warm = x_base.clone().requires_grad_(True)
            out = module(x_warm, mask)
            if isinstance(out, tuple):
                out = out[0]
            out.backward(d_out)

    torch.cuda.synchronize(x.device)

    accumulated_ms: dict[str, float] = defaultdict(float)

    # Profile backward only.
    for _ in range(profile_iters):
        module.zero_grad(set_to_none=True)

        with torch.enable_grad():
            x_iter = x_base.clone().requires_grad_(True)

            # Required to build the graph, but not included in the timing.
            out = module(x_iter, mask)
            if isinstance(out, tuple):
                out = out[0]
            start_event.record()
            out.backward(d_out)
            end_event.record()

        torch.cuda.synchronize(x.device)
        accumulated_ms["backward"] += start_event.elapsed_time(end_event)

    return {
        section_name: total_ms / profile_iters
        for section_name, total_ms in accumulated_ms.items()
    }

def profile_window_attention_sections(
    module: Any,
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



def profile_window_attention_backward(
    module: Any,
    x: torch.Tensor,
    mask: torch.Tensor | None,
    warmup_iters: int = 25,
    profile_iters: int = 50,
) -> dict[str, float]:
    module.eval()

    # Backward profiling must run with grad enabled.
    module.profile_sections = False

    # Use a detached leaf so we do not mutate the caller's x.grad.
    x_base = x.detach()

    # Get output shape once, so we can create a stable dOut.
    with torch.enable_grad():
        x_tmp = x_base.clone().requires_grad_(True)
        out_tmp = module(x_tmp, mask)
        if isinstance(out_tmp, tuple):
            out_tmp = out_tmp[0]
        d_out = torch.randn_like(out_tmp)

    torch.cuda.synchronize(x.device)

    # Warmup backward.
    for _ in range(warmup_iters):
        module.zero_grad(set_to_none=True)

        with torch.enable_grad():
            x_warm = x_base.clone().requires_grad_(True)
            out = module(x_warm, mask)
            if isinstance(out, tuple):
                out = out[0]

            out.backward(d_out)

    torch.cuda.synchronize(x.device)

    accumulated_ms: dict[str, float] = defaultdict(float)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for _ in range(profile_iters):
        module.zero_grad(set_to_none=True)

        with torch.enable_grad():
            x_iter = x_base.clone().requires_grad_(True)

            # Forward pass is required to build the graph, but is not timed.
            out = module(x_iter, mask)
            out = module(x_warm, mask)
            if isinstance(out, tuple):
                out = out[0]


            start_event.record()
            out.backward(d_out)
            end_event.record()

        torch.cuda.synchronize(x.device)
        accumulated_ms["backward"] += start_event.elapsed_time(end_event)

    module.profile_sections = False

    return {
        section_name: total_ms / profile_iters
        for section_name, total_ms in accumulated_ms.items()
    }



def profile_forward_peak_memory(
    module: torch.nn.Module,
    x: torch.Tensor,
    mask: torch.Tensor | None,
    warmup_iters: int = 10,
    profile_iters: int = 20,
) -> dict[str, float]:
    module.eval()
    device = x.device

    # Warmup.
    with torch.inference_mode():
        for _ in range(warmup_iters):
            _ = module(x, mask)

    torch.cuda.synchronize(device)

    peak_bytes = []

    with torch.inference_mode():
        for _ in range(profile_iters):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)

            _ = module(x, mask)

            torch.cuda.synchronize(device)
            peak_bytes.append(torch.cuda.max_memory_allocated(device))

    avg_peak_bytes = sum(peak_bytes) / len(peak_bytes)

    return {
        "forward_peak_memory_mb": avg_peak_bytes / 1024**2,
    }

def profile_forward_backward_peak_memory(
    module: torch.nn.Module,
    x: torch.Tensor,
    mask: torch.Tensor | None,
    warmup_iters: int = 10,
    profile_iters: int = 20,
) -> dict[str, float]:
    module.eval()
    device = x.device

    x_base = x.detach()

    # Create a stable upstream gradient once.
    with torch.enable_grad():
        x_tmp = x_base.clone().requires_grad_(True)
        out_tmp = module(x_tmp, mask)
        if isinstance(out_tmp, tuple):
            out_tmp = out_tmp[0]
        d_out = torch.randn_like(out_tmp)

    torch.cuda.synchronize(device)

    # Warmup.
    for _ in range(warmup_iters):
        module.zero_grad(set_to_none=True)

        with torch.enable_grad():
            x_warm = x_base.clone().requires_grad_(True)
            out = module(x_warm, mask)
            if isinstance(out, tuple):
                out = out[0]

            out.backward(d_out)

    torch.cuda.synchronize(device)

    peak_bytes = []

    for _ in range(profile_iters):
        module.zero_grad(set_to_none=True)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

        with torch.enable_grad():
            x_iter = x_base.clone().requires_grad_(True)
            out = module(x_iter, mask)
            if isinstance(out, tuple):
                out = out[0]
            out.backward(d_out)

        torch.cuda.synchronize(device)
        peak_bytes.append(torch.cuda.max_memory_allocated(device))

    avg_peak_bytes = sum(peak_bytes) / len(peak_bytes)

    return {
        "forward_backward_peak_memory_mb": avg_peak_bytes / 1024**2,
    }