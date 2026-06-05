from __future__ import annotations
from collections.abc import Sequence
import time
import torch
from torch import nn
import math
from triton_fa_updated_test import TritonAttention


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    """Tensor initialization with truncated normal distribution.
    Based on:
    https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    https://github.com/rwightman/pytorch-image-models

    Args:
       tensor: an n-dimensional `torch.Tensor`.
       mean: the mean of the normal distribution.
       std: the standard deviation of the normal distribution.
       a: the minimum cutoff value.
       b: the maximum cutoff value.
    """

    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    """Tensor initialization with truncated normal distribution.
    Based on:
    https://github.com/rwightman/pytorch-image-models

    Args:
       tensor: an n-dimensional `torch.Tensor`
       mean: the mean of the normal distribution
       std: the standard deviation of the normal distribution
       a: the minimum cutoff value
       b: the maximum cutoff value
    """

    if std <= 0:
        raise ValueError("the standard deviation should be greater than zero.")

    if a >= b:
        raise ValueError("minimum cutoff value (a) should be smaller than maximum cutoff value (b).")

    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

def make_event_pair():
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    return start, end


class FastWindowAttention(nn.Module):
    """
    Window based multi-head self attention module with relative position bias based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Sequence[int],
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            num_heads: number of attention heads.
            window_size: local window size.
            qkv_bias: add a learnable bias to query, key, value.
            attn_drop: attention dropout rate.
            proj_drop: dropout rate of output.
        """

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        mesh_args = torch.meshgrid.__kwdefaults__

        if len(self.window_size) == 3:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(
                    (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1),
                    num_heads,
                )
            )
            coords_d = torch.arange(self.window_size[0])
            coords_h = torch.arange(self.window_size[1])
            coords_w = torch.arange(self.window_size[2])
            if mesh_args is not None:
                coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij"))
            else:
                coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 2] += self.window_size[2] - 1
            relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
            relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        elif len(self.window_size) == 2:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
            )
            coords_h = torch.arange(self.window_size[0])
            coords_w = torch.arange(self.window_size[1])
            if mesh_args is not None:
                coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
            else:
                coords = torch.stack(torch.meshgrid(coords_h, coords_w))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        else:
            raise RuntimeError(f"Invalid window_size dimensions: {len(self.window_size)}.")
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)
        self.profile_sections = False
        self.last_section_times_ms: dict[str, float] = {}

    def section_profile_lines(self) -> list[str]:
        total_ms = sum(self.last_section_times_ms.values())
        lines: list[str] = []
        for section_name, elapsed_ms in self.last_section_times_ms.items():
            percentage = (elapsed_ms / total_ms * 100.0) if total_ms > 0.0 else 0.0
            lines.append(f"{section_name}: {elapsed_ms:.3f} ms ({percentage:.1f}%)")

        if self.last_section_times_ms:
            slowest_section_name, slowest_elapsed_ms = max(
                self.last_section_times_ms.items(),
                key=lambda section_time: section_time[1],
            )
            slowest_percentage = (slowest_elapsed_ms / total_ms * 100.0) if total_ms > 0.0 else 0.0
            lines.append(
                "Summary: "
                f"total={total_ms:.3f} ms, "
                f"sections={len(self.last_section_times_ms)}, "
                f"slowest={slowest_section_name} "
                f"({slowest_elapsed_ms:.3f} ms, {slowest_percentage:.1f}%)"
            )

        return lines



    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b, n, c = x.shape

        profile_sections = self.profile_sections
        self.last_section_times_ms = {}

        section_events: dict[str, tuple[torch.cuda.Event, torch.cuda.Event]] = {}

        def record_section(section_name: str, fn):
            if not profile_sections:
                return fn()

            if not x.is_cuda:
                section_start = time.perf_counter()
                result = fn()
                self.last_section_times_ms[section_name] = (
                                                                   time.perf_counter() - section_start
                                                           ) * 1000.0
                return result

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            result = fn()
            end.record()

            section_events[section_name] = (start, end)
            return result

        # Section 1, projection
        q, k, v = record_section(
            "Section 1, projection",
            lambda: self.qkv(x)
            .reshape(b, n, 3, self.num_heads, c // self.num_heads)
            .permute(2, 0, 3, 1, 4)
            .unbind(0),
        )

        # Section 2, scale and Q@K.T
        # attn = record_section(
        #     "Section 2, scale and Q@K.T",
        #     lambda: (q * self.scale) @ k.transpose(-2, -1),
        # )

        # Section 3, relative position bias
        relative_position_bias = record_section(
            "Section 3, relative position bias",
            lambda: self.relative_position_bias_table[
                self.relative_position_index.clone()[:n, :n].reshape(-1)
            ]
            .reshape(n, n, -1)
            .permute(2, 0, 1)
            .contiguous().to(dtype=q.dtype)
        )
        if mask is not None:
            mask = mask.contiguous().to(dtype=q.dtype)

        attn = TritonAttention.apply(q.contiguous(), k.contiguous(), v.contiguous(), relative_position_bias, mask, self.scale).transpose(1, 2).reshape(b, n, c)
        # # Section 4, apply RPB to attention
        # attn = record_section(
        #     "Section 4, apply RPB to attention",
        #     lambda: attn.add_(relative_position_bias.unsqueeze(0))
        # )
        #
        #
        # # Section 5, add mask
        # def add_mask():
        #     if mask is None:
        #         return attn
        #
        #
        #     nw = mask.shape[0]
        #     masked_attn = attn.view(
        #         b // nw, nw, self.num_heads, n, n
        #     ) + mask.unsqueeze(1).unsqueeze(0)
        #
        #     return masked_attn.view(-1, self.num_heads, n, n)
        #
        # attn = record_section("Section 5, add mask", add_mask)
        #
        # # Section 6, apply softmax
        # attn = record_section(
        #     "Section 6, apply softmax",
        #     lambda: self.softmax(attn),
        # )

        # Section 7, apply dropout and cast
        # attn = record_section(
        #     "Section 7, apply dropout and cast",
        #     lambda: self.attn_drop(attn).to(v.dtype),
        # )
        #
        # # Section 8, matmul by V
        # x = record_section(
        #     "Section 8, matmul by V",
        #     lambda: (attn @ v).transpose(1, 2).reshape(b, n, c),
        # )

        # Section 9, projection and dropout
        x = record_section(
            "Section 9, projection and dropout",
            lambda: self.proj_drop(self.proj(attn)),
        )

        if profile_sections and x.is_cuda:
            torch.cuda.synchronize(x.device)
            self.last_section_times_ms = {
                section_name: start.elapsed_time(end)
                for section_name, (start, end) in section_events.items()
            }

        return x, q, k, v, relative_position_bias






