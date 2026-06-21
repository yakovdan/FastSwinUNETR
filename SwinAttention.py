from __future__ import annotations
from collections.abc import Sequence

from torch import nn
from math_utills import *

from triton_shifted_window_fa import TritonAttention




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
    shapes_seen = set()
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Sequence[int],
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        debug_mode = False,
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
        self.debug_mode = debug_mode
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
        q, k, v = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)

        relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.clone()[:n, :n].reshape(-1)
            ].reshape(n, n, -1).permute(2, 0, 1).contiguous().to(dtype=q.dtype)

        if mask is not None:
            mask = mask.contiguous().to(dtype=q.dtype)
        FastWindowAttention.shapes_seen.add(tuple([self.num_heads] + list(x.shape)))
        attn = TritonAttention.apply(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            relative_position_bias,
            mask,
            self.scale,
        ).transpose(1, 2).reshape(b, n, c)

        x = self.proj_drop(self.proj(attn))

        if self.debug_mode:
            return x, q, k, v, relative_position_bias
        return x






