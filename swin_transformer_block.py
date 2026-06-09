import time
from typing import Sequence, TypeVar, Union
import os
os.environ["TRITON_CACHE_DIR"] = "./triton_cache"
import numpy as np
from torch import nn
from torch.nn import LayerNorm
from torch.utils import checkpoint

from profiling import profile_module_forward
from swin_unter_utils import ensure_tuple_rep, LayerFactory, look_up_option, split_args, get_act_layer, get_window_size, window_partition, window_reverse, compute_mask
from torch.nn import functional as F
import torch
from fast_swin_attention import FastWindowAttention as WindowAttention
from fast_layer_norm import FastLayerNorm
from fast_mlp import FastMLP as MLPBlock
#from mlp_block import MLPBlock

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")


class InitArgsMixin:
    def save_init_args(self, path: str) -> None:
        """Save constructor arguments so an equivalent object can be recreated."""
        torch.save(
            {
                "class_name": self.__class__.__name__,
                "init_args": self.init_args,
            },
            path,
        )

    @classmethod
    def from_init_args(cls, path: str):
        data = torch.load(path, weights_only=False)
        class_name = data.get("class_name")
        if class_name != cls.__name__:
            raise ValueError(f"Saved init args are for {class_name}, not {cls.__name__}.")
        return cls(**data["init_args"])


class DropPath(nn.Module):
    """Stochastic drop paths per sample for residual blocks.
    Based on:
    https://github.com/rwightman/pytorch-image-models
    """

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True) -> None:
        """
        Args:
            drop_prob: drop path probability.
            scale_by_keep: scaling by non-dropped probability.
        """
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

        if not (0 <= drop_prob <= 1):
            raise ValueError("Drop path prob should be between 0 and 1.")

    def drop_path(self, x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
        if drop_prob == 0.0 or not training:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor

    def forward(self, x):
        return self.drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

class SwinTransformerBlock3D(InitArgsMixin, nn.Module):
    """
    Swin Transformer block based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Sequence[int],
        shift_size: Sequence[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: str = "GELU",
        norm_layer: type[LayerNorm] = nn.LayerNorm,
        use_checkpoint: bool = False,
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            num_heads: number of attention heads.
            window_size: local window size.
            shift_size: window shift size.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            drop_path: stochastic depth rate.
            act_layer: activation layer.
            norm_layer: normalization layer.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
        """

        super().__init__()
        self.save_inputs = False
        self.init_args = {
            "dim": dim,
            "num_heads": num_heads,
            "window_size": window_size,
            "shift_size": shift_size,
            "mlp_ratio": mlp_ratio,
            "qkv_bias": qkv_bias,
            "drop": drop,
            "attn_drop": attn_drop,
            "drop_path": drop_path,
            "act_layer": act_layer,
            "norm_layer": norm_layer,
            "use_checkpoint": use_checkpoint,
        }
        #print(f"{self.__class__.__name__}.__init__ args: {self.init_args}")
        if drop_path != 0:
            raise ValueError("Drop path is not supported in Swin Transformer blocks. No stochastic depth for now.")
        self.last_section_times_ms = None
        self.profile_sections = True
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        if self.dim == 48 or self.dim == 96:
            self.norm1 = FastLayerNorm(dim)
            self.norm2 = FastLayerNorm(dim)
        else:
            self.norm1 = norm_layer(dim)
            self.norm2 = norm_layer(dim)

        self.attn = WindowAttention(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            use_tf32=True
        )

        self.drop_path = nn.Identity()

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLPBlock(hidden_size=dim, mlp_dim=mlp_hidden_dim, act=act_layer, dropout_rate=drop, dropout_mode="swin", precision="tf32")

    def forward_part1(self, x, mask_matrix):
        profile_sections = self.profile_sections
        if self.last_section_times_ms is None:
            self.last_section_times_ms = {}

        section_events: dict[str, tuple[torch.cuda.Event, torch.cuda.Event]] = {}

        def record_section(section_name: str, fn):
            section_name = f"fp1_{section_name}"
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

        def get_shape():
            return x.size()

        def apply_norm1():
            #print(f"Norm 1 shape: {x.shape}, {self.dim}, M = {x_shape[0] * x_shape[1] * x_shape[2] * x_shape[3]}")
            return self.norm1(x)

        def compute_window_and_padding():
            b, d, h, w, c = x.shape
            window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
            pad_l = pad_t = pad_d0 = 0
            pad_d1 = (window_size[0] - d % window_size[0]) % window_size[0]
            pad_b = (window_size[1] - h % window_size[1]) % window_size[1]
            pad_r = (window_size[2] - w % window_size[2]) % window_size[2]
            return b, d, h, w, c, window_size, shift_size, pad_l, pad_t, pad_d0, pad_d1, pad_b, pad_r

        def apply_padding():
            padded_x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
            _, dp, hp, wp, _ = padded_x.shape
            return padded_x, [b, dp, hp, wp]

        def apply_shift():
            if any(i > 0 for i in shift_size):
                return torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3)), mask_matrix
            return x, None

        def partition_windows():
            return window_partition(shifted_x, window_size)

        def apply_attention():
            return self.attn(x_windows, mask=attn_mask)

        def reshape_attention_windows():
            return attn_windows.view(-1, *(window_size + (c,)))

        def reverse_windows():
            return window_reverse(attn_windows, window_size, dims)

        def reverse_shift():
            if any(i > 0 for i in shift_size):
                return torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
            return shifted_x

        def remove_padding():
            if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
                return x[:, :d, :h, :w, :].contiguous()
            return x

        record_section("shape", get_shape)
        x = record_section("norm1", apply_norm1)
        (
            b,
            d,
            h,
            w,
            c,
            window_size,
            shift_size,
            pad_l,
            pad_t,
            pad_d0,
            pad_d1,
            pad_b,
            pad_r,
        ) = record_section("window_and_padding", compute_window_and_padding)
        x, dims = record_section("pad", apply_padding)
        shifted_x, attn_mask = record_section("shift", apply_shift)
        x_windows = record_section("window_partition", partition_windows)
        attn_windows = record_section("attention", apply_attention)
        attn_windows = record_section("attention_view", reshape_attention_windows)
        shifted_x = record_section("window_reverse", reverse_windows)
        x = record_section("reverse_shift", reverse_shift)
        x = record_section("remove_padding", remove_padding)

        if profile_sections and x.is_cuda:
            torch.cuda.synchronize()
            for section_name, (start, end) in section_events.items():
                self.last_section_times_ms[section_name] = start.elapsed_time(end)

        return x

    def forward_part2(self, x):
        x_shape = x.shape
        #print(f"YAKOV INPUT SHAPE: {x_shape}")
        #print(f"Norm 2 shape: {x.shape}, {self.dim}, M = {x_shape[0] * x_shape[1] * x_shape[2] * x_shape[3]}")
        return self.mlp(self.norm2(x))

    def load_from(self, weights, n_block, layer):
        root = f"module.{layer}.0.blocks.{n_block}."
        block_names = [
            "norm1.weight",
            "norm1.bias",
            "attn.relative_position_bias_table",
            "attn.relative_position_index",
            "attn.qkv.weight",
            "attn.qkv.bias",
            "attn.proj.weight",
            "attn.proj.bias",
            "norm2.weight",
            "norm2.bias",
            "mlp.fc1.weight",
            "mlp.fc1.bias",
            "mlp.fc2.weight",
            "mlp.fc2.bias",
        ]
        with torch.no_grad():
            self.norm1.weight.copy_(weights["state_dict"][root + block_names[0]])
            self.norm1.bias.copy_(weights["state_dict"][root + block_names[1]])
            self.attn.relative_position_bias_table.copy_(weights["state_dict"][root + block_names[2]])
            self.attn.relative_position_index.copy_(weights["state_dict"][root + block_names[3]])  # type: ignore[operator]
            self.attn.qkv.weight.copy_(weights["state_dict"][root + block_names[4]])
            self.attn.qkv.bias.copy_(weights["state_dict"][root + block_names[5]])
            self.attn.proj.weight.copy_(weights["state_dict"][root + block_names[6]])
            self.attn.proj.bias.copy_(weights["state_dict"][root + block_names[7]])
            self.norm2.weight.copy_(weights["state_dict"][root + block_names[8]])
            self.norm2.bias.copy_(weights["state_dict"][root + block_names[9]])
            self.mlp.linear1.weight.copy_(weights["state_dict"][root + block_names[10]])
            self.mlp.linear1.bias.copy_(weights["state_dict"][root + block_names[11]])
            self.mlp.linear2.weight.copy_(weights["state_dict"][root + block_names[12]])
            self.mlp.linear2.bias.copy_(weights["state_dict"][root + block_names[13]])


    def forward(self, x, mask_matrix):
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

        def save_inputs_if_needed():
            if not self.save_inputs:
                self.save_inputs = True
                from pathlib import Path
                fname = Path(f"input_STB_0_{self.dim}.pth")
                if not (fname.exists() and fname.is_file()):
                    torch.save(x, fname)
                    torch.save(mask_matrix, f"mask_STB_0_{self.dim}.pth")
                else:
                    fname = Path(f"input_STB_1_{self.dim}.pth")
                    torch.save(x, fname)
                    torch.save(mask_matrix, f"mask_STB_1_{self.dim}.pth")

        def run_forward_part1():
            if self.use_checkpoint:
                return checkpoint.checkpoint(self.forward_part1, x, mask_matrix, use_reentrant=False)
            return self.forward_part1(x, mask_matrix)

        def run_forward_part2():
            if self.use_checkpoint:
                return checkpoint.checkpoint(self.forward_part2, x, use_reentrant=False)
            return self.forward_part2(x)

        #record_section("save_inputs", save_inputs_if_needed)
        shortcut = record_section("shortcut", lambda: x)
        x = run_forward_part1()
        #x = record_section("forward_part1", run_forward_part1)
        x = record_section("residual_part1", lambda: shortcut + x)
        #part2 = record_section("forward_part2", run_forward_part2)
        x = record_section("forward_part2", run_forward_part2)
        #x = record_section("residual_part2", lambda: x + part2)

        if profile_sections and x.is_cuda:
            torch.cuda.synchronize()
            for section_name, (start, end) in section_events.items():
                self.last_section_times_ms[section_name] = start.elapsed_time(end)
        return x

class SwinTransformerBlock2D(InitArgsMixin, nn.Module):
    """
    Swin Transformer block based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Sequence[int],
        shift_size: Sequence[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: str = "GELU",
        norm_layer: type[LayerNorm] = nn.LayerNorm,
        use_checkpoint: bool = False,
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            num_heads: number of attention heads.
            window_size: local window size.
            shift_size: window shift size.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            drop_path: stochastic depth rate.
            act_layer: activation layer.
            norm_layer: normalization layer.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
        """

        super().__init__()
        self.init_args = {
            "dim": dim,
            "num_heads": num_heads,
            "window_size": window_size,
            "shift_size": shift_size,
            "mlp_ratio": mlp_ratio,
            "qkv_bias": qkv_bias,
            "drop": drop,
            "attn_drop": attn_drop,
            "drop_path": drop_path,
            "act_layer": act_layer,
            "norm_layer": norm_layer,
            "use_checkpoint": use_checkpoint,
        }
        print(f"{self.__class__.__name__}.__init__ args: {self.init_args}")
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        if self.dim == 48 or self.dim == 96:
            self.norm1 = FastLayerNorm(dim)
            self.norm2 = FastLayerNorm(dim)
        else:
            self.norm1 = norm_layer(dim)
            self.norm2 = norm_layer(dim)

        self.attn = WindowAttention(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            use_tf32=True,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLPBlock(hidden_size=dim, mlp_dim=mlp_hidden_dim, act=act_layer, dropout_rate=drop, dropout_mode="swin", precision="tf32")

    def forward_part1(self, x, mask_matrix):

        x_shape = x.size()
        #print(f"Norm 1 shape: {x.shape}, {self.dim}, M = {x_shape[0] * x_shape[1] * x_shape[2] * x_shape[3]}")
        x = self.norm1(x)

        b, h, w, c = x.shape
        window_size, shift_size = get_window_size((h, w), self.window_size, self.shift_size)
        pad_l = pad_t = 0
        pad_b = (window_size[0] - h % window_size[0]) % window_size[0]
        pad_r = (window_size[1] - w % window_size[1]) % window_size[1]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, hp, wp, _ = x.shape
        dims = [b, hp, wp]

        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        x_windows = window_partition(shifted_x, window_size)

        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, *(window_size + (c,)))
        shifted_x = window_reverse(attn_windows, window_size, dims)

        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))

        if pad_r > 0 or pad_b > 0:
            x = x[:, :h, :w, :].contiguous()

        return x

    def forward_part2(self, x):
        x_shape = x.shape
        #print(f"Norm 2 shape: {x.shape}, {self.dim}, M = {x_shape[0] * x_shape[1] * x_shape[2] * x_shape[3]}")
        return self.drop_path(self.mlp(self.norm2(x)))

    def load_from(self, weights, n_block, layer):
        root = f"module.{layer}.0.blocks.{n_block}."
        block_names = [
            "norm1.weight",
            "norm1.bias",
            "attn.relative_position_bias_table",
            "attn.relative_position_index",
            "attn.qkv.weight",
            "attn.qkv.bias",
            "attn.proj.weight",
            "attn.proj.bias",
            "norm2.weight",
            "norm2.bias",
            "mlp.fc1.weight",
            "mlp.fc1.bias",
            "mlp.fc2.weight",
            "mlp.fc2.bias",
        ]
        with torch.no_grad():
            self.norm1.weight.copy_(weights["state_dict"][root + block_names[0]])
            self.norm1.bias.copy_(weights["state_dict"][root + block_names[1]])
            self.attn.relative_position_bias_table.copy_(weights["state_dict"][root + block_names[2]])
            self.attn.relative_position_index.copy_(weights["state_dict"][root + block_names[3]])  # type: ignore[operator]
            self.attn.qkv.weight.copy_(weights["state_dict"][root + block_names[4]])
            self.attn.qkv.bias.copy_(weights["state_dict"][root + block_names[5]])
            self.attn.proj.weight.copy_(weights["state_dict"][root + block_names[6]])
            self.attn.proj.bias.copy_(weights["state_dict"][root + block_names[7]])
            self.norm2.weight.copy_(weights["state_dict"][root + block_names[8]])
            self.norm2.bias.copy_(weights["state_dict"][root + block_names[9]])
            self.mlp.linear1.weight.copy_(weights["state_dict"][root + block_names[10]])
            self.mlp.linear1.bias.copy_(weights["state_dict"][root + block_names[11]])
            self.mlp.linear2.weight.copy_(weights["state_dict"][root + block_names[12]])
            self.mlp.linear2.bias.copy_(weights["state_dict"][root + block_names[13]])

    def forward(self, x, mask_matrix):
        shortcut = x
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix, use_reentrant=False)
        else:
            x = self.forward_part1(x, mask_matrix)
        x = shortcut + self.drop_path(x)
        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x, use_reentrant=False)
        else:
            x = x + self.forward_part2(x)
        return x



if __name__ == "__main__":
    from pathlib import Path
    files = list(Path().glob("SwinTransformerBlock_*.pt"))
    files = sorted(files, key=lambda f: int(f.stem.split("_")[-1]))
    for f in files:
        print(f)
        count = int(f.stem.split("_")[1])
        dim=int(int(f.stem.split("_")[2]))
        x = torch.load(f"input_STB_{count}_{dim}.pth", weights_only=False).to("cuda")
        mask = torch.load(f"mask_STB_{count}_{dim}.pth", weights_only=False).to("cuda")
        stb = SwinTransformerBlock3D.from_init_args(str(f)).to("cuda")

        section_times = profile_module_forward(stb, (x, mask))

        total_ms = sum(section_times.values())

        for section_name, elapsed_ms in section_times.items():
            percentage = elapsed_ms / total_ms * 100.0
            print(f"{section_name}: {elapsed_ms:.3f} ms ({percentage:.1f}%)")

        print(f"Total: {total_ms:.3f} ms")
        #stb(x, mask)


    # model = SwinTransformerBlock3D(
    #     in_channels=1,
    #     patch_size=2,
    #     out_channels=1,
    #     drop_rate=0.25,
    #     attn_drop_rate=0.25,
    #     feature_size=48,
    #     use_checkpoint=True,  # gradient checkpointing for reduced memory use at the cost of compute
    # ).to(torch.float32).to("cuda")
    #
    # section_times = profile_module_forward(model, x)
    #
    #
    # total_ms = sum(section_times.values())
    #
    # for section_name, elapsed_ms in section_times.items():
    #     percentage = elapsed_ms / total_ms * 100.0
    #     print(f"{section_name}: {elapsed_ms:.3f} ms ({percentage:.1f}%)")
    #
    # print(f"Total: {total_ms:.3f} ms")
