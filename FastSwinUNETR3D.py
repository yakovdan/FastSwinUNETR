import time
from typing import Sequence, TypeVar, Union
import os

from swin_transformer_3d import SwinTransformer3D

os.environ["TRITON_CACHE_DIR"] = "./triton_cache"
import numpy as np
from torch import nn
from torch.nn import LayerNorm
from torch.utils import checkpoint

from profiling import profile_module_forward, profile_module_torch_profiler
from torch.profiler import record_function
from swin_unter_utils import ensure_tuple_rep, LayerFactory, look_up_option, split_args, get_act_layer, get_window_size, window_partition, window_reverse, compute_mask
from torch.nn import functional as F
import torch
from patch_merging import PatchMerging, PatchMergingV2
from einops import rearrange


from unter import UnetrBasicBlock, UnetrUpBlock, UnetOutBlock

from swin_transformer_block import SwinTransformerBlock3D as SwinTransformerBlock
Conv = LayerFactory(name="Convolution layers", description="Factory for creating convolution layers.")

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
@Conv.factory_function("conv")
def conv_factory(dim: int) -> type[nn.Conv1d | nn.Conv2d | nn.Conv3d]:
    """
    Convolutional layers in 1,2,3 dimensions.

    Args:
        dim: desired dimension of the convolutional layer

    Returns:
        Conv[dim]d
    """
    types = (nn.Conv1d, nn.Conv2d, nn.Conv3d)
    return types[dim - 1]


@Conv.factory_function("convtrans")
def convtrans_factory(dim: int) -> type[nn.ConvTranspose1d | nn.ConvTranspose2d | nn.ConvTranspose3d]:
    """
    Transposed convolutional layers in 1,2,3 dimensions.

    Args:
        dim: desired dimension of the transposed convolutional layer

    Returns:
        ConvTranspose[dim]d
    """
    types = (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)
    return types[dim - 1]

MERGING_MODE = {"merging": PatchMerging, "mergingv2": PatchMergingV2}

class SwinUNETR(nn.Module):
    """
    Swin UNETR based on: "Hatamizadeh et al.,
    Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images
    <https://arxiv.org/abs/2201.01266>"
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int = 2,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        window_size: Sequence[int] | int = 7,
        qkv_bias: bool = True,
        mlp_ratio: float = 4.0,
        feature_size: int = 24,
        norm_name: tuple | str = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        norm_layer: type[LayerNorm] = nn.LayerNorm,
        patch_norm: bool = False,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        downsample: str | nn.Module = "merging",
        use_v2: bool = False,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            patch_size: size of the patch token.
            feature_size: dimension of network feature size.
            depths: number of layers in each stage.
            num_heads: number of attention heads.
            window_size: local window size.
            qkv_bias: add a learnable bias to query, key, value.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            norm_name: feature normalization type and arguments.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            dropout_path_rate: drop path rate.
            normalize: normalize output intermediate features in each stage.
            norm_layer: normalization layer.
            patch_norm: whether to apply normalization to the patch embedding. Default is False.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
            spatial_dims: number of spatial dims.
            downsample: module used for downsampling, available options are `"mergingv2"`, `"merging"` and a
                user-specified `nn.Module` following the API defined in :py:class:`monai.networks.nets.PatchMerging`.
                The default is currently `"merging"` (the original version defined in v0.9.0).
            use_v2: using swinunetr_v2, which adds a residual convolution block at the beggining of each swin stage.

        Examples::

            # for 3D single channel input with size (96,96,96), 4-channel output and feature size of 48.
            >>> net = SwinUNETR(in_channels=1, out_channels=4, feature_size=48)

            # for 3D 4-channel input with size (128,128,128), 3-channel output and (2,4,2,2) layers in each stage.
            >>> net = SwinUNETR(in_channels=4, out_channels=3, depths=(2,4,2,2))

            # for 2D single channel input with size (96,96), 2-channel output and gradient checkpointing.
            >>> net = SwinUNETR(in_channels=3, out_channels=2, use_checkpoint=True, spatial_dims=2)

        """

        super().__init__()
        self.last_section_times_ms = None
        self.profile_sections = True 
        
        if spatial_dims not in (2, 3):
            raise ValueError("spatial dimension should be 2 or 3.")

        self.patch_size = patch_size

        patch_sizes = ensure_tuple_rep(self.patch_size, spatial_dims)
        window_size = ensure_tuple_rep(window_size, spatial_dims)

        if not (0 <= drop_rate <= 1):
            raise ValueError("dropout rate should be between 0 and 1.")

        if not (0 <= attn_drop_rate <= 1):
            raise ValueError("attention dropout rate should be between 0 and 1.")

        if not (0 <= dropout_path_rate <= 1):
            raise ValueError("drop path rate should be between 0 and 1.")

        if feature_size % 12 != 0:
            raise ValueError("feature_size should be divisible by 12.")

        self.normalize = normalize

        swinViT = SwinTransformer3D(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=window_size,
            patch_size=patch_sizes,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dropout_path_rate,
            norm_layer=norm_layer,
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
            downsample=look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample,
            use_v2=use_v2,
        )
        self.swinViT = swinViT
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder10 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=16 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)

    def load_from(self, weights):
        layers1_0: BasicLayer = self.swinViT.layers1[0]  # type: ignore[assignment]
        layers2_0: BasicLayer = self.swinViT.layers2[0]  # type: ignore[assignment]
        layers3_0: BasicLayer = self.swinViT.layers3[0]  # type: ignore[assignment]
        layers4_0: BasicLayer = self.swinViT.layers4[0]  # type: ignore[assignment]
        wstate = weights["state_dict"]

        with torch.no_grad():
            self.swinViT.patch_embed.proj.weight.copy_(wstate["module.patch_embed.proj.weight"])
            self.swinViT.patch_embed.proj.bias.copy_(wstate["module.patch_embed.proj.bias"])
            for bname, block in layers1_0.blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers1")  # type: ignore[operator]

            if layers1_0.downsample is not None:
                d = layers1_0.downsample
                d.reduction.weight.copy_(wstate["module.layers1.0.downsample.reduction.weight"])  # type: ignore
                d.norm.weight.copy_(wstate["module.layers1.0.downsample.norm.weight"])  # type: ignore
                d.norm.bias.copy_(wstate["module.layers1.0.downsample.norm.bias"])  # type: ignore

            for bname, block in layers2_0.blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers2")  # type: ignore[operator]

            if layers2_0.downsample is not None:
                d = layers2_0.downsample
                d.reduction.weight.copy_(wstate["module.layers2.0.downsample.reduction.weight"])  # type: ignore
                d.norm.weight.copy_(wstate["module.layers2.0.downsample.norm.weight"])  # type: ignore
                d.norm.bias.copy_(wstate["module.layers2.0.downsample.norm.bias"])  # type: ignore

            for bname, block in layers3_0.blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers3")  # type: ignore[operator]

            if layers3_0.downsample is not None:
                d = layers3_0.downsample
                d.reduction.weight.copy_(wstate["module.layers3.0.downsample.reduction.weight"])  # type: ignore
                d.norm.weight.copy_(wstate["module.layers3.0.downsample.norm.weight"])  # type: ignore
                d.norm.bias.copy_(wstate["module.layers3.0.downsample.norm.bias"])  # type: ignore

            for bname, block in layers4_0.blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers4")  # type: ignore[operator]

            if layers4_0.downsample is not None:
                d = layers4_0.downsample
                d.reduction.weight.copy_(wstate["module.layers4.0.downsample.reduction.weight"])  # type: ignore
                d.norm.weight.copy_(wstate["module.layers4.0.downsample.norm.weight"])  # type: ignore
                d.norm.bias.copy_(wstate["module.layers4.0.downsample.norm.bias"])  # type: ignore

    @torch.jit.unused
    def _check_input_size(self, spatial_shape):
        img_size = np.array(spatial_shape)
        remainder = (img_size % np.power(self.patch_size, 5)) > 0
        if remainder.any():
            wrong_dims = (np.where(remainder)[0] + 2).tolist()
            raise ValueError(
                f"spatial dimensions {wrong_dims} of input image (spatial shape: {spatial_shape})"
                f" must be divisible by {self.patch_size}**5."
            )

    def forward(self, x_in):
        # in: [N, C, D, H , W]
        profile_sections = self.profile_sections
        self.last_section_times_ms = {}

        section_events: dict[str, tuple[torch.cuda.Event, torch.cuda.Event]] = {}

        def record_section(section_name: str, fn):
            # Emit NVTX + torch.profiler ranges unconditionally so external
            # profilers (nsys, torch.profiler) attribute kernels to sections
            # even when the CUDA-event timing below is disabled.
            with torch.cuda.nvtx.range(section_name), record_function(section_name):
                if not profile_sections:
                    return fn()

                if not x_in.is_cuda:
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
        
        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            self._check_input_size(x_in.shape[2:])
        hidden_states_out = record_section(
            "swinViT",
            lambda: self.swinViT(x_in, self.normalize),
        )

        enc0 = record_section(
            "encoder1",
            lambda: self.encoder1(x_in),
        )

        enc1 = record_section(
            "encoder2",
            lambda: self.encoder2(hidden_states_out[0]),
        )

        enc2 = record_section(
            "encoder3",
            lambda: self.encoder3(hidden_states_out[1]),
        )

        enc3 = record_section(
            "encoder4",
            lambda: self.encoder4(hidden_states_out[2]),
        )

        dec4 = record_section(
            "encoder10",
            lambda: self.encoder10(hidden_states_out[4]),
        )

        dec3 = record_section(
            "decoder5",
            lambda: self.decoder5(dec4, hidden_states_out[3]),
        )

        dec2 = record_section(
            "decoder4",
            lambda: self.decoder4(dec3, enc3),
        )

        dec1 = record_section(
            "decoder3",
            lambda: self.decoder3(dec2, enc2),
        )

        dec0 = record_section(
            "decoder2",
            lambda: self.decoder2(dec1, enc1),
        )
        #print(f"DEC0 shape: {dec0.shape}, ENC0 shape: {enc0.shape}")
        out = record_section(
            "decoder1",
            lambda: self.decoder1(dec0, enc0),
        )
#        out = self.decoder1(dec0, enc0)
        logits = record_section(
            "out",
            lambda: self.out(out),
        )

        if profile_sections and x_in.is_cuda:
            torch.cuda.synchronize()
            self.last_section_times_ms.update(
                {
                    section_name: start.elapsed_time(end)
                    for section_name, (start, end) in section_events.items()
                }
            )

        return logits

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Profile SwinUNETR.")
    parser.add_argument(
        "--mode", choices=["sections", "torch", "nsys"], default="sections",
        help="sections: per-section CUDA-event timing (default). "
             "torch: kernel-level torch.profiler table + trace. "
             "nsys: plain warmup+loop to capture under `nsys profile`.",
    )
    parser.add_argument(
        "--train", action="store_true",
        help="profile a forward+backward step instead of inference forward.",
    )
    parser.add_argument("--iters", type=int, default=None, help="profiled iterations.")
    parser.add_argument("--trace", default="trace.json", help="torch.profiler trace path.")
    args = parser.parse_args()

    x = torch.load('model_input.pt', weights_only=False).to(torch.float32).to("cuda")
    model = SwinUNETR(
        in_channels=1,
        patch_size=2,
        out_channels=1,
        drop_rate=0.25,
        attn_drop_rate=0.25,
        feature_size=48,
        use_checkpoint=True,  # gradient checkpointing for reduced memory use at the cost of compute
    ).to(torch.float32).to("cuda")

    if args.mode == "sections":
        section_times = profile_module_forward(model, (x, ))

        total_ms = sum(section_times.values())

        for section_name, elapsed_ms in section_times.items():
            percentage = elapsed_ms / total_ms * 100.0
            print(f"{section_name}: {elapsed_ms:.3f} ms ({percentage:.1f}%)")

        print(f"Total: {total_ms:.3f} ms")

    elif args.mode == "torch":
        # The section CUDA events add a sync per section and are redundant with
        # the profiler's own timing; disable them so kernels flow uninterrupted
        # (the NVTX/record_function section labels are still emitted).
        model.profile_sections = False
        profile_module_torch_profiler(
            model, (x, ), train=args.train,
            profile_iters=args.iters or 8, trace_path=args.trace,
        )

    elif args.mode == "nsys":
        # Run under:
        #   nsys profile -t cuda,nvtx,cudnn,cublas -o swin \
        #       python FastSwinUNETR3D.py --mode nsys [--train]
        model.profile_sections = False
        n = args.iters or 20

        if args.train:
            model.train()
            base = x.detach()
            for _ in range(10):  # warmup (outside the NVTX range below)
                model.zero_grad(set_to_none=True)
                xi = base.clone().requires_grad_(True)
                model(xi).float().pow(2).mean().backward()
            torch.cuda.synchronize()
            with torch.cuda.nvtx.range("profiled_iters"):
                for _ in range(n):
                    model.zero_grad(set_to_none=True)
                    xi = base.clone().requires_grad_(True)
                    model(xi).float().pow(2).mean().backward()
                    torch.cuda.synchronize()
        else:
            model.eval()
            with torch.inference_mode():
                for _ in range(10):  # warmup
                    model(x)
                torch.cuda.synchronize()
                with torch.cuda.nvtx.range("profiled_iters"):
                    for _ in range(n):
                        model(x)
                        torch.cuda.synchronize()

        print(f"ran {n} {'train' if args.train else 'inference'} iters for nsys capture")