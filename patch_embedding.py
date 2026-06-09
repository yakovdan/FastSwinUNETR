from torch import Tensor, nn
from typing import Sequence
from torch.nn import LayerNorm
from swin_unter_utils import ensure_tuple_rep
import torch.nn.functional as F

from jaxtyping import Float, jaxtyped
from typeguard import typechecked as typechecker

class PatchEmbed2D(nn.Module):
    """
    Patch embedding block based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

    Unlike ViT patch embedding block: (1) input is padded to satisfy window size requirements (2) normalized if
    specified (3) position embedding is not used.

    Example::

        >>> from monai.networks.blocks import PatchEmbed
        >>> PatchEmbed(patch_size=2, in_chans=1, embed_dim=48, norm_layer=nn.LayerNorm, spatial_dims=3)
    """

    def __init__(
        self,
        patch_size: Sequence[int] | int = 2,
        in_chans: int = 1,
        embed_dim: int = 48,
        norm_layer: type[LayerNorm] | None = nn.LayerNorm,
        spatial_dims: int = 2,
    ) -> None:
        """
        Args:
            patch_size: dimension of patch size.
            in_chans: dimension of input channels.
            embed_dim: number of linear projection output channels.
            norm_layer: normalization layer.
            spatial_dims: spatial dimension.
        """

        super().__init__()

        if spatial_dims != 2:
            raise ValueError("spatial dimension should be 2.")

        if norm_layer is not None:
            raise ValueError("Currently only None is supported for norm_layer")

        patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(
            in_channels=in_chans, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size
        )


    def forward(self, x):
        x_shape = x.size()
        if len(x_shape) != 4:
            raise ValueError(f"expecting 4D x, got {x.shape}.")
        _, _, h, w = x_shape
        if w % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - w % self.patch_size[1]))
        if h % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - h % self.patch_size[0]))
        x = self.proj(x)

        return x


class PatchEmbed3D(nn.Module):
    """
    Patch embedding block based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

    Unlike ViT patch embedding block: (1) input is padded to satisfy window size requirements (2) normalized if
    specified (3) position embedding is not used.

    Example::

        >>> from monai.networks.blocks import PatchEmbed
        >>> PatchEmbed(patch_size=2, in_chans=1, embed_dim=48, norm_layer=nn.LayerNorm, spatial_dims=3)
    """

    def __init__(
        self,
        patch_size: Sequence[int] | int = 2,
        in_chans: int = 1,
        embed_dim: int = 48,
        norm_layer: type[LayerNorm] | None = nn.LayerNorm,
        spatial_dims: int = 3,
    ) -> None:
        """
        Args:
            patch_size: dimension of patch size.
            in_chans: dimension of input channels.
            embed_dim: number of linear projection output channels.
            norm_layer: normalization layer.
            spatial_dims: spatial dimension.
        """

        super().__init__()

        if spatial_dims != 3:
            raise ValueError("spatial dimension should be 3.")

        if norm_layer is not None:
            raise ValueError("Currently only None is supported for norm_layer")

        patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = nn.Conv3d(
            in_channels=in_chans, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size
        )

    @jaxtyped(typechecker=typechecker)
    def forward(
        self,
        x: Float[Tensor, "b 1 d h w"],
    ) -> Float[
        Tensor,
        "b {self.embed_dim} "
        "(d+{self.patch_size[0]}-1)//{self.patch_size[0]} "
        "(h+{self.patch_size[1]}-1)//{self.patch_size[1]} "
        "(w+{self.patch_size[2]}-1)//{self.patch_size[2]}",
    ]:
        x_shape = x.size()
        if len(x_shape) != 5:
            raise ValueError(f"expecting 5D x, got {x.shape}.")
        _, _, d, h, w = x_shape
        if w % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - w % self.patch_size[2]))
        if h % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - h % self.patch_size[1]))
        if d % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - d % self.patch_size[0]))

        x = self.proj(x)

        return x
