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
from patch_merging import PatchMerging, PatchMergingV2
from einops import rearrange
from SwinAttention import FastWindowAttention as WindowAttention
from unter import UnetrBasicBlock, UnetrUpBlock, UnetOutBlock
from fast_layer_norm import FastLayerNorm


SUPPORTED_DROPOUT_MODE = {"vit", "swin", "vista3d"}
class MLPBlock(nn.Module):
    """
    A multi-layer perceptron block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    """

    def __init__(
        self, hidden_size: int, mlp_dim: int, dropout_rate: float = 0.0, act: tuple | str = "GELU", dropout_mode="vit"
    ) -> None:
        """
        Args:
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer. If 0, `hidden_size` will be used.
            dropout_rate: fraction of the input units to drop.
            act: activation type and arguments. Defaults to GELU. Also supports "GEGLU" and others.
            dropout_mode: dropout mode, can be "vit" or "swin".
                "vit" mode uses two dropout instances as implemented in
                https://github.com/google-research/vision_transformer/blob/main/vit_jax/models.py#L87
                "swin" corresponds to one instance as implemented in
                https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_mlp.py#L23
                "vista3d" mode does not use dropout.

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")
        mlp_dim = mlp_dim or hidden_size
        act_name, _ = split_args(act)
        self.linear1 = nn.Linear(hidden_size, mlp_dim) if act_name != "GEGLU" else nn.Linear(hidden_size, mlp_dim * 2)
        self.linear2 = nn.Linear(mlp_dim, hidden_size)
        self.fn = get_act_layer(act)
        # Use Union[nn.Dropout, nn.Identity] for type annotations
        self.drop1: Union[nn.Dropout, nn.Identity]
        self.drop2: Union[nn.Dropout, nn.Identity]

        dropout_opt = look_up_option(dropout_mode, SUPPORTED_DROPOUT_MODE)
        if dropout_opt == "vit":
            self.drop1 = nn.Dropout(dropout_rate)
            self.drop2 = nn.Dropout(dropout_rate)
        elif dropout_opt == "swin":
            self.drop1 = nn.Dropout(dropout_rate)
            self.drop2 = self.drop1
        elif dropout_opt == "vista3d":
            self.drop1 = nn.Identity()
            self.drop2 = nn.Identity()
        else:
            raise ValueError(f"dropout_mode should be one of {SUPPORTED_DROPOUT_MODE}")

    def forward(self, x):
        x = self.fn(self.linear1(x))
        x = self.drop1(x)
        x = self.linear2(x)
        x = self.drop2(x)
        return x