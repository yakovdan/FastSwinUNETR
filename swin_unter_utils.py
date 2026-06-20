from __future__ import annotations

import enum
import inspect

import numpy as np
import torch

from collections import namedtuple
from collections.abc import Iterable, Collection, Hashable, Mapping
from keyword import iskeyword
from textwrap import dedent, indent
from typing import Any, Callable, TypeVar, Sequence
from monai.networks.blocks.convolutions import Convolution
from torch import nn

T = TypeVar("T")

def ensure_tuple(vals: Any, wrap_array: bool = False) -> tuple:
    """
    Returns a tuple of `vals`.

    Args:
        vals: input data to convert to a tuple.
        wrap_array: if `True`, treat the input numerical array (ndarray/tensor) as one item of the tuple.
            if `False`, try to convert the array with `tuple(vals)`, default to `False`.

    """
    if wrap_array and isinstance(vals, (np.ndarray, torch.Tensor)):
        return (vals,)
    return tuple(vals) if issequenceiterable(vals) else (vals,)

def has_option(obj: Callable, keywords: str | Sequence[str]) -> bool:
    """
    Return a boolean indicating whether the given callable `obj` has the `keywords` in its signature.
    """
    if not callable(obj):
        return False
    sig = inspect.signature(obj)
    return all(key in sig.parameters for key in ensure_tuple(keywords))

def get_norm_layer(name: tuple | str, spatial_dims: int | None = 1, channels: int | None = 1):
    """
    Create a normalization layer instance.

    For example, to create normalization layers:

    .. code-block:: python

        from monai.networks.layers import get_norm_layer

        g_layer = get_norm_layer(name=("group", {"num_groups": 1}))
        n_layer = get_norm_layer(name="instance", spatial_dims=2)

    Args:
        name: a normalization type string or a tuple of type string and parameters.
        spatial_dims: number of spatial dimensions of the input.
        channels: number of features/channels when the normalization layer requires this parameter
            but it is not specified in the norm parameters.
    """
    if name == "":
        return torch.nn.Identity()
    norm_name, norm_args = split_args(name)
    norm_type = Norm[norm_name, spatial_dims]
    kw_args = dict(norm_args)
    if has_option(norm_type, "num_features") and "num_features" not in kw_args:
        kw_args["num_features"] = channels
    if has_option(norm_type, "num_channels") and "num_channels" not in kw_args:
        kw_args["num_channels"] = channels
    return norm_type(**kw_args)

def split_args(args):
    """
    Split arguments in a way to be suitable for using with the factory types. If `args` is a string it's interpreted as
    the type name.

    Args:
        args (str or a tuple of object name and kwarg dict): input arguments to be parsed.

    Raises:
        TypeError: When ``args`` type is not in ``Union[str, Tuple[Union[str, Callable], dict]]``.

    Examples::

        >>> act_type, args = split_args("PRELU")
        >>> monai.networks.layers.Act[act_type]
        <class 'torch.nn.modules.activation.PReLU'>

        >>> act_type, args = split_args(("PRELU", {"num_parameters": 1, "init": 0.25}))
        >>> monai.networks.layers.Act[act_type](**args)
        PReLU(num_parameters=1)

    """

    if isinstance(args, str):
        return args, {}
    name_obj, name_args = args

    if not (isinstance(name_obj, str) or callable(name_obj)) or not isinstance(name_args, dict):
        msg = "Layer specifiers must be single strings or pairs of the form (name/object-types, argument dict)"
        raise TypeError(msg)

    return name_obj, name_args

def get_window_size(x_size, window_size, shift_size=None):
    """Computing window size based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        x_size: input size.
        window_size: local window size.
        shift_size: window shifting size.
    """

    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)

def damerau_levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculates the Damerau–Levenshtein distance between two strings for spelling correction.
    https://en.wikipedia.org/wiki/Damerau–Levenshtein_distance
    """
    if s1 == s2:
        return 0
    string_1_length = len(s1)
    string_2_length = len(s2)
    if not s1:
        return string_2_length
    if not s2:
        return string_1_length
    d = {(i, -1): i + 1 for i in range(-1, string_1_length + 1)}
    for j in range(-1, string_2_length + 1):
        d[(-1, j)] = j + 1

    for i, s1i in enumerate(s1):
        for j, s2j in enumerate(s2):
            cost = 0 if s1i == s2j else 1
            d[(i, j)] = min(
                d[(i - 1, j)] + 1, d[(i, j - 1)] + 1, d[(i - 1, j - 1)] + cost  # deletion  # insertion  # substitution
            )
            if i and j and s1i == s2[j - 1] and s1[i - 1] == s2j:
                d[(i, j)] = min(d[(i, j)], d[i - 2, j - 2] + cost)  # transposition

    return d[string_1_length - 1, string_2_length - 1]

def look_up_option(
    opt_str: Hashable,
    supported: Collection | enum.EnumMeta,
    default: Any = "no_default",
    print_all_options: bool = True,
) -> Any:
    """
    Look up the option in the supported collection and return the matched item.
    Raise a value error possibly with a guess of the closest match.

    Args:
        opt_str: The option string or Enum to look up.
        supported: The collection of supported options, it can be list, tuple, set, dict, or Enum.
        default: If it is given, this method will return `default` when `opt_str` is not found,
            instead of raising a `ValueError`. Otherwise, it defaults to `"no_default"`,
            so that the method may raise a `ValueError`.
        print_all_options: whether to print all available options when `opt_str` is not found. Defaults to True

    Examples:

    .. code-block:: python

        from enum import Enum
        from monai.utils import look_up_option
        class Color(Enum):
            RED = "red"
            BLUE = "blue"
        look_up_option("red", Color)  # <Color.RED: 'red'>
        look_up_option(Color.RED, Color)  # <Color.RED: 'red'>
        look_up_option("read", Color)
        # ValueError: By 'read', did you mean 'red'?
        # 'read' is not a valid option.
        # Available options are {'blue', 'red'}.
        look_up_option("red", {"red", "blue"})  # "red"

    Adapted from https://github.com/NifTK/NiftyNet/blob/v0.6.0/niftynet/utilities/util_common.py#L249
    """
    if not isinstance(opt_str, Hashable):
        raise ValueError(f"Unrecognized option type: {type(opt_str)}:{opt_str}.")
    if isinstance(opt_str, str):
        opt_str = opt_str.strip()
    if isinstance(supported, enum.EnumMeta):
        if isinstance(opt_str, str) and opt_str in {item.value for item in supported}:  # type: ignore
            # such as: "example" in MyEnum
            return supported(opt_str)
        if isinstance(opt_str, enum.Enum) and opt_str in supported:
            # such as: MyEnum.EXAMPLE in MyEnum
            return opt_str
    elif isinstance(supported, Mapping) and opt_str in supported:
        # such as: MyDict[key]
        return supported[opt_str]
    elif isinstance(supported, Collection) and opt_str in supported:
        return opt_str

    if default != "no_default":
        return default

    # find a close match
    set_to_check: set
    if isinstance(supported, enum.EnumMeta):
        set_to_check = {item.value for item in supported}  # type: ignore
    else:
        set_to_check = set(supported) if supported is not None else set()
    if not set_to_check:
        raise ValueError(f"No options available: {supported}.")
    edit_dists = {}
    opt_str = f"{opt_str}"
    for key in set_to_check:
        edit_dist = damerau_levenshtein_distance(f"{key}", opt_str)
        if edit_dist <= 3:
            edit_dists[key] = edit_dist

    supported_msg = f"Available options are {set_to_check}.\n" if print_all_options else ""
    if edit_dists:
        guess_at_spelling = min(edit_dists, key=edit_dists.get)  # type: ignore
        raise ValueError(
            f"By '{opt_str}', did you mean '{guess_at_spelling}'?\n"
            + f"'{opt_str}' is not a valid value.\n"
            + supported_msg
        )
    raise ValueError(f"Unsupported option '{opt_str}', " + supported_msg)

def issequenceiterable(obj: Any) -> bool:
    """
    Determine if the object is an iterable sequence and is not a string.
    """
    try:
        if hasattr(obj, "ndim") and obj.ndim == 0:
            return False  # a 0-d tensor is not iterable
    except Exception:
        return False
    return isinstance(obj, Iterable) and not isinstance(obj, (str, bytes))

def ensure_tuple_rep(tup: Any, dim: int) -> tuple[Any, ...]:
    """
    Returns a copy of `tup` with `dim` values by either shortened or duplicated input.

    Raises:
        ValueError: When ``tup`` is a sequence and ``tup`` length is not ``dim``.

    Examples::

        >>> ensure_tuple_rep(1, 3)
        (1, 1, 1)
        >>> ensure_tuple_rep(None, 3)
        (None, None, None)
        >>> ensure_tuple_rep('test', 3)
        ('test', 'test', 'test')
        >>> ensure_tuple_rep([1, 2, 3], 3)
        (1, 2, 3)
        >>> ensure_tuple_rep(range(3), 3)
        (0, 1, 2)
        >>> ensure_tuple_rep([1, 2], 3)
        ValueError: Sequence must have length 3, got length 2.

    """
    if isinstance(tup, torch.Tensor):
        tup = tup.detach().cpu().numpy()
    if isinstance(tup, np.ndarray):
        tup = tup.tolist()
    if not issequenceiterable(tup):
        return (tup,) * dim
    if len(tup) == dim:
        return tuple(tup)

    raise ValueError(f"Sequence must have length {dim}, got {len(tup)}.")


def is_variable(name):
    """Returns True if `name` is a valid Python variable name and also not a keyword."""
    return name.isidentifier() and not iskeyword(name)

class ComponentStore:
    """
    Represents a storage object for other objects (specifically functions) keyed to a name with a description.

    These objects act as global named places for storing components for objects parameterised by component names.
    Typically this is functions although other objects can be added. Printing a component store will produce a
    list of members along with their docstring information if present.

    Example:

    .. code-block:: python

        TestStore = ComponentStore("Test Store", "A test store for demo purposes")

        @TestStore.add_def("my_func_name", "Some description of your function")
        def _my_func(a, b):
            '''A description of your function here.'''
            return a * b

        print(TestStore)  # will print out name, description, and 'my_func_name' with the docstring

        func = TestStore["my_func_name"]
        result = func(7, 6)

    """

    _Component = namedtuple("_Component", ("description", "value"))  # internal value pair

    def __init__(self, name: str, description: str) -> None:
        self.components: dict[str, ComponentStore._Component] = {}
        self.name: str = name
        self.description: str = description

        self.__doc__ = f"Component Store '{name}': {description}\n{self.__doc__ or ''}".strip()

    def add(self, name: str, desc: str, value: T) -> T:
        """Store the object `value` under the name `name` with description `desc`."""
        if not is_variable(name):
            raise ValueError("Name of component must be valid Python identifier")

        self.components[name] = self._Component(desc, value)
        return value

    def add_def(self, name: str, desc: str) -> Callable:
        """Returns a decorator which stores the decorated function under `name` with description `desc`."""

        def deco(func):
            """Decorator to add a function to a store."""
            return self.add(name, desc, func)

        return deco

    @property
    def names(self) -> tuple[str, ...]:
        """
        Produces all factory names.
        """

        return tuple(self.components)

    def __contains__(self, name: str) -> bool:
        """Returns True if the given name is stored."""
        return name in self.components

    def __len__(self) -> int:
        """Returns the number of stored components."""
        return len(self.components)

    def __iter__(self) -> Iterable:
        """Yields name/component pairs."""
        for k, v in self.components.items():
            yield k, v.value

    def __str__(self):
        result = f"Component Store '{self.name}': {self.description}\nAvailable components:"
        for k, v in self.components.items():
            result += f"\n* {k}:"

            if hasattr(v.value, "__doc__") and v.value.__doc__:
                doc = indent(dedent(v.value.__doc__.lstrip("\n").rstrip()), "    ")
                result += f"\n{doc}\n"
            else:
                result += f" {v.description}"

        return result

    def __getattr__(self, name: str) -> Any:
        """Returns the stored object under the given name."""
        if name in self.components:
            return self.components[name].value
        else:
            return self.__getattribute__(name)

    def __getitem__(self, name: str) -> Any:
        """Returns the stored object under the given name."""
        if name in self.components:
            return self.components[name].value
        else:
            raise ValueError(f"Component '{name}' not found")

class LayerFactory(ComponentStore):
    """
    Factory object for creating layers, this uses given factory functions to actually produce the types or constructing
    callables. These functions are referred to by name and can be added at any time.
    """

    def __init__(self, name: str, description: str) -> None:
        super().__init__(name, description)
        self.__doc__ = (
            f"Layer Factory '{name}': {description}\n".strip()
            + "\nPlease see :py:class:`monai.networks.layers.split_args` for additional args parsing."
            + "\n\nThe supported members are:"
        )

    def add_factory_callable(self, name: str, func: Callable, desc: str | None = None) -> None:
        """
        Add the factory function to this object under the given name, with optional description.
        """
        description: str = desc or func.__doc__ or ""
        self.add(name.upper(), description, func)
        # append name to the docstring
        assert self.__doc__ is not None
        self.__doc__ += f"{', ' if len(self.names)>1 else ' '}``{name}``"

    def add_factory_class(self, name: str, cls: type, desc: str | None = None) -> None:
        """
        Adds a factory function which returns the supplied class under the given name, with optional description.
        """
        self.add_factory_callable(name, lambda x=None: cls, desc)

    def factory_function(self, name: str) -> Callable:
        """
        Decorator for adding a factory function with the given name.
        """

        def _add(func: Callable) -> Callable:
            self.add_factory_callable(name, func)
            return func

        return _add

    def get_constructor(self, factory_name: str, *args) -> Any:
        """
        Get the constructor for the given factory name and arguments.

        Raises:
            TypeError: When ``factory_name`` is not a ``str``.

        """

        if not isinstance(factory_name, str):
            raise TypeError(f"factory_name must a str but is {type(factory_name).__name__}.")

        component = look_up_option(factory_name.upper(), self.components)

        return component.value(*args)

    def __getitem__(self, args) -> Any:
        """
        Get the given name or name/arguments pair. If `args` is a callable it is assumed to be the constructor
        itself and is returned, otherwise it should be the factory name or a pair containing the name and arguments.
        """

        # `args[0]` is actually a type or constructor
        if callable(args):
            return args

        # `args` is a factory name or a name with arguments
        if isinstance(args, str):
            name_obj, args = args, ()
        else:
            name_obj, *args = args

        return self.get_constructor(name_obj, *args)

    def __getattr__(self, key):
        """
        If `key` is a factory name, return it, otherwise behave as inherited. This allows referring to factory names
        as if they were constants, eg. `Fact.FOO` for a factory Fact with factory function foo.
        """

        if key in self.components:
            return key

        return super().__getattribute__(key)

Act = LayerFactory(name="Activation layers", description="Factory for creating activation layers.")



Act.add_factory_class("elu", nn.modules.ELU)
Act.add_factory_class("relu", nn.modules.ReLU)
Act.add_factory_class("leakyrelu", nn.modules.LeakyReLU)
Act.add_factory_class("prelu", nn.modules.PReLU)
Act.add_factory_class("relu6", nn.modules.ReLU6)
Act.add_factory_class("selu", nn.modules.SELU)
Act.add_factory_class("celu", nn.modules.CELU)
Act.add_factory_class("gelu", nn.modules.GELU)
Act.add_factory_class("sigmoid", nn.modules.Sigmoid)
Act.add_factory_class("tanh", nn.modules.Tanh)
Act.add_factory_class("softmax", nn.modules.Softmax)
Act.add_factory_class("logsoftmax", nn.modules.LogSoftmax)

def get_act_layer(name: tuple | str):
    """
    Create an activation layer instance.

    For example, to create activation layers:

    .. code-block:: python

        from monai.networks.layers import get_act_layer

        s_layer = get_act_layer(name="swish")
        p_layer = get_act_layer(name=("prelu", {"num_parameters": 1, "init": 0.25}))

    Args:
        name: an activation type string or a tuple of type string and parameters.
    """
    if name == "":
        return torch.nn.Identity()
    act_name, act_args = split_args(name)
    act_type = Act[act_name]
    return act_type(**act_args)

def window_partition(x, window_size):
    """window partition operation based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        x: input tensor.
        window_size: local window size.
    """
    x_shape = x.size()  # length 4 or 5 only
    if len(x_shape) == 5:
        b, d, h, w, c = x_shape
        x = x.view(
            b,
            d // window_size[0],
            window_size[0],
            h // window_size[1],
            window_size[1],
            w // window_size[2],
            window_size[2],
            c,
        )
        windows = (
            x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0] * window_size[1] * window_size[2], c)
        )
    else:  # if len(x_shape) == 4:
        b, h, w, c = x.shape
        x = x.view(b, h // window_size[0], window_size[0], w // window_size[1], window_size[1], c)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0] * window_size[1], c)

    return windows


def window_reverse(windows, window_size, dims):
    """window reverse operation based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        windows: windows tensor.
        window_size: local window size.
        dims: dimension values.
    """
    if len(dims) == 4:
        b, d, h, w = dims
        x = windows.view(
            b,
            d // window_size[0],
            h // window_size[1],
            w // window_size[2],
            window_size[0],
            window_size[1],
            window_size[2],
            -1,
        )
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(b, d, h, w, -1)

    elif len(dims) == 3:
        b, h, w = dims
        x = windows.view(b, h // window_size[0], w // window_size[1], window_size[0], window_size[1], -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    else:
        raise RuntimeError("len(dims) must be 4 or 3")
    return x


def compute_mask(dims, window_size, shift_size, device):
    """Computing region masks based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        dims: dimension values.
        window_size: local window size.
        shift_size: shift size.
        device: device.
    """

    cnt = 0

    if len(dims) == 3:
        d, h, w = dims
        img_mask = torch.zeros((1, d, h, w, 1), device=device)
        for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
            for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
                for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                    img_mask[:, d, h, w, :] = cnt
                    cnt += 1

    elif len(dims) == 2:
        h, w = dims
        img_mask = torch.zeros((1, h, w, 1), device=device)
        for h in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
            for w in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
                img_mask[:, h, w, :] = cnt
                cnt += 1

    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

    return attn_mask

Norm = LayerFactory(name="Normalization layers", description="Factory for creating normalization layers.")
Norm.add_factory_class("group", nn.GroupNorm)
Norm.add_factory_class("layer", nn.LayerNorm)
Norm.add_factory_class("localresponse", nn.LocalResponseNorm)
Norm.add_factory_class("syncbatch", nn.SyncBatchNorm)


@Norm.factory_function("instance")
def instance_factory(dim: int) -> type[nn.InstanceNorm1d | nn.InstanceNorm2d | nn.InstanceNorm3d]:
    """
    Instance normalization layers in 1,2,3 dimensions.

    Args:
        dim: desired dimension of the instance normalization layer

    Returns:
        InstanceNorm[dim]d
    """
    types = (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)
    return types[dim - 1]


@Norm.factory_function("batch")
def batch_factory(dim: int) -> type[nn.BatchNorm1d | nn.BatchNorm2d | nn.BatchNorm3d]:
    """
    Batch normalization layers in 1,2,3 dimensions.

    Args:
        dim: desired dimension of the batch normalization layer

    Returns:
        BatchNorm[dim]d
    """
    types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
    return types[dim - 1]

def get_padding(kernel_size: Sequence[int] | int, stride: Sequence[int] | int) -> tuple[int, ...] | int:
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = (kernel_size_np - stride_np + 1) / 2
    if np.min(padding_np) < 0:
        raise AssertionError("padding value should not be negative, please change the kernel size and/or stride.")
    padding = tuple(int(p) for p in padding_np)

    return padding if len(padding) > 1 else padding[0]
def get_output_padding(
    kernel_size: Sequence[int] | int, stride: Sequence[int] | int, padding: Sequence[int] | int
) -> tuple[int, ...] | int:
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = np.atleast_1d(padding)

    out_padding_np = 2 * padding_np + stride_np - kernel_size_np
    if np.min(out_padding_np) < 0:
        raise AssertionError("out_padding value should not be negative, please change the kernel size and/or stride.")
    out_padding = tuple(int(p) for p in out_padding_np)

    return out_padding if len(out_padding) > 1 else out_padding[0]

def get_conv_layer(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    kernel_size: Sequence[int] | int = 3,
    stride: Sequence[int] | int = 1,
    act: tuple | str | None = Act.PRELU,
    norm: tuple | str | None = Norm.INSTANCE,
    dropout: tuple | str | float | None = None,
    bias: bool = False,
    conv_only: bool = True,
    is_transposed: bool = False,
):
    padding = get_padding(kernel_size, stride)
    output_padding = None
    if is_transposed:
        output_padding = get_output_padding(kernel_size, stride, padding)
    return Convolution(
        spatial_dims,
        in_channels,
        out_channels,
        strides=stride,
        kernel_size=kernel_size,
        act=act,
        norm=norm,
        dropout=dropout,
        bias=bias,
        conv_only=conv_only,
        is_transposed=is_transposed,
        padding=padding,
        output_padding=output_padding,
    )
