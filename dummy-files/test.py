"""Collection of PyTorch network layers, wrapped to fit Ivy syntax and signature."""

from typing import Optional, Tuple, Union, Sequence

# global
import torch

# local
import ivy
from ivy.func_wrapper import with_unsupported_dtypes, with_supported_dtypes
from . import backend_version
from ivy.functional.ivy.layers import _get_embed_dim, _handle_padding, _deconv_length


@with_supported_dtypes(
    {"2.1.0 and below": ("float32", "float64", "complex")},
    backend_version,
)
def multi_head_attention(
    query: torch.Tensor,
    /,
    *,
    key: torch.Tensor = None,
    value: torch.Tensor = None,
    batch_first: bool = True,
    num_heads: Optional[int] = 8,
    scale: Optional[float] = None,
    attention_mask: torch.Tensor = None,
    in_proj_weights: torch.Tensor = None,
    q_proj_weights: torch.Tensor = None,
    k_proj_weights: torch.Tensor = None,
    v_proj_weights: torch.Tensor = None,
    out_proj_weights: torch.Tensor = None,
    in_proj_bias: torch.Tensor = None,
    out_proj_bias: torch.Tensor = None,
    is_causal: Optional[bool] = False,
    key_padding_mask: Optional[torch.Tensor] = None,
    bias_k: Optional[torch.Tensor] = None,
    bias_v: Optional[torch.Tensor] = None,
    static_k: Optional[torch.Tensor] = None,
    static_v: Optional[torch.Tensor] = None,
    add_zero_attn: bool = False,
    return_attention_weights: Optional[bool] = False,
    average_attention_weights: Optional[bool] = True,
    dropout: Optional[float] = 0.0,
    training: Optional[bool] = False,
    out: torch.Tensor = None,
) -> torch.Tensor:
    
    if key is None and value is None:
        key = value = query
    emb_dim = _get_embed_dim(
        in_proj_weights,
        q_proj_weights,
        k_proj_weights,
        v_proj_weights,
        query,
    )[1]
    num_dims = query.ndim
    if num_dims == 3 and batch_first:
        query, key, value = [torch.swapaxes(x, 0, 1) for x in [query, key, value]]
    ret = torch.nn.functional.multi_head_attention_forward(
        query,
        key,
        value,
        emb_dim,
        num_heads,
        in_proj_weights,
        in_proj_bias,
        bias_k,
        bias_v,
        add_zero_attn,
        dropout,
        out_proj_weights,
        out_proj_bias,
        training=training,
        key_padding_mask=key_padding_mask,
        need_weights=return_attention_weights,
        attn_mask=attention_mask,
        use_separate_proj_weight=not ivy.exists(in_proj_weights),
        q_proj_weight=q_proj_weights,
        k_proj_weight=k_proj_weights,
        v_proj_weight=v_proj_weights,
        static_k=static_k,
        static_v=static_v,
        average_attn_weights=average_attention_weights,
        is_causal=is_causal,
    )
    ret = list(ret) if isinstance(ret, tuple) else [ret]
    if num_dims == 3 and batch_first:
        ret[0] = ret[0].swapaxes(0, 1)
    if return_attention_weights:
        return tuple(ret)
    return ret[0]


multi_head_attention.partial_mixed_handler = (
    lambda *args, scale=None, out_proj_weights=None, is_causal=False, attention_mask=None, return_attention_weights=False, in_proj_weights=None, q_proj_weights=None, k_proj_weights=None, v_proj_weights=None, **kwargs: not ivy.exists(  # noqa: E501
        scale
    )
    and ivy.exists(out_proj_weights)
    and (not is_causal or ivy.exists(attention_mask))
    and (not is_causal or not return_attention_weights)
    and (
        ivy.exists(in_proj_weights)
        or all(ivy.exists(x) for x in [q_proj_weights, k_proj_weights, v_proj_weights])
    )
    and len(
        set(
            _get_embed_dim(
                in_proj_weights, q_proj_weights, k_proj_weights, v_proj_weights, args[0]
            )
        )
    )
    == 1
)


@with_unsupported_dtypes(
    {"2.1.0 and below": ("float16", "bfloat16", "complex")},
    backend_version,
)
def linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    /,
    *,
    bias: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    
    return torch.nn.functional.linear(x, weight, bias)


linear.partial_mixed_handler = lambda x, weight, **kwargs: weight.ndim == 2


def _ff_xd_before_conv(x, filters, dims, filter_format, x_dilations):
    """Adds dilations to the input before performing convolution.
    
    This function adds dilations to the input tensor ``x`` before performing 
    convolution with ``filters``. 
    
    Parameters
    ----------
    x : torch.Tensor
        The input tensor.
    filters : torch.Tensor 
        The convolution filters.
    dims : int
        The number of dimensions for convolution.
    filter_format : str
        The format of the filters, either 'channel_first' or 'channel_last'.
    x_dilations : int or tuple of ints
        The dilation rates to apply to the input tensor. Can be a single int 
        if same dilation along all axes, or a tuple specifying dilation for each axis.
    
    Returns
    -------
    tuple
        x : torch.Tensor
            The input tensor after dilations have been applied.
        filters : torch.Tensor
            The (optionally permuted) convolution filters.
            
    """
    if filter_format == "channel_last":
        filters = filters.permute(-1, -2, *range(dims))

    # adding dilation to input
    x_dilations = [x_dilations] * dims if isinstance(x_dilations, int) else x_dilations
    for i in range(dims):
        if x_dilations[i] > 1:
            h = x.shape[2 + i]
            new_height = h + (h - 1) * (x_dilations[i] - 1)
            h = torch.eye(
                new_height,
                dtype=x.dtype,
                device=ivy.as_native_dev(ivy.default_device()),
            )[:: x_dilations[i]]
            x = torch.swapaxes(x, 2 + i, -1)
            x = torch.matmul(x, h)
            x = torch.swapaxes(x, -1, 2 + i)
    return x, filters


def _pad_before_conv(
    x, filters, strides, padding, dims, dilations, filter_format="channel_last"
):
    """
    Pads the input x before passing it into a convolution operation.
    
    This handles padding in the case where strides > 1 or dilations > 1, 
    which require padding the input in a non-symmetric way to match PyTorch's
    behavior.
    
    Parameters
    ----------
    x : torch.Tensor
        The input tensor to be padded.
    filters : torch.Tensor 
        The filters/kernel for the convolution.
    strides : int or tuple of ints
        The stride for the convolution.
    padding : str, int or tuple of tuple of ints 
        The padding mode or amount of padding to apply.
    dims : int
        The number of spatial dimensions for the convolution.
    dilations: int or tuple of ints
        The dilation amounts.
    filter_format : str
        Either 'channel_first' or 'channel_last' indicating the ordering of 
        channels in filters.
    
    Returns
    -------
    x_padded : torch.Tensor
        The input x with the appropriate padding applied.
    padding_str : str 
        The padding argument to pass to the PyTorch convolution function.
        This will be 'valid' if custom padding was applied to x, or the original
        padding mode otherwise.
    
    """
    dilations = [dilations] * dims if isinstance(dilations, int) else dilations
    strides = [strides] * dims if isinstance(strides, int) else strides
    filter_shape = (
        filters.shape[2:] if filter_format == "channel_first" else filters.shape[:dims]
    )
    if isinstance(padding, str):
        # use torch's padding in conv if strides are all 1
        if len(strides) == strides.count(1):
            return x, padding.lower()
        filter_shape = [
            filter_shape[i] + (filter_shape[i] - 1) * (dilations[i] - 1)
            for i in range(dims)
        ]
        pad_specific = [
            _handle_padding(x.shape[2 + i], strides[i], filter_shape[i], padding)
            for i in range(dims - 1, -1, -1)
        ]
        pad_list_top = [pad_specific[i] // 2 for i in range(dims)]
        pad_list_bot = [pad_specific[i] - pad_specific[i] // 2 for i in range(dims)]
        pad_list = [None] * len(pad_list_top) * 2
        pad_list[::2] = pad_list_top
        pad_list[1::2] = pad_list_bot
    else:
        if isinstance(padding, int):
            return x, padding
        # if symmetric padding is used, use torch's padding in conv function
        if all(pad[0] == pad[1] for pad in padding):
            return x, [pad[0] for pad in padding]
        pad_list = [item for sublist in padding for item in sublist[::-1]][::-1]
    return torch.nn.functional.pad(x, pad_list), "valid"


def _pad_before_conv_tranpose(
    x, filters, strides, padding, dims, dilations, output_shape, filter_shape
):
    """
    Pads the input x before passing it into a convolution operation.
    
    This handles padding in the case where strides > 1 or dilations > 1, 
    which require padding the input in a non-symmetric way to match PyTorch's
    behavior.
    
    Parameters
    ----------
    x : torch.Tensor
        The input tensor to be padded.
    filters : torch.Tensor 
        The filters/kernel for the convolution.
    strides : int or tuple of ints
        The stride for the convolution.
    padding : str, int or tuple of tuple of ints 
        The padding mode or amount of padding to apply.
    dims : int
        The number of spatial dimensions for the convolution.
    dilations: int or tuple of ints
        The dilation amounts.
    filter_format : str
        Either 'channel_first' or 'channel_last' indicating the ordering of 
        channels in filters.
    
    Returns
    -------
    x_padded : torch.Tensor
        The input x with the appropriate padding applied.
    padding_str : str 
        The padding argument to pass to the PyTorch convolution function.
        This will be 'valid' if custom padding was applied to x, or the original
        padding mode otherwise.
    
    """
    if output_shape is None:
        out_shape = [
            _deconv_length(
                x.shape[i + 2], strides[i], filter_shape[i], padding, dilations[i]
            )
            for i in range(dims)
        ]
        output_shape = [x.shape[0], *out_shape, filters.shape[1]]
    elif len(output_shape) == dims:
        output_shape = [x.shape[0]] + output_shape + [filters.shape[1]]
    not_valid_pad = [False] * dims
    filter_shape = [
        filter_shape[i] + (filter_shape[i] - 1) * (dilations[i] - 1)
        for i in range(dims)
    ]
    pad_specific = [
        _handle_padding(output_shape[i + 1], strides[i], filter_shape[i], padding)
        for i in range(dims)
    ]
    if padding == "VALID":
        padding_list = [0] * dims
    else:
        for i in range(dims):
            if pad_specific[i] % 2 != 0:
                pad_specific[i] -= 1
                not_valid_pad[i] = True
        padding_list = [pad_specific[i] // 2 for i in range(dims)]
    out_shape = [
        (x.shape[i + 2] - 1) * strides[i]
        - 2 * padding_list[i]
        + dilations[i] * (filters.shape[i + 2] - 1)
        + 1
        for i in range(dims)
    ]
    output_padding = [max(output_shape[i + 1] - out_shape[i], 0) for i in range(dims)]
    return not_valid_pad, padding_list, output_padding


@with_unsupported_dtypes(
    {"2.1.0 and below": ("float16", "bfloat16", "complex")},
    backend_version,
)
# noinspection PyUnresolvedReferences
def conv1d(
    x: torch.Tensor,
    filters: torch.Tensor,
    strides: Union[int, Tuple[int]],
    padding: Union[str, int, Sequence[Tuple[int, int]]],
    /,
    *,
    data_format: str = "NWC",
    filter_format: str = "channel_last",
    x_dilations: Union[int, Tuple[int]] = 1,
    dilations: Union[int, Tuple[int]] = 1,
    bias: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if data_format == "NWC":
        x = x.permute(0, 2, 1)
    x, filters = _ff_xd_before_conv(x, filters, 1, filter_format, x_dilations)
    x, padding = _pad_before_conv(
        x, filters, strides, padding, 1, dilations, "channel_first"
    )
    res = torch.nn.functional.conv1d(x, filters, bias, strides, padding, dilations)
    if data_format == "NWC":
        res = res.permute(0, 2, 1)
    return res


@with_unsupported_dtypes(
    {
        "2.1.0 and below": (
            "float16",
            "bfloat16",
            "complex",
        )
    },
    backend_version,
)
# noinspection PyUnresolvedReferences
def conv1d_transpose(
    x: torch.Tensor,
    filters: torch.Tensor,
    strides: Union[int, Tuple[int]],
    padding: str,
    /,
    *,
    output_shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    data_format: str = "NWC",
    dilations: Union[int, Tuple[int]] = 1,
    bias: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
):
    if data_format == "NWC":
        x = x.permute(0, 2, 1)
    filters = filters.permute(1, 2, 0)
    strides = [strides] if isinstance(strides, int) else strides
    dilations = [dilations] if isinstance(dilations, int) else dilations
    not_valid_pad, padding_list, output_padding = _pad_before_conv_tranpose(
        x, filters, strides, padding, 1, dilations, output_shape, filters.shape[2:]
    )
    res = torch.nn.functional.conv_transpose1d(
        x,
        filters,
        bias,
        strides,
        padding_list,
        dilation=dilations,
        output_padding=output_padding,
    )
    if not_valid_pad[0]:
        res = res[:, :, 0:-1]
    if data_format == "NWC":
        res = res.permute(0, 2, 1)
    return res


@with_unsupported_dtypes(
    {"2.1.0 and below": ("float16", "bfloat16", "complex")},
    backend_version,
)
# noinspection PyUnresolvedReferences
def conv2d(
    x: torch.Tensor,
    filters: torch.Tensor,
    strides: Union[int, Tuple[int, int]],
    padding: Union[str, int, Sequence[Tuple[int, int]]],
    /,
    *,
    data_format: str = "NHWC",
    filter_format: str = "channel_last",
    x_dilations: Union[int, Tuple[int, int]] = 1,
    dilations: Union[int, Tuple[int, int]] = 1,
    bias: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if data_format == "NHWC":
        x = x.permute(0, 3, 1, 2)
    x, filters = _ff_xd_before_conv(x, filters, 2, filter_format, x_dilations)
    x, padding = _pad_before_conv(
        x, filters, strides, padding, 2, dilations, "channel_first"
    )
    res = torch.nn.functional.conv2d(x, filters, bias, strides, padding, dilations)
    if data_format == "NHWC":
        return res.permute(0, 2, 3, 1)
    return res


@with_unsupported_dtypes(
    {
        "2.1.0 and below": (
            "float16",
            "bfloat16",
            "complex",
        )
    },
    backend_version,
)
# noinspection PyUnresolvedReferences
def conv2d_transpose(
    x: torch.Tensor,
    filters: torch.Tensor,
    strides: Union[int, Tuple[int, int]],
    padding: str,
    /,
    *,
    output_shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    data_format: str = "NHWC",
    dilations: Union[int, Tuple[int, int]] = 1,
    bias: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
):
    """Transpose convolution (deconvolution) layer.
    
    Performs a 2D transpose convolution (deconvolution) on an input tensor, 
    using the provided filters, strides, padding and output padding. 
    
    Parameters
    ----------
    x : torch.Tensor
        Input tensor to perform transpose convolution on.
    filters : torch.Tensor 
        Convolution filters.  
    strides : Union[int, Tuple[int, int]]
        Stride sizes for the transpose convolution.
    padding : str
        Type of padding, either 'VALID' or 'SAME'.  
    output_shape : Optional[Union[ivy.NativeShape, Sequence[int]]]
        Shape of the output produced by the transpose convolution. If not provided,
        calculated automatically based on the input size, filter size and strides.  
    data_format : str
        Either 'NHWC' or 'NCHW' indicating the data format.
    dilations : Union[int, Tuple[int, int]]
        Dilation factors for the filters.  
    bias : Optional[torch.Tensor] 
        Optional bias tensor to add to the output. 
    out : Optional[torch.Tensor]
        Optional output tensor to write the result to.
    
    Returns
    -------
    torch.Tensor
        Tensor result of the 2D transpose convolution.
    
    """
    if data_format == "NHWC":
        x = x.permute(0, 3, 1, 2)
    strides = [strides] * 2 if isinstance(strides, int) else strides
    dilations = [dilations] * 2 if isinstance(dilations, int) else dilations
    filters = filters.permute(2, 3, 0, 1)
    not_valid_pad, padding_list, output_padding = _pad_before_conv_tranpose(
        x, filters, strides, padding, 2, dilations, output_shape, filters.shape[2:]
    )

    res = torch.nn.functional.conv_transpose2d(
        x,
        filters,
        bias,
        strides,
        padding_list,
        dilation=dilations,
        output_padding=output_padding,
    )
    if not_valid_pad[0]:
        res = res[..., :-1, :]
    if not_valid_pad[1]:
        res = res[..., :-1]
    if data_format == "NHWC":
        res = res.permute(0, *range(2, 4), 1)

    return res


@with_unsupported_dtypes(
    {
        "2.1.0 and below": (
            "float16",
            "bfloat16",
            "complex",
        )
    },
    backend_version,
)
# noinspection PyUnresolvedReferences
def depthwise_conv2d(
    x: torch.Tensor,
    filters: torch.Tensor,
    strides: Union[int, Tuple[int, int]],
    padding: Union[str, int, Sequence[Tuple[int, int]]],
    /,
    *,
    data_format: str = "NHWC",
    dilations: Union[int, Tuple[int, int]] = 1,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    strides = [strides] * 2 if isinstance(strides, int) else strides
    dilations = [dilations] * 2 if isinstance(dilations, int) else dilations
    if data_format == "NHWC":
        x = x.permute(0, 3, 1, 2)
    filters = ivy.squeeze(filters, 3).to_native() if filters.ndim == 4 else filters
    filters = torch.unsqueeze(filters, -1)
    dims_in = filters.shape[-2]
    filters = filters.permute(2, 3, 0, 1)
    x, padding = _pad_before_conv(
        x, filters, strides, padding, 2, dilations, "channel_first"
    )
    # noinspection PyArgumentEqualDefault
    res = torch.nn.functional.conv2d(
        x, filters, None, strides, padding, dilations, dims_in
    )
    if data_format == "NHWC":
        return res.permute(0, 2, 3, 1)
    return res


@with_unsupported_dtypes(
    {"2.1.0 and below": ("float16", "bfloat16", "complex")}, backend_version
)
# noinspection PyUnresolvedReferences
def conv3d(
    x: torch.Tensor,
    filters: torch.Tensor,
    strides: Union[int, Tuple[int, int, int]],
    padding: Union[str, int, Sequence[Tuple[int, int]]],
    /,
    *,
    data_format: str = "NDHWC",
    filter_format: str = "channel_last",
    x_dilations: Union[int, Tuple[int, int, int]] = 1,
    dilations: Union[int, Tuple[int, int, int]] = 1,
    bias: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
):
    if data_format == "NDHWC":
        x = x.permute(0, 4, 1, 2, 3)
    x, filters = _ff_xd_before_conv(x, filters, 3, filter_format, x_dilations)
    x, padding = _pad_before_conv(
        x, filters, strides, padding, 3, dilations, "channel_first"
    )
    res = torch.nn.functional.conv3d(x, filters, bias, strides, padding, dilations)
    if data_format == "NDHWC":
        res = res.permute(0, 2, 3, 4, 1)
    return res


@with_unsupported_dtypes(
    {"2.1.0 and below": ("float16", "bfloat16", "complex")},
    backend_version,
)
# noinspection PyUnresolvedReferences
def conv3d_transpose(
    x: torch.Tensor,
    filters: torch.Tensor,
    strides: Union[int, Tuple[int, int, int]],
    padding: str,
    /,
    *,
    output_shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    data_format: str = "NDHWC",
    dilations: Union[int, Tuple[int, int, int]] = 1,
    bias: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if data_format == "NDHWC":
        x = x.permute(0, 4, 1, 2, 3)
    strides = [strides] * 3 if isinstance(strides, int) else strides
    dilations = [dilations] * 3 if isinstance(dilations, int) else dilations
    filters = filters.permute(3, 4, 0, 1, 2)
    not_valid_pad, padding_list, output_padding = _pad_before_conv_tranpose(
        x, filters, strides, padding, 3, dilations, output_shape, filters.shape[2:]
    )
    res = torch.nn.functional.conv_transpose3d(
        x,
        filters,
        bias,
        strides,
        padding_list,
        dilation=dilations,
        output_padding=output_padding,
    )
    if not_valid_pad[0]:
        res = res[:, :, 0:-1, :, :]
    if not_valid_pad[1]:
        res = res[:, :, :, 0:-1, :]
    if not_valid_pad[2]:
        res = res[:, :, :, :, 0:-1]
    if data_format == "NDHWC":
        res = res.permute(0, 2, 3, 4, 1)
    return res


@with_unsupported_dtypes(
    {"2.1.0 and below": ("float16", "bfloat16", "complex")},
    backend_version,
)
def conv_general_dilated(
    x: torch.Tensor,
    filters: torch.Tensor,
    strides: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]],
    padding: Union[str, int, Sequence[Tuple[int, int]]],
    /,
    *,
    dims: int = 2,
    data_format: str = "channel_last",
    filter_format: str = "channel_last",
    feature_group_count: int = 1,
    x_dilations: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]] = 1,
    dilations: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]] = 1,
    bias: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
):
    
    # permuting dims based on formats
    if data_format == "channel_last":
        x = x.permute(0, dims + 1, *range(1, dims + 1))

    if filter_format == "channel_last":
        filters = filters.permute(-1, -2, *range(dims))

    # adding dilation to input
    x_dilations = [x_dilations] * dims if isinstance(x_dilations, int) else x_dilations
    for i in range(dims):
        if x_dilations[i] > 1:
            h = x.shape[2 + i]
            new_height = h + (h - 1) * (x_dilations[i] - 1)
            h = torch.eye(
                new_height,
                dtype=x.dtype,
                device=ivy.as_native_dev(ivy.default_device()),
            )[:: x_dilations[i]]
            x = torch.swapaxes(x, 2 + i, -1)
            x = torch.matmul(x, h)
            x = torch.swapaxes(x, -1, 2 + i)

    x, padding = _pad_before_conv(
        x, filters, strides, padding, dims, dilations, "channel_first"
    )

    if dims == 1:
        res = torch.nn.functional.conv1d(
            x, filters, bias, strides, padding, dilations, feature_group_count
        )
    elif dims == 2:
        res = torch.nn.functional.conv2d(
            x, filters, bias, strides, padding, dilations, feature_group_count
        )
    else:
        res = torch.nn.functional.conv3d(
            x, filters, bias, strides, padding, dilations, feature_group_count
        )
    if data_format == "channel_last":
        return res.permute(0, *range(2, dims + 2), 1)
    return res


@with_unsupported_dtypes(
    {"2.1.0 and below": ("float16", "bfloat16", "complex")},
    backend_version,
)
def conv_general_transpose(
    x: torch.Tensor,
    filters: torch.Tensor,
    strides: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]],
    padding: str,
    /,
    *,
    dims: int = 2,
    output_shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    data_format: str = "NDHWC",
    dilations: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]] = 1,
    feature_group_count: int = 1,
    bias: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
):
    
    if data_format == "channel_last":
        x = x.permute(0, dims + 1, *range(1, dims + 1))
    strides = [strides] * dims if isinstance(strides, int) else strides
    dilations = [dilations] * dims if isinstance(dilations, int) else dilations
    filters = filters.permute(dims, dims + 1, *range(dims))
    not_valid_pad, padding_list, output_padding = _pad_before_conv_tranpose(
        x, filters, strides, padding, dims, dilations, output_shape, filters.shape[2:]
    )
    if dims == 1:
        res = torch.nn.functional.conv_transpose1d(
            x,
            filters,
            bias,
            strides,
            padding_list,
            dilation=dilations,
            output_padding=output_padding,
            groups=feature_group_count,
        )
        if not_valid_pad[0]:
            res = res[:, :, :-1]
    elif dims == 2:
        res = torch.nn.functional.conv_transpose2d(
            x,
            filters,
            bias,
            strides,
            padding_list,
            dilation=dilations,
            output_padding=output_padding,
            groups=feature_group_count,
        )
        if not_valid_pad[0]:
            res = res[..., :-1, :]
        if not_valid_pad[1]:
            res = res[..., :-1]
    else:
        res = torch.nn.functional.conv_transpose3d(
            x,
            filters,
            bias,
            strides,
            padding_list,
            dilation=dilations,
            output_padding=output_padding,
            groups=feature_group_count,
        )
        if not_valid_pad[0]:
            res = res[..., :-1, :, :]
        if not_valid_pad[1]:
            res = res[..., :, :-1, :]
        if not_valid_pad[2]:
            res = res[..., :, :, :-1]
    if data_format == "channel_last":
        res = res.permute(0, *range(2, dims + 2), 1)
    return res


def scaled_dot_product_attention_v_2p0p0_and_above(
    q,
    k,
    v,
    scale: float,
    /,
    *,
    mask=None,
    out=None,
):
    """
    Performs scaled dot-product attention.
    
    Scaled dot-product attention is an attention mechanism that takes 
    in a query, key and value, and returns an attention weight representing
    the relevance of each value to the query. This implementation includes
    optional masking and scaling.
    
    This version is for PyTorch 2.0.0 and above.
    
    Parameters
    ----------
    q : torch.Tensor
        Query tensor of shape (..., query_seq_len, dk).
    k : torch.Tensor
        Key tensor of shape (..., key_seq_len, dk).  
    v : torch.Tensor
        Value tensor of shape (..., key_seq_len, dv).
    scale : float
        Scaling factor for the dot products.   
    mask : torch.Tensor, optional
        Mask tensor applied to the dot product scores before softmax, shape
        (..., query_seq_len, key_seq_len).
    out : torch.Tensor, optional
        Output tensor.
            
    Returns
    -------
    torch.Tensor
        Attention weight tensor of shape (..., query_seq_len, key_seq_len).
    
    """
    pass


merf.support_native_out = True


@with_unsupported_dtypes({"2.1.0 and below": ("complex",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def minimum(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    /,
    *,
    use_where: bool = True,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    if use_where:
        return torch.where(x1 <= x2, x1, x2, out=out)
    return torch.minimum(x1, x2, out=out)


minimum.support_native_out = True


@with_unsupported_dtypes({"2.1.0 and below": ("complex",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def maximum(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    /,
    *,
    use_where: bool = True,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Computes the maximum between two arrays elementwise.
    
    Parameters
    ----------
    x1 : array_like
        First array to compare.
    x2 : array_like 
        Second array to compare. Must be compatible with x1 (see Broadcasting).
    use_where : bool, optional
        If True, calculates the maximum using a where function. Otherwise uses torch.maximum. Default is True.  
    out : ndarray, optional
        Output array. Must be compatible with the expected output.
    
    Returns
    -------
    ret : ndarray or scalar
        The maximum of x1 and x2, element-wise. Returns scalar if both x1 and x2 are scalars.
    
    Examples
    --------
    >>> x1 = [1, 4 ,7]
    >>> x2 = [3, 2, 2]
    >>> maximum(x1, x2)
    [3, 4, 7]
    """
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    if use_where:
        return torch.where(x1 >= x2, x1, x2, out=out)
    return torch.maximum(x1, x2, out=out)


maximum.support_native_out = True


@with_unsupported_dtypes({"2.1.0 and below": ("float16",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def reciprocal(
    x: Union[float, torch.Tensor], /, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Computes the reciprocal of the input tensor element-wise.
    
    Parameters
    ----------
    x : Tensor
        The input tensor.
    out : Tensor, optional
        Optional output tensor to store the result.    
    
    Returns
    -------
    ret : Tensor
        A tensor containing the reciprocal of the input tensor.
    
    """
    x = _cast_for_unary_op(x)
    return torch.reciprocal(x, out=out)


reciprocal.support_native_out = True


@with_unsupported_dtypes(
    {"2.1.0 and below": ("complex64", "complex128")}, backend_version
)
@handle_numpy_arrays_in_specific_backend
def deg2rad(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Converts angles from degrees to radians element-wise.
    
    Parameters
    ----------
    x : array_like
        Input array in degrees.
        
    out : optional
        A location into which the result is stored. If provided, it must have a shape that the 
        inputs broadcast to. If not provided or None, a freshly-allocated array is returned.
        
    Returns
    -------
    ret : ndarray
        The values in radians. This is a scalar if x is a scalar.
    
    """
    return torch.deg2rad(x, out=out)


deg2rad.support_native_out = True


@with_unsupported_dtypes(
    {"2.1.0 and below": ("complex64", "complex128")}, backend_version
)
@handle_numpy_arrays_in_specific_backend
def rad2deg(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Converts angles from radians to degrees.
    
    Parameters
    ----------
    x : Tensor
        Input tensor in radians.
    out : Tensor, optional
        Output tensor.
    
    Returns
    -------
    Tensor
        A tensor containing the angles in degrees.
    
    Examples
    --------
    >>> rad2deg(3.14) 
    180.0
    """
    return torch.rad2deg(x, out=out)


rad2deg.support_native_out = True


@with_unsupported_dtypes({"2.1.0 and below": ("complex",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def trunc_divide(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Divides the first input array by the second input array element-wise. 
    
    Supports broadcasting. Promotes inputs to a common dtype.
    
    Parameters
    ----------
    x1 : array_like
        Numerator array. 
    x2 : array_like 
        Denominator array.
    out : Optional[torch.Tensor], optional
        Output tensor. If provided, the result will be placed in this tensor. 
        Default is None.
    
    Returns
    -------
    ret : torch.Tensor
        Element-wise division result.
    
    Raises
    ------
    ZeroDivisionError
        If attempting to divide by zero.
    """
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    ret = torch.div(x1, x2, rounding_mode="trunc")
    if ivy.is_float_dtype(x1.dtype):
        ret = ret.to(x1.dtype)
    else:
        ret = ret.to(ivy.default_float_dtype(as_native=True))
    return ret


@handle_numpy_arrays_in_specific_backend
def isreal(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Checks element-wise if input contains exclusively real values.
    
    Parameters
    ----------
    x : array_like
        Input array. 
    
    Returns
    -------
    ret : ndarray
        Boolean array of same shape as x indicating whether each element 
        is real.
    
    Examples
    --------
    >>> x = np.array([1+1j, 1+0j, 4.5, 3, 2, 2j])
    >>> np.isreal(x)
    array([False,  True,  True,  True,  True, False])
    """
    return torch.isreal(x)


@with_unsupported_dtypes(
    {"2.1.0 and below": ("bfloat16", "complex")},
    backend_version,
)
@handle_numpy_arrays_in_specific_backend
def fmod(
    x1: torch.Tensor,
    x2: torch.Tensor,
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Calculates the remainder of dividing x1 by x2 element-wise. 
    
    Parameters
    ----------
    x1 : array_like
        Dividend input array.
    x2 : array_like  
        Divisor input array.
    out : Optional[torch.Tensor], optional
        Output array. If provided, the result will be placed in this array.
    
    Returns
    -------
    tensor
        The remainder of dividing x1 by x2.
        
    """
    x1, x2 = promote_types_of_inputs(x1, x2)
    return torch.fmod(x1, x2, out=None)


fmod.support_native_out = True


def gcd(
    x1: Union[torch.Tensor, int, list, tuple],
    x2: Union[torch.Tensor, float, list, tuple],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Calculates the greatest common divisor between two numbers or tensors elementwise.
    
    Parameters
    ----------
    x1 : int, float, tensor-like
        The first input array or number.
    x2 : int, float, tensor-like        
        The second input array or number.
    out : tensor or None, optional          
        A location in which to store the results. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a fresh tensor is
        allocated. 
        
    Returns
    -------
    ret : tensor
        The greatest common divisor of the elements of x1 and x2.
        
    Examples
    --------
    >>> gcd(12, 18)
    6
    
    >>> x1 = [12, 18, 33]  
    >>> x2 = [6, 9, 15]
    >>> gcd(x1, x2)
    array([6, 9, 3])
    """
    x1, x2 = promote_types_of_inputs(x1, x2)
    return torch.gcd(x1, x2, out=out)


gcd.support_native_out = True


def angle(
    input: torch.Tensor,
    /,
    *,
    deg: Optional[bool] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Calculates the angle of the complex input array in radians or degrees.
    
    Parameters
    ----------
    input : array_like
        Input array.
    deg : bool, optional
        If True, returns the angle in degrees instead of radians. Default is radians.
    out : ndarray, optional
        A location in which to store the results. If provided, it must have a shape that the 
        inputs broadcast to. If not provided or None, a freshly-allocated array is returned.
    
    Returns
    -------
    angle_array : ndarray
        An array containing the angle of each element in radians, unless deg=True.
        If deg=True, the angles are returned in degrees.
    
    """
    if deg:
        return torch.angle(input, out=out) * (180 / pi)
    else:
        return torch.angle(input, out=out)


angle.support_native_out = True


def nan_to_num(
    x: torch.Tensor,
    /,
    *,
    copy: bool = True,
    nan: Union[float, int] = 0.0,
    posinf: Optional[Union[float, int]] = None,
    neginf: Optional[Union[float, int]] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if copy:
        return torch.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf, out=out)
    else:
        x = torch.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)
        return x


def real(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Returns the real part of the complex argument.
    
    Parameters
    ----------
    x : array_like
        Input array.
    
    out : array_like, optional
        Optional output array to place results in. Must be of the same shape 
        and dtype as the expected output.
    
    Returns
    -------
    real_x : array_like
        The real component of the complex argument. If `x` is real, the type 
        of `real_x` is float.  If `x` is complex, the type of `real_x` is 
        the same as `x.real`.
    
    Examples
    --------
    >>> x = 3 + 4j
    >>> real(x)
    3.0
    """
    return torch.real(x)

@with_unsupported_dtypes(
    {"2.5.1 and below": ("float16", "int16", "int8")}, backend_version
)
def get_item(
    x: paddle.Tensor,
    /,
    query: Union[paddle.Tensor, Tuple],
    *,
    copy: bool = None,
) -> paddle.Tensor:
    pass


def gather_nd(
    params: paddle.Tensor,
    indices: paddle.Tensor,
    /,
    *,
    batch_dims: Optional[int] = 0,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    """gather_nd implementation with batch support."""
    ivy.utils.assertions.check_gather_nd_input_valid(params, indices, batch_dims)
    if not isinstance(batch_dims, int):
        raise TypeError(f"Argument `batch_dims` must be an int; got {batch_dims}")
    if batch_dims < 0:
        raise ValueError("gather_nd does not allow negative batch_dims.")
    params_ndims = params.ndim
    indices_ndims = indices.ndim
    if indices_ndims is not None and batch_dims >= indices_ndims:
        raise ValueError(
            f"Argument `batch_dims` = {batch_dims} must be "
            f"less than rank(`indices`) = {indices_ndims}"
        )
    if params_ndims is not None and batch_dims >= params_ndims:
        raise ValueError(
            f"Argument `batch_dims` = {batch_dims} must be "
            f"less than rank(`params`) = {params_ndims}"
        )
    expand = batch_dims == 0
    if expand:
        # Normally gather_nd will be called when batch_dims == 0.
        # But if this function is called with batch_dims = 0, e.g. for testing
        # purposes, this adds a dummy batch dimension to make batch_dims = 1.
        params = paddle_backend.expand_dims(params, axis=0)
        indices = paddle_backend.expand_dims(indices, axis=0)
        batch_dims = 1

    if indices.dtype not in [paddle.int32, paddle.int64]:
        indices = indices.cast(paddle.int32)

    params_shape = paddle.to_tensor(params.shape)
    indices_shape = indices.shape
    batch_shape = params_shape[:batch_dims]
    batch_size = paddle.prod(batch_shape, [0]).numpy().tolist()
    index_internal_ndims = indices.ndim - batch_dims - 1
    indices_internal_shape = indices_shape[batch_dims:-1]

    # Assuming a 'params' with shape [b1, ..., bM, g1, ..., gN] and an 'indices'
    # with shape [b1, ..., bM, i1, ..., iK, C], where C <= N, we need to modify
    # 'indices' s.t. it has shape [i1, ..., iK, D], where D <= M + N and slices
    # to the entire 'params' tensor.
    # Assuming we have a batch of shape [B1, B2], we use meshgrid to create a
    # grid of size B1 x B2.
    batch_dim_list = paddle_backend.unstack(batch_shape, axis=0)
    dim_ranges = [
        paddle.arange(0, x.item(), 1, dtype=indices.dtype) for x in batch_dim_list
    ]
    if dim_ranges:
        if len(dim_ranges) > 1:
            mesh_list = paddle_backend.meshgrid(*dim_ranges, indexing="ij")
        else:
            mesh_list = dim_ranges
    else:
        mesh_list = []
    # Then we flatten and stack the tensors to form a (B1.B2) by 2 matrix.
    flat_list = [paddle_backend.reshape(x, shape=(-1,)) for x in mesh_list]
    stacked_list = (
        paddle_backend.stack(flat_list, axis=0) if flat_list else paddle.to_tensor([])
    )
    index_grid = paddle_backend.permute_dims(
        stacked_list, axes=[axis for axis in range(stacked_list.ndim)][::-1]
    )
    # We need to concatenate these batch coordinates with the internal indices.
    # concat -> index_grid [B1.B2, 2] with indices [i1, ..., iK, C]
    # So we reshape them both to [(B1.B2), i1, ..., iK, *]
    index_grid_shape = index_grid.shape
    index_grid = paddle_backend.reshape(
        index_grid,
        index_grid_shape[:1]
        + [
            1,
        ]
        * index_internal_ndims
        + index_grid_shape[1:],
    )
    tile_shape = (
        [
            1,
        ]
        + indices_internal_shape
        + [
            1,
        ]
    )
    index_grid = paddle_backend.tile(index_grid, repeats=paddle.to_tensor(tile_shape))
    # index_grid now has shape [(B1.B2), i1, ..., iK, 2]
    flat_shape = batch_size + indices_shape[batch_dims:]
    flat_indices = paddle_backend.reshape(indices, shape=flat_shape)
    # flat_indices now has shape [(B1.B2), i1, ..., iK, C]
    indices = paddle_backend.concat((index_grid, flat_indices), axis=-1)
    # indices has shape [(B1.B2), i1, ..., iK, 2+C]
    if params.dtype in [
        paddle.int8,
        paddle.float16,
        paddle.complex64,
        paddle.complex128,
    ]:
        if paddle.is_complex(params):
            out = paddle.complex(
                paddle.gather_nd(params.real(), indices),
                paddle.gather_nd(params.imag(), indices),
            )
        else:
            out = paddle.gather_nd(params.cast("float32"), indices).cast(params.dtype)
    else:
        out = paddle.gather_nd(params, indices)
    # out has shape [(B1.B2), i1, ..., iK, N-C]. Now we reshape batch to
    # its original form.
    out_shape = out.shape
    out = paddle_backend.reshape(out, shape=batch_shape.tolist() + out_shape[1:])
    if expand:
        out = paddle_backend.squeeze(out, axis=0)
    return out