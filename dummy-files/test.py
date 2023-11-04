# global
from numbers import Number
import numpy as np
from typing import Union, Optional, List, Sequence, Tuple

import jax.dlpack
import jax.numpy as jnp
import jax._src as _src
import jaxlib.xla_extension
import tensorflow as tf

# local
import ivy
from ivy import as_native_dtype
from ivy.functional.backends.jax import JaxArray
from ivy.functional.ivy.creation import (
    _asarray_to_native_arrays_and_back,
    _asarray_infer_device,
    _asarray_infer_dtype,
    _asarray_handle_nestable,
    NestedSequence,
    SupportsBufferProtocol,
    _asarray_inputs_to_native_shapes,
)


# Array API Standard #
# ------------------ #

@_asarray_to_native_arrays_and_back
@_asarray_infer_device
@_asarray_handle_nestable
@_asarray_inputs_to_native_shapes
@_asarray_infer_dtype
def asarray(
    obj: Union[
        JaxArray,
        bool,
        int,
        float,
        tuple,
        NestedSequence,
        SupportsBufferProtocol,
        np.ndarray,
    ],
    /,
    *,
    copy: Optional[bool] = None,
    dtype: Optional[jnp.dtype] = None,
    device: jaxlib.xla_extension.Device = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    ivy.utils.assertions._check_jax_x64_flag(dtype)
    if copy is True:
        return jnp.array(obj, dtype=dtype, copy=True)
    else:
        return jnp.asarray(obj, dtype=dtype)
    
def arange(
    start: float,
    /,
    stop: Optional[float] = None,
    step: float = 1,
    *,
    dtype: Optional[jnp.dtype] = None,
    device: jaxlib.xla_extension.Device = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    if dtype:
        dtype = as_native_dtype(dtype)
        ivy.utils.assertions._check_jax_x64_flag(dtype.name)
    res = jnp.arange(start, stop, step, dtype=dtype)
    if not dtype:
        if res.dtype == jnp.float64:
            return res.astype(jnp.float32)
        elif res.dtype == jnp.int64:
            return res.astype(jnp.int32)
    return res

def meshgrid(
    *arrays: JaxArray,
    sparse: bool = False,
    indexing: str = "xy",
    out: Optional[JaxArray] = None,
) -> List[JaxArray]:
    return jnp.meshgrid(*arrays, sparse=sparse, indexing=indexing)

def broadcast_arrays(
    *arrays: Union[tf.Tensor, tf.Variable],
) -> List[Union[tf.Tensor, tf.Variable]]:
    if len(arrays) > 1:
        try:
            desired_shape = tf.broadcast_dynamic_shape(arrays[0].shape, arrays[1].shape)
        except tf.errors.InvalidArgumentError as e:
            raise ivy.utils.exceptions.IvyBroadcastShapeError(e)
        if len(arrays) > 2:
            for i in range(2, len(arrays)):
                try:
                    desired_shape = tf.broadcast_dynamic_shape(
                        desired_shape, arrays[i].shape
                    )
                except tf.errors.InvalidArgumentError as e:
                    raise ivy.utils.exceptions.IvyBroadcastShapeError(e)
    else:
        return [arrays[0]]
    result = []
    for tensor in arrays:
        result.append(tf.broadcast_to(tensor, desired_shape))

    return result

@with_unsupported_dtypes({"2.13.0 and below": ("uint16",)}, backend_version)
def eye(
    n_rows: int,
    n_cols: Optional[int] = None,
    /,
    *,
    k: int = 0,
    batch_shape: Optional[Union[int, Sequence[int]]] = None,
    dtype: tf.DType,
    device: str = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if n_cols is None:
        n_cols = n_rows
    if batch_shape is None:
        batch_shape = []
    i = tf.eye(n_rows, n_cols, dtype=dtype)
    reshape_dims = [1] * len(batch_shape) + [n_rows, n_cols]
    tile_dims = list(batch_shape) + [1, 1]

    # k=index of the diagonal. A positive value refers to an upper diagonal,
    # a negative value to a lower diagonal, and 0 to the main diagonal.
    # Default: ``0``.
    # value of k ranges from -n_rows < k < n_cols

    # k=0 refers to the main diagonal
    if k == 0:
        return tf.eye(n_rows, n_cols, batch_shape=batch_shape, dtype=dtype)

    # when k is negative
    elif -n_rows < k < 0:
        mat = tf.concat(
            [tf.zeros([-k, n_cols], dtype=dtype), i[: n_rows + k]],
            0,
        )
        return tf.tile(tf.reshape(mat, reshape_dims), tile_dims)

    elif 0 < k < n_cols:
        mat = tf.concat(
            [
                tf.zeros([n_rows, k], dtype=dtype),
                i[:, : n_cols - k],
            ],
            1,
        )
        return tf.tile(tf.reshape(mat, reshape_dims), tile_dims)
    else:
        return tf.zeros(batch_shape + [n_rows, n_cols], dtype=dtype)

def to_dlpack(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
):
    if isinstance(x, tf.Variable):
        x = x.read_value()
    dlcapsule = tf.experimental.dlpack.to_dlpack(x)
    return dlcapsule

def asin(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.asin(x)


def asinh(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.asinh(x)


def atan(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.math.atan(x)

@with_unsupported_dtypes({"2.13.0 and below": ("complex",)}, backend_version)
def atan2(
    x1: Union[tf.Tensor, tf.Variable],
    x2: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return tf.math.atan2(x1, x2)


def atanh(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.math.atanh(x)

def concat(
    xs: Union[Tuple[tf.Tensor, ...], List[tf.Tensor]],
    /,
    *,
    axis: int = 0,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    is_tuple = type(xs) is tuple
    is_axis_none = axis is None
    if is_tuple:
        xs = list(xs)
    highest_dtype = xs[0].dtype
    for i in xs:
        highest_dtype = ivy.as_native_dtype(ivy.promote_types(highest_dtype, i.dtype))

    for i in range(len(xs)):
        if is_axis_none:
            xs[i] = tf.reshape(xs[i], -1)
        xs[i] = ivy.astype(xs[i], highest_dtype, copy=False).to_native()
    if is_axis_none:
        axis = 0
        if is_tuple:
            xs = tuple(xs)
    try:
        return tf.concat(xs, axis)
    except (tf.errors.InvalidArgumentError, np.AxisError) as error:
        raise ivy.utils.exceptions.IvyIndexError(error)


def expand_dims(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    copy: Optional[bool] = None,
    axis: Union[int, Sequence[int]] = 0,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    try:
        out_shape = _calculate_out_shape(axis, x.shape)
        ret = tf.reshape(x, shape=out_shape)
        return ret
    except (tf.errors.InvalidArgumentError, np.AxisError) as error:
        raise ivy.utils.exceptions.IvyIndexError(error)