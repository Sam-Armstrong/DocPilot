"""
TensorFlow random functions.

Collection of TensorFlow random functions, wrapped to fit Ivy syntax and
signature.
"""

from typing import Optional, Union, Sequence

# global
import tensorflow as tf
from tensorflow.python.framework.dtypes import DType

# local
import ivy
from ivy.func_wrapper import with_unsupported_dtypes, with_supported_dtypes
from ivy.functional.ivy.random import (
    _check_bounds_and_get_shape,
    _randint_check_dtype_and_bound,
    _check_valid_scale,
)
from . import backend_version


# Extra #
# ------#


@with_supported_dtypes(
    {"2.13.0 and below": ("float", "int32", "int64")}, backend_version
)
def random_uniform(
    *,
    low: Union[float, tf.Tensor, tf.Variable] = 0.0,
    high: Union[float, tf.Tensor, tf.Variable] = 1.0,
    shape: Optional[Union[ivy.NativeShape, Sequence[int], tf.Tensor]] = None,
    dtype: DType,
    device: str = None,
    seed: Optional[int] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    shape = _check_bounds_and_get_shape(low, high, shape).shape
    low = tf.cast(low, dtype)
    high = tf.cast(high, dtype)
    if seed:
        tf.random.set_seed(seed)
    return tf.random.uniform(shape, low, high, dtype=dtype, seed=seed)


def random_normal(
    *,
    mean: Union[float, tf.Tensor, tf.Variable] = 0.0,
    std: Union[float, tf.Tensor, tf.Variable] = 1.0,
    shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    dtype: DType,
    seed: Optional[int] = None,
    device: str = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    _check_valid_scale(std)
    shape = _check_bounds_and_get_shape(mean, std, shape).shape
    mean = tf.cast(mean, dtype)
    std = tf.cast(std, dtype)
    if seed:
        tf.random.set_seed(seed)
    return tf.random.normal(shape, mean, std, dtype=dtype, seed=seed)


@with_unsupported_dtypes({"2.13.0 and below": ("bfloat16",)}, backend_version)
def multinomial(
    population_size: int,
    num_samples: int,
    /,
    *,
    batch_size: int = 1,
    probs: Optional[Union[tf.Tensor, tf.Variable]] = None,
    replace: bool = True,
    device: str = None,
    seed: Optional[int] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if probs is None:
        probs = (
            tf.ones(
                (
                    batch_size,
                    population_size,
                )
            )
            / population_size
        )

    # We set the global seed, but not the operation seeds below. In this way, we
    # get different results for every random op call but the same sequence for
    # every re-run of the program
    if seed:
        tf.random.set_seed(seed)

    if not replace:
        orig_probs_shape = list(probs.shape)
        probs_flat = tf.reshape(probs, (-1, orig_probs_shape[-1]))
        probs_flat = probs_flat / tf.math.reduce_sum(probs_flat, axis=-1, keepdims=True)
        probs_stack = tf.split(probs_flat, probs_flat.shape[0])
        samples_stack = []
        for prob in probs_stack:
            logits = tf.dtypes.cast(tf.math.log(prob), tf.float64)
            # Gumbel-max trick
            # https://github.com/tensorflow/tensorflow/issues/9260
            z = tf.dtypes.cast(
                -tf.math.log(-tf.math.log(tf.random.uniform(tf.shape(logits), 0, 1))),
                tf.float64,
            )
            _, indices = tf.nn.top_k(logits + z, k=num_samples)
            samples_stack.append(indices)
        samples_flat = tf.stack(samples_stack)
        return tf.convert_to_tensor(
            tf.reshape(samples_flat, orig_probs_shape[:-1] + [num_samples])
        )
    else:
        if len(probs.numpy().shape) == 1:
            probs = tf.expand_dims(probs, axis=0)
        return tf.random.categorical(tf.math.log(probs), num_samples)


def randint(
    low: Union[float, tf.Tensor, tf.Variable],
    high: Union[float, tf.Tensor, tf.Variable],
    /,
    *,
    shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    device: str = None,
    dtype: Optional[Union[DType, ivy.Dtype]] = None,
    seed: Optional[int] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if not dtype:
        dtype = ivy.default_int_dtype()
    dtype = ivy.as_native_dtype(dtype)
    _randint_check_dtype_and_bound(low, high, dtype)
    shape = _check_bounds_and_get_shape(low, high, shape).shape
    low = tf.cast(low, "float32")
    high = tf.cast(high, "float32")
    if seed:
        tf.random.set_seed(seed)
    return tf.cast(tf.random.uniform(shape, low, high, "float32", seed=seed), dtype)


def seed(*, seed_value: int = 0) -> None:
    tf.random.set_seed(seed_value)
    return


def shuffle(
    x: Union[tf.Tensor, tf.Variable],
    axis: Optional[int] = 0,
    /,
    *,
    seed: Optional[int] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if seed:
        tf.random.set_seed(seed)
    return tf.random.shuffle(x, seed=seed)


def argmin(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    axis: Optional[int] = None,
    keepdims: bool = False,
    dtype: Optional[tf.dtypes.DType] = None,
    select_last_index: bool = False,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    n_dims = tf.rank(x).numpy()
    if axis is None:
        x = tf.reshape(x, [-1])
    if select_last_index:
        x = tf.experimental.numpy.flip(x, axis=axis)
        ret = tf.argmin(x, axis=axis)
        if axis is not None:
            ret = x.shape[axis] - ret - 1
        else:
            ret = tf.size(x, out_type=tf.int64) - ret - 1
    else:
        ret = tf.argmin(x, axis=axis)

    if keepdims:
        if axis is None:
            ret = tf.reshape(ret, [1] * n_dims)
        else:
            ret = tf.expand_dims(ret, axis)

    return tf.cast(ret, dtype) if dtype is not None else ret


def where(
    condition: Union[tf.Tensor, tf.Variable],
    x1: Union[tf.Tensor, tf.Variable],
    x2: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return tf.cast(tf.experimental.numpy.where(condition, x1, x2), x1.dtype)

def empty(
    shape: Union[ivy.NativeShape, Sequence[int]],
    *,
    dtype: paddle.dtype,
    device: core.Place = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if isinstance(shape, int):
        shape = [shape]
    return paddle.empty(shape=shape).cast(dtype)


def empty_like(
    x: paddle.Tensor,
    /,
    *,
    dtype: paddle.dtype,
    device: core.Place = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    return paddle.empty(shape=x.shape).cast(dtype)

@with_unsupported_device_and_dtypes(
    {
        "2.5.1 and below": {
            "cpu": (
                "uint8",
                "int8",
                "int16",
                "float16",
                "complex64",
                "complex128",
                "bool",
            )
        }
    },
    backend_version,
)
def eye(
    n_rows: int,
    n_cols: Optional[int] = None,
    /,
    *,
    k: int = 0,
    batch_shape: Optional[Union[int, Sequence[int]]] = None,
    dtype: paddle.dtype,
    device: core.Place = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if n_cols is None:
        n_cols = n_rows
    if batch_shape is None:
        batch_shape = []
    i = paddle.eye(n_rows, n_cols, dtype=dtype)
    reshape_dims = [1] * len(batch_shape) + [n_rows, n_cols]
    tile_dims = list(batch_shape) + [1, 1]

    # handle index of the diagonal k
    if k == 0:
        return paddle.reshape(i, reshape_dims)

    elif -n_rows < k < 0:
        mat = paddle.concat(
            [
                paddle.zeros([-k, n_cols], dtype=dtype),
                i[: n_rows + k],
            ],
            0,
        )
        return paddle.tile(paddle.reshape(mat, reshape_dims), tile_dims)

    elif 0 < k < n_cols:
        mat = paddle.concat(
            [
                paddle.zeros([n_rows, k], dtype=dtype),
                i[:, : n_cols - k],
            ],
            1,
        )
        return paddle.tile(paddle.reshape(mat, reshape_dims), tile_dims)
    else:
        return paddle.zeros(batch_shape + [n_rows, n_cols], dtype=dtype)


def from_dlpack(x, /, *, out: Optional[paddle.Tensor] = None):
    return paddle.utils.dlpack.from_dlpack(x)


def full(
    shape: Union[ivy.NativeShape, Sequence[int]],
    fill_value: Union[int, float, bool],
    *,
    dtype: Optional[Union[ivy.Dtype, paddle.dtype]] = None,
    device: core.Place = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if dtype is None:
        dtype = ivy.default_dtype(item=fill_value)
    if not isinstance(shape, Sequence):
        shape = [shape]
    if isinstance(fill_value, complex):
        fill_value = paddle.to_tensor(fill_value)
        ret_real = paddle.full(shape=shape, fill_value=fill_value.real())
        ret_imag = paddle.full(shape=shape, fill_value=fill_value.imag())
        ret = paddle.complex(ret_real, ret_imag)
    else:
        dtype_ = None if ivy.as_native_dtype(dtype) == paddle.int8 else dtype
        ret = paddle.full(shape=shape, fill_value=fill_value, dtype=dtype_)
    if ret.dtype != ivy.as_native_dtype(dtype):
        return ret.cast(dtype)
    return ret


def flip(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    copy: Optional[bool] = None,
    axis: Optional[Union[int, Sequence[int]]] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    num_dims = len(x.shape)
    if not num_dims:
        ret = x
    else:
        if axis is None:
            new_axis = list(range(num_dims))
        else:
            new_axis = axis
        if type(new_axis) is int:
            new_axis = [new_axis]
        else:
            new_axis = new_axis
        new_axis = [item + num_dims if item < 0 else item for item in new_axis]
        ret = tf.reverse(x, new_axis)
    return ret


def permute_dims(
    x: Union[tf.Tensor, tf.Variable],
    /,
    axes: Tuple[int, ...],
    *,
    copy: Optional[bool] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.transpose(x, perm=axes)


def to_dlpack(x, /, *, out: Optional[paddle.Tensor] = None):
    return paddle.utils.dlpack.to_dlpack(x)


def _differentiable_linspace(start, stop, num, *, dtype=None):
    start = ivy.to_native(start)
    num = paddle.to_tensor(num, stop_gradient=False)
    if num == 1:
        return paddle_backend.expand_dims(start, axis=0)
    n_m_1 = paddle_backend.subtract(num, 1)
    increment = paddle_backend.divide(paddle_backend.subtract(stop, start), n_m_1)
    increment_tiled = paddle_backend.repeat(increment, n_m_1)
    increments = paddle_backend.multiply(
        increment_tiled,
        paddle.linspace(1, n_m_1, n_m_1.cast(paddle.int32), dtype=dtype),
    )
    if isinstance(start, int) or start.ndim == 0:
        start = paddle_backend.expand_dims(start, axis=0)
    res = paddle_backend.concat((start, paddle_backend.add(start, increments)), axis=0)
    return res.cast(dtype)

@with_unsupported_device_and_dtypes(
    {"2.5.1 and below": {"cpu": ("uint16", "bfloat16", "float16")}}, backend_version
)
def linspace(
    start: Union[paddle.Tensor, float],
    stop: Union[paddle.Tensor, float],
    /,
    num: int,
    *,
    axis: Optional[int] = None,
    endpoint: bool = True,
    dtype: paddle.dtype,
    device: core.Place = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if not isinstance(start, (paddle.Tensor, int)):
        start = paddle.to_tensor(start)

    if not isinstance(start, (paddle.Tensor, int)):
        start = paddle.to_tensor(stop)

    if axis is None:
        axis = -1
    if not endpoint:
        if dtype is not None:
            ans = _linspace_helper(start, stop, num + 1, axis, dtype=dtype)
        else:
            ans = _linspace_helper(start, stop, num + 1, axis)
        if axis < 0:
            axis += len(ans.shape)
        ans = paddle_backend.get_item(ans, _slice_at_axis(slice(None, -1), axis))
    else:
        if dtype is not None:
            ans = _linspace_helper(start, stop, num, axis, dtype=dtype)
        else:
            ans = _linspace_helper(start, stop, num, axis)
    if (
        endpoint
        and ans.shape[0] > 1
        and (not isinstance(start, paddle.Tensor))
        and (not isinstance(stop, paddle.Tensor))
    ):
        ans[-1] = stop
    if (
        ans.shape[0] >= 1
        and (not isinstance(start, paddle.Tensor))
        and (not isinstance(stop, paddle.Tensor))
        and ans[0] != start
    ):
        ans[0] = start
    if ivy.is_ivy_array(ans):
        ans = paddle.to_tensor(ans.data)
    if "int" in str(dtype) and paddle.is_floating_point(ans):
        ans = paddle.floor(ans)
    return ans.cast(dtype)


def to_dlpack(x, /, *, out: Optional[paddle.Tensor] = None):
    return paddle.utils.dlpack.to_dlpack(x)


def flip(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    copy: Optional[bool] = None,
    axis: Optional[Union[int, Sequence[int]]] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    num_dims = len(x.shape)
    if not num_dims:
        ret = x
    else:
        if axis is None:
            new_axis = list(range(num_dims))
        else:
            new_axis = axis
        if type(new_axis) is int:
            new_axis = [new_axis]
        else:
            new_axis = new_axis
        new_axis = [item + num_dims if item < 0 else item for item in new_axis]
        ret = tf.reverse(x, new_axis)
    return ret


def permute_dims(
    x: Union[tf.Tensor, tf.Variable],
    /,
    axes: Tuple[int, ...],
    *,
    copy: Optional[bool] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.transpose(x, perm=axes)

@with_unsupported_dtypes({"2.13.0 and below": ("bool",)}, backend_version)
def reshape(
    x: Union[tf.Tensor, tf.Variable],
    /,
    shape: Union[ivy.NativeShape, Sequence[int]],
    *,
    copy: Optional[bool] = None,
    order: str = "C",
    allowzero: bool = True,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    ivy.utils.assertions.check_elem_in_list(order, ["C", "F"])
    if not allowzero:
        shape = [
            new_s if con else old_s
            for new_s, con, old_s in zip(shape, tf.constant(shape) != 0, x.shape)
        ]
    if order == "F":
        return _reshape_fortran_tf(x, shape)
    return tf.reshape(x, shape)




