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

