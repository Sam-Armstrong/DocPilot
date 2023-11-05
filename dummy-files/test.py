# global
import math
from numbers import Number
from typing import Union, Tuple, Optional, List, Sequence

import numpy as np
import tensorflow as tf

# local
import ivy

# noinspection PyProtectedMember
from ivy.func_wrapper import with_supported_dtypes, with_unsupported_dtypes
from ivy.functional.ivy.manipulation import _calculate_out_shape
from . import backend_version


def _reshape_fortran_tf(x, shape):
    """
    Reshapes the input array x into the shape provided in the shape argument. 
    
    Parameters
    ----------
    x : tf.Tensor or tf.Variable
        Input tensor to be reshaped.
    shape : Sequence[int]
        The new shape of the input array.
    copy : bool, optional
        Default is None. Whether to copy the input array or operate in-place.
    order : {'C', 'F'}, optional
        Default is 'C'. Whether to reshape in C-order (row-major) or Fortran-order (column-major).
    allowzero : bool, optional
        Default is True. Whether to allow one of the shape dimensions to be zero. 
        If False, a zero dimension raises an error.
    out : tf.Tensor or tf.Variable, optional 
        Default is None. Output tensor.
    
    Returns
    -------
    ret : tf.Tensor or tf.Variable
        Reshaped input tensor.
    
    Raises
    ------
    ivy.exceptions.IvyIndexError
        If input shape contains a zero dimension when allowzero=False.
    """
    if len(x.shape) > 0:
        x = tf.transpose(x)
    return tf.transpose(tf.reshape(x, shape[::-1]))


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
    """
    Reshapes the input array x into the shape provided in the shape argument. 
    
    Parameters
    ----------
    x : tf.Tensor or tf.Variable
        Input tensor to be reshaped.
    shape : Sequence[int]
        The new shape of the input array.
    copy : bool, optional
        Default is None. Whether to copy the input array or operate in-place.
    order : {'C', 'F'}, optional
        Default is 'C'. Whether to reshape in C-order (row-major) or Fortran-order (column-major).
    allowzero : bool, optional
        Default is True. Whether to allow one of the shape dimensions to be zero. 
        If False, a zero dimension raises an error.
    out : tf.Tensor or tf.Variable, optional 
        Default is None. Output tensor.
    
    Returns
    -------
    ret : tf.Tensor or tf.Variable
        Reshaped input tensor.
    
    Raises
    ------
    ivy.exceptions.IvyIndexError
        If input shape contains a zero dimension when allowzero=False.
    """
    ivy.utils.assertions.check_elem_in_list(order, ["C", "F"])
    if not allowzero:
        shape = [
            new_s if con else old_s
            for new_s, con, old_s in zip(shape, tf.constant(shape) != 0, x.shape)
        ]
    if order == "F":
        return _reshape_fortran_tf(x, shape)
    return tf.reshape(x, shape)




