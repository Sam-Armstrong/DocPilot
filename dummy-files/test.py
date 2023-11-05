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




@with_unsupported_dtypes({"0.4.19 and below": ("complex",)}, backend_version)
def diagonal(
    x: JaxArray,
    /,
    *,
    offset: int = 0,
    axis1: int = -2,
    axis2: int = -1,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    """
    Returns the diagonal of the input array.
    
    Parameters
    ----------
    x : array_like
        Input array from which the diagonals are taken.
    offset : int, optional
        Offset of the diagonal from the main diagonal. Default is 0. 
    axis1 : int, optional
        Axis to take the first diagonals from. Default is -2.
    axis2 : int, optional
        Axis to take the second diagonals from. Default is -1.
    
    Returns
    -------
    ret : ndarray
        The extracted diagonal or diagonals. 
    
    Raises
    ------
    IvyError
        If the input array has less than two dimensions.
    """
    if x.dtype != bool and not jnp.issubdtype(x.dtype, jnp.integer):
        ret = jnp.diagonal(x, offset=offset, axis1=axis1, axis2=axis2)
        ret_edited = jnp.diagonal(
            x.at[1 / x == -jnp.inf].set(-jnp.inf),
            offset=offset,
            axis1=axis1,
            axis2=axis2,
        )
        ret_edited = ret_edited.at[ret_edited == -jnp.inf].set(-0.0)
        ret = ret.at[ret == ret_edited].set(ret_edited[ret == ret_edited])
    else:
        ret = jnp.diagonal(x, offset=offset, axis1=axis1, axis2=axis2)
    return ret


def tensorsolve(
    x1: JaxArray,
    x2: JaxArray,
    /,
    *,
    axes: Optional[Union[int, Tuple[Sequence[int], Sequence[int]]]] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    """
    Solves the tensor equation ``a x = b`` for x.
    
    Parameters
    ----------
    a : array_like
        Coefficient tensor, of shape (M, N, ..., M, N)
    b : array_like
        Right-hand side tensor, of shape (M, N, ..., Q)
    axes : 2-tuple of lists of ints, optional
        Axes in `a` to apply solve to (for each vector in `b`).
        Default is ``(-2, -1)``
    
    Returns
    -------
    x : ndarray
        Solution tensor, shape ``(M, N, ..., Q)``.
    
    Raises
    ------
    LinAlgError
        If `a` is singular or not `a.shape[axis] == b.shape[axis]` for all 
        `axis` in ``axes[0]``
    """
    return jnp.linalg.tensorsolve(x1, x2, axes)


@with_unsupported_dtypes(
    {"0.4.19 and below": ("bfloat16", "float16", "complex")},
    backend_version,
)
def eigh(
    x: JaxArray, /, *, UPLO: str = "L", out: Optional[JaxArray] = None
) -> Tuple[JaxArray]:
    """Computes the eigenvalues and eigenvectors of a Hermitian or real symmetric matrix.
    
    Parameters
    ----------
    x : array_like
        Matrix whose eigenvalues and eigenvectors will be computed. 
    
    UPLO : {'L', 'U'}, optional
        Specifies whether the calculation is done with the lower triangular part of matrix ('L', default) or the upper triangular part ('U').
    
    out: tuple of arrays, optional
        Tuple of output arrays. The first array contains the eigenvalues and the second contains the eigenvectors.
    
    Returns
    -------
    eigenvalues : ndarray
        Array containing the eigenvalues of the input matrix.
    
    eigenvectors : ndarray
        Array containing the eigenvectors of the input matrix.
    
    """
    result_tuple = NamedTuple(
        "eigh", [("eigenvalues", JaxArray), ("eigenvectors", JaxArray)]
    )
    eigenvalues, eigenvectors = jnp.linalg.eigh(x, UPLO=UPLO)
    return result_tuple(eigenvalues, eigenvectors)


@with_unsupported_dtypes(
    {"0.4.19 and below": ("bfloat16", "float16", "complex")},
    backend_version,
)
def eigvalsh(
    x: JaxArray, /, *, UPLO: str = "L", out: Optional[JaxArray] = None
) -> JaxArray:
    """
    Computes the eigenvalues of a complex Hermitian or real symmetric matrix.
    
    Parameters
    ----------
    x : array_like
        Input array. Must be a square 2-D array.
    UPLO : {'L', 'U'}, optional
        Specifies whether the calculation is done with the lower triangular part of
        `x` ('L', default) or the upper triangular part ('U').
    out : ndarray, optional
        A location in which to store the results. If provided, it must have the same
        shape as the eigenvalues.
    
    Returns
    -------
    eigenvalues : ndarray
        The eigenvalues, each repeated according to its multiplicity.
    
    Raises
    ------
    LinAlgError
        If the eigenvalue computation does not converge.
    
    Examples
    --------
    >>> x = np.array([[1, -2j], [2j, 5]]) 
    >>> ivy.eigvalsh(x)
    array([ 0.+2.23606802j,  0.-2.23606802j])
    
    """
    return jnp.linalg.eigvalsh(x, UPLO=UPLO)


@with_unsupported_dtypes({"0.4.19 and below": ("complex",)}, backend_version)
def inner(x1: JaxArray, x2: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    """Computes the inner product of two arrays.
    
    Parameters
    ----------
    x1 : array_like
        First array to compute the inner product against.
    x2 : array_like
        Second array to compute the inner product against. 
    
    out : optional
        Output array. If not provided, a new array will be created.
    
    Returns
    -------
    ret : ndarray
        Inner product of `x1` and `x2`.
    
    Examples
    --------
    >>> x1 = [1, 2, 3]
    >>> x2 = [4, 5, 6]
    >>> inner(x1, x2)
    32
    
    """
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return jnp.inner(x1, x2)


