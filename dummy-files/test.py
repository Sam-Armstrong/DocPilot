# global
from typing import Union, Optional
from math import pi
import torch

# local
import ivy
from ivy.func_wrapper import (
    with_unsupported_dtypes,
    with_supported_dtypes,
    handle_numpy_arrays_in_specific_backend,
)
from ivy import promote_types_of_inputs
from . import backend_version


def _cast_for_unary_op(x):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    return x


@handle_numpy_arrays_in_specific_backend
def add(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    /,
    *,
    alpha: Optional[Union[int, float]] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Adds two arrays element-wise. 
    
    Supports both scalar and array inputs and optionally allows specifying a 
    scaling factor.
    
    Parameters
    ----------
    x1: array_like
        First input array to add.
    x2: array_like 
        Second input array to add.
    alpha: int or float, optional
        Scaling factor for the addition. Default is 1.
    out: tensor, optional
        Output tensor. 
    
    Returns
    -------
    tensor
        Element-wise sum of the input arrays, optionally scaled.
    
    """
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    if alpha not in (1, None):
        return torch.add(x1, x2, alpha=alpha, out=out)
    return torch.add(x1, x2, out=out)


add.support_native_out = True


@with_unsupported_dtypes({"2.1.0 and below": ("complex",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def bitwise_xor(
    x1: Union[int, bool, torch.Tensor],
    x2: Union[int, bool, torch.Tensor],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Performs an element-wise bitwise XOR operation between two arrays. 
    
    XOR stands for "exclusive or", meaning the output is True only when inputs 
    differ (one true, one false). 
    
    Parameters
    ----------
    x1 : int or array_like
        First input operand. 
    x2 : int or array_like
        Second input operand. Must be the same shape as x1 if not scalar.
    
    out : ndarray, optional
        Output array. Must be the same shape as the expected output if provided.
    
    Returns
    -------
    ret : ndarray
        The XOR truth table results of the element-wise operation.
    
    Examples
    --------
    >>> import ivy
    >>> ivy.bitwise_xor(1, 2)
    3
    >>> ivy.bitwise_xor([True, False], [False, True])  
    array([ True,  True])
    
    """
    x1, x2 = ivy.promote_types_of_inputs(x1, x2, array_api_promotion=True)
    return torch.bitwise_xor(x1, x2, out=out)


bitwise_xor.support_native_out = True


@with_supported_dtypes({"2.1.0 and below": ("complex",)}, backend_version)
def imag(
    val: torch.Tensor,
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.imag(val)


imag.support_native_out = False


@with_unsupported_dtypes({"2.1.0 and below": ("float16", "complex")}, backend_version)
@handle_numpy_arrays_in_specific_backend
def expm1(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Computes exponential minus 1 of the input tensor element-wise. 
    
    This calculates e^x - 1, where e is Euler's number. This function is useful when operations are performed using log1p and expm1 to avoid loss of precision with very small inputs.
    
    Parameters
    ----------
    x : torch.Tensor
        Input tensor. Must have floating point or complex data type.
    
    Returns
    -------
    torch.Tensor
        Exponential of `x` minus 1 computed element-wise.
    
    Examples
    --------
    >>> x = torch.tensor([-1., -0.5, 0, 0.5, 1])
    >>> expm1(x)
    tensor([-0.6321, -0.3935,  0.0000,  0.6487,  1.7183])
    """
    x = _cast_for_unary_op(x)
    return torch.expm1(x, out=out)


expm1.support_native_out = True


@with_unsupported_dtypes({"2.1.0 and below": ("complex",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def bitwise_invert(
    x: Union[int, bool, torch.Tensor], /, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Computes the bitwise inversion of an integer elementwise. 
    
    The bitwise inversion of `x` is defined as `-x - 1`.
    
    Parameters
    ----------
    x : int or array_like of ints
        Integer or array to invert.
    
    out : ndarray, optional
        A location into which the result is stored. If provided, it must have a shape that the 
        inputs broadcast to. If not provided or None, a freshly-allocated array is returned.
        
    Returns
    -------
    ret : ndarray of ints
        Result of the bitwise inversion. 
        
    Examples
    --------
    >>> bitwise_invert(25)
    -26
    >>> bitwise_invert([True, False])
    array([False,  True])
    """
    x = _cast_for_unary_op(x)
    return torch.bitwise_not(x, out=out)


bitwise_invert.support_native_out = True


@handle_numpy_arrays_in_specific_backend
def isfinite(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Checks element-wise if input contains finite numbers.  
    
    Parameters
    ----------
    x : array_like
        Input array.
    
    Returns
    -------
    ret : bool ndarray
        An array of booleans with the same shape as x, with True where x is finite and False otherwise.
    
    Examples
    --------
    >>> x = np.array([np.inf, 2, np.NINF])
    >>> np.isfinite(x)
    array([False,  True, False])
    """
    x = _cast_for_unary_op(x)
    return torch.isfinite(x)


@handle_numpy_arrays_in_specific_backend
def isinf(
    x: torch.Tensor,
    /,
    *,
    detect_positive: bool = True,
    detect_negative: bool = True,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Tests element-wise for positive or negative infinity.
    
    Parameters
    ----------
    x : array_like
        Input array.
    detect_positive : bool, optional
        Whether to detect positive infinity (True by default).
    detect_negative : bool, optional  
        Whether to detect negative infinity (True by default).
    out : array_like, optional
        Output array with result.
    
    Returns
    -------
    ret : ndarray
        A boolean array with the same shape as x indicating whether each element is 
        infinite (True) or not (False).
    
    Examples
    --------
    >>> x = [5, 6, np.inf, -np.inf, np.nan]
    >>> isinf(x)
    array([False, False,  True,  True, False])
    
    """
    x = _cast_for_unary_op(x)
    if detect_negative and detect_positive:
        return torch.isinf(x)
    elif detect_negative:
        return torch.isneginf(x)
    elif detect_positive:
        return torch.isposinf(x)
    return torch.full_like(x, False, dtype=torch.bool)


@handle_numpy_arrays_in_specific_backend
def equal(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compares two arrays element-wise for equality.
    
    Parameters
    ----------
    x1 : array_like
        First array to compare.
    x2 : array_like
        Second array to compare. 
    
    out : ndarray, optional
        Output array. If not provided, a new array will be created.
    
    Returns
    -------
    ret : ndarray
        An array containing the element-wise comparisons.
        Values are True where x1[i] == x2[i] and False otherwise.
    
    Raises
    ------
    TypeError
        If x1 and x2 have incompatible shapes or data types. 
    """
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return torch.eq(x1, x2, out=out)


equal.support_native_out = True


@with_unsupported_dtypes({"2.1.0 and below": ("complex",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def less_equal(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compares two arrays element-wise for equality.
    
    Parameters
    ----------
    x1 : array_like
        First array to compare.
    x2 : array_like
        Second array to compare. 
    
    out : ndarray, optional
        Output array. If not provided, a new array will be created.
    
    Returns
    -------
    ret : ndarray
        An array containing the element-wise comparisons.
        Values are True where x1[i] == x2[i] and False otherwise.
    
    Raises
    ------
    TypeError
        If x1 and x2 have incompatible shapes or data types. 
    """
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return torch.less_equal(x1, x2, out=out)


less_equal.support_native_out = True


@with_unsupported_dtypes({"2.1.0 and below": ("complex",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def bitwise_and(
    x1: Union[int, bool, torch.Tensor],
    x2: Union[int, bool, torch.Tensor],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Performs a bitwise AND operation element-wise between two arrays. 
    
    This computes the bitwise AND between each element in ``x1`` and ``x2``.
    
    Parameters
    ----------
    x1 : array_like
        First input array.
    x2 : array_like
        Second input array.
    
    Returns
    -------
    ret : ndarray
        An array containing the element-wise bitwise AND result.
        
    Examples
    --------
    >>> x1 = np.array([1, 2, 3], dtype=np.int8)
    >>> x2 = np.array([4, 5, 6], dtype=np.int8)  
    >>> np.bitwise_and(x1, x2)
    array([0, 0, 2], dtype=int8)
    
    The operation is performed on the binary representations.
    """
    x1, x2 = ivy.promote_types_of_inputs(x1, x2, array_api_promotion=True)
    return torch.bitwise_and(x1, x2, out=out)


bitwise_and.support_native_out = True


@with_unsupported_dtypes({"2.1.0 and below": ("float16", "complex")}, backend_version)
@handle_numpy_arrays_in_specific_backend
def ceil(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Rounds each element of x up to the nearest integer.
    
    Parameters
    ----------
    x : array_like
        Input array containing elements to round up.
    out : array_like, optional
        Output array, for writing the result to. It must have a shape that the 
        inputs broadcast to.
    
    Returns
    -------
    ret : array_like
        An array of the same shape as x, containing the rounded values.
    
    Examples
    --------
    >>> a = np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
    >>> np.ceil(a)
    array([-1., -1., -0.,  1.,  2.,  2.,  2.])
    """
    x = _cast_for_unary_op(x)
    if "int" in str(x.dtype):
        if ivy.exists(out):
            return ivy.inplace_update(out, x)
        return x
    return torch.ceil(x, out=out)


ceil.support_native_out = True


@with_unsupported_dtypes({"2.1.0 and below": ("float16", "complex")}, backend_version)
@handle_numpy_arrays_in_specific_backend
def floor(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Rounds floating point array elements downward to the next lower integer value.
    
    Parameters
    ----------
    x : tensor
        Input array.
    out : tensor, optional
        Output tensor. If not provided, a new tensor will be created.
        
    Returns
    -------
    ret : tensor
        An array with the elements of `x` rounded downward to the nearest integer.
    
    """
    x = _cast_for_unary_op(x)
    if "int" in str(x.dtype):
        if ivy.exists(out):
            return ivy.inplace_update(out, x)
        return x
    return torch.floor(x, out=out)


floor.support_native_out = True


@with_unsupported_dtypes({"2.1.0 and below": ("complex",)}, backend_version)
def fmin(
    x1: torch.Tensor,
    x2: torch.Tensor,
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Element-wise minimum of array elements.
    
    Compares two arrays and returns a new array containing the element-wise minima. 
    NaN values are propagated. If one of the elements being compared is a NaN, the 
    non-NaN element is returned. If both elements are NaNs then the first is returned.
    
    Parameters
    ----------
    x1 : array_like
        First array to compare. 
    x2 : array_like
        Second array to compare. Must have the same shape as x1.
    out : array_like, optional
        Output array. Must have the same shape as x1 and x2.
    
    Returns
    -------
    fmin : ndarray or scalar
        The minimum of x1 and x2, element-wise. Returns scalar if both x1 and x2 are scalars.
    
    """
    return torch.fmin(x1, x2, out=None)


fmin.support_native_out = True


@with_unsupported_dtypes({"2.1.0 and below": ("float16",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def asin(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Calculates the arcsine of the input tensor x. 
    
    Parameters
    ----------
    x: torch.Tensor
        Input tensor.
    out: torch.Tensor, optional
        Output tensor to store the result.
    
    Returns
    -------
    ret: torch.Tensor
        The arcsine of each element of x.
        This will be in the range [-pi/2, pi/2] for real-valued inputs.
    
    Raises
    ------
    TypeError
        If `x` is not a `torch.Tensor`. 
    """
    x = _cast_for_unary_op(x)
    return torch.asin(x, out=out)


asin.support_native_out = True


@with_unsupported_dtypes({"2.1.0 and below": ("float16",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def asinh(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Calculates the arcsine of the input tensor x. 
    
    Parameters
    ----------
    x: torch.Tensor
        Input tensor.
    out: torch.Tensor, optional
        Output tensor to store the result.
    
    Returns
    -------
    ret: torch.Tensor
        The arcsine of each element of x.
        This will be in the range [-pi/2, pi/2] for real-valued inputs.
    
    Raises
    ------
    TypeError
        If `x` is not a `torch.Tensor`. 
    """
    x = _cast_for_unary_op(x)
    return torch.asinh(x, out=out)


asinh.support_native_out = True


@with_unsupported_dtypes({"2.1.0 and below": ("float16",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def sign(
    x: torch.Tensor,
    /,
    *,
    np_variant: Optional[bool] = True,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Returns an element-wise indication of the sign of a number.
    
    The `sign` function returns ``-1 if x < 0, 0 if x==0, 1 if x > 0``. 
    nan is returned for nan inputs.
    
    Parameters
    ----------
    x : array_like
        Input values.
    
    np_variant: bool, optional
        If set to True, the numpy variant of the sign function is used. 
        Default is True.
    
    out : array_like, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned.
    
    Returns
    -------
    ret : array_like
        The sign of `x`. This is a scalar if `x` is a scalar.
        
    """
    x = _cast_for_unary_op(x)
    if "complex" in str(x.dtype):
        if np_variant:
            return torch.where(
                x.real != 0, torch.sign(x.real) + 0.0j, torch.sign(x.imag) + 0.0j
            )
        return torch.sgn(x, out=out)
    return torch.sign(x, out=out)


sign.support_native_out = True


@with_unsupported_dtypes({"2.1.0 and below": ("float16",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def sqrt(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Calculates the square root of the input tensor element-wise.
    
    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    out : torch.Tensor, optional
        Optional output tensor to hold the result.
    
    Returns
    ------- 
    ret : torch.Tensor
        An array containing the square root of each element in x.
        This is a float array with the same shape as x.
    
    Examples
    --------
    >>> x = torch.tensor([4., 9., 16.])
    >>> ivy.sqrt(x)
    tensor([2., 3., 4.])
    """
    x = _cast_for_unary_op(x)
    return torch.sqrt(x, out=out)


sqrt.support_native_out = True


@with_unsupported_dtypes({"2.1.0 and below": ("float16",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def cosh(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Calculates the hyperbolic cosine of the input array.
    
    Parameters
    ----------
    x : array_like
        Input array.
    out : Tensor, optional
        Output tensor. 
    
    Returns
    -------
    ret : Tensor
        The hyperbolic cosine of the input tensor computed element-wise.
    
    Examples
    --------
    >>> x = torch.tensor([1.0, 2.0])
    >>> cosh(x)
    tensor([1.54308063, 3.76219569])
    """
    x = _cast_for_unary_op(x)
    return torch.cosh(x, out=out)


cosh.support_native_out = True


@with_unsupported_dtypes({"2.1.0 and below": ("float16",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def log10(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Calculates the base 10 logarithm of the input array, element-wise. 
    
    Parameters
    ----------
    x : array_like
        Input array.
    out : Tensor, optional
        Output tensor. 
    
    Returns
    -------
    ret : Tensor
        The base 10 log of each element in x.
    
    """
    x = _cast_for_unary_op(x)
    return torch.log10(x, out=out)


log10.support_native_out = True


@with_unsupported_dtypes({"2.1.0 and below": ("float16",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def log2(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Calculates the base 2 logarithm of the input tensor element-wise. 
    
    Parameters
    ----------
    x: array_like
        Input array.
    
    out: Tensor, optional
        Output tensor to write the result to.
    
    Returns
    -------
    ret: Tensor
        Base 2 logarithm of each element in x.
    
    Examples
    --------
    >>> x = [4, 8, 16]
    >>> ivy.log2(x)
    [2, 3, 4]
    """
    x = _cast_for_unary_op(x)
    return torch.log2(x, out=out)


@with_unsupported_dtypes({"2.1.0 and below": ("float16", "complex")}, backend_version)
@handle_numpy_arrays_in_specific_backend
def log1p(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    
    x = _cast_for_unary_op(x)
    return torch.log1p(x, out=out)


log1p.support_native_out = True


@handle_numpy_arrays_in_specific_backend
def isnan(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Tests element-wise whether the input contains NaN value(s). 
    
    Parameters
    ----------
    x : tensor
        Input tensor to check for NaN values.
    
    Returns
    -------
    ret : tensor
        A boolean tensor with the same shape as x, True where x is NaN, False otherwise.
    
    """
    x = _cast_for_unary_op(x)
    return torch.isnan(x)


@with_unsupported_dtypes({"2.1.0 and below": ("complex",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def less(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Performs element-wise comparison to check if values in x1 are less than those in x2.
    
    Parameters
    ----------
    x1 : array_like
        First array to compare. 
    x2 : array_like
        Second array to compare. Must be compatible (broadcastable) with x1.
        
    out : optional
        Output tensor.
    
    Returns
    -------
    ret : bool ndarray
        Boolean array containing the result of the element-wise comparison.
        Values in ret are True where x1 < x2 and False otherwise.
    
    Raises
    ------
    ValueError
        If x1 and x2 cannot be broadcast together.
    """
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return torch.lt(x1, x2, out=out)


less.support_native_out = True


@handle_numpy_arrays_in_specific_backend
def multiply(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Multiplies two arrays element-wise. 
    
    This function accepts arrays of any shape as long as they are broadcastable to each other.
    
    Parameters
    ----------
    x1 : array_like
        First array to multiply.
    x2 : array_like 
        Second array to multiply.
    out : array_like, optional
        Output array. Must be able to broadcast against x1 and x2.
    
    Returns
    -------
    ret : ndarray
        The product of x1 and x2, element-wise. Returns a scalar if both x1 and x2 are scalars.
    
    Examples
    -------- 
    >>> x1 = [1, 2, 3]
    >>> x2 = [4, 5, 6]
    >>> multiply(x1, x2)
    array([ 4, 10, 18])
    
    """
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return torch.multiply(x1, x2, out=out)


multiply.support_native_out = True


@with_unsupported_dtypes({"2.1.0 and below": ("float16",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def cos(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Calculates the cosine of the input array, element-wise.
    
    Parameters
    ----------
    x : array_like
        Input array containing elements to calculate cosine of.
    out : optional
        Output tensor to store results in.
    
    Returns
    -------
    ret : Tensor
        An array containing the cosine of each element in x.
        
    Raises
    ------
    RuntimeError
        If input contains unsupported dtypes.
    """
    x = _cast_for_unary_op(x)
    return torch.cos(x, out=out)


cos.support_native_out = True


@handle_numpy_arrays_in_specific_backend
def logical_not(
    x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Computes the logical NOT of an array element-wise.
    
    Parameters
    ----------
    x : array_like
        Input array.
    
    out : optional
        Output tensor, for writing the result to. It must have a shape that the
        inputs broadcast to.
    
    Returns
    -------
    ret : ndarray or scalar
        True where x is False, False otherwise. This is a scalar if x is a scalar.
    
    Examples
    --------
    >>> x = np.array([True, False])
    >>> logical_not(x)
    array([False, True])
    """
    x = _cast_for_unary_op(x)
    return torch.logical_not(x.type(torch.bool), out=out)


logical_not.support_native_out = True


@handle_numpy_arrays_in_specific_backend
def divide(
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
    ret = torch.div(x1, x2)
    if ivy.is_float_dtype(x1.dtype) or ivy.is_complex_dtype(x1.dtype):
        ret = ivy.astype(ret, x1.dtype, copy=False)
    else:
        ret = ivy.astype(ret, ivy.default_float_dtype(as_native=True), copy=False)
    return ret


divide.support_native_out = True


@with_unsupported_dtypes({"2.1.0 and below": ("complex",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def greater(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    /,
    *,
<<<<<<< HEAD
    axes: Optional[Union[int, Tuple[Sequence[int], Sequence[int]]]] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    """
    Compares two tensors and returns a new tensor containing the element-wise 
    greater results.
    
    Parameters
    ----------
    x1 : float or torch.Tensor
        First input tensor to compare.
    x2 : float or torch.Tensor 
        Second input tensor to compare.
    out : torch.Tensor, optional
        Output tensor.
    
    Returns
    -------
    ret : torch.Tensor
        New tensor containing the element-wise greater results.
    
    """
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
=======
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
>>>>>>> 800e5f8 (add a complete file of undocumented functions)
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return torch.greater(x1, x2, out=out)


greater.support_native_out = True


@with_unsupported_dtypes({"2.1.0 and below": ("complex",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def greater_equal(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compares two arrays element-wise for equality.
    
    Parameters
    ----------
    x1 : array_like
        First array to compare.
    x2 : array_like
        Second array to compare. 
    
    out : ndarray, optional
        Output array. If not provided, a new array will be created.
    
    Returns
    -------
    ret : ndarray
        An array containing the element-wise comparisons.
        Values are True where x1[i] == x2[i] and False otherwise.
    
    Raises
    ------
    TypeError
        If x1 and x2 have incompatible shapes or data types. 
    """
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return torch.greater_equal(x1, x2, out=out)


greater_equal.support_native_out = True


@with_unsupported_dtypes({"2.1.0 and below": ("float16",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def acos(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Calculates the cosine of the input array, element-wise.
    
    Parameters
    ----------
    x : array_like
        Input array containing elements to calculate cosine of.
    out : optional
        Output tensor to store results in.
    
    Returns
    -------
    ret : Tensor
        An array containing the cosine of each element in x.
        
    Raises
    ------
    RuntimeError
        If input contains unsupported dtypes.
    """
    x = _cast_for_unary_op(x)
    return torch.acos(x, out=out)


acos.support_native_out = True


@with_unsupported_dtypes({"2.1.0 and below": ("float",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def lcm(
    x1: torch.Tensor,
    x2: torch.Tensor,
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Calculates the lowest common multiple of the input arrays.
    
    The lowest common multiple is the smallest positive integer that is 
    divisible by both x1 and x2.
    
    Parameters
    ----------
    x1 : array_like
        First input array.
    x2 : array_like
        Second input array. 
    
    out : Optional[torch.Tensor], optional
        Output tensor. If not provided, a new tensor will be created.
    
    Returns
    -------
    ret : torch.Tensor
        Lowest common multiple of the inputs.
    
    Examples
    --------
    >>> x1 = [4, 6]
    >>> x2 = [6, 8]
    >>> lcm(x1, x2)
    array([12, 24])
    """
    x1, x2 = promote_types_of_inputs(x1, x2)
    return torch.lcm(x1, x2, out=out)


lcm.support_native_out = True


@handle_numpy_arrays_in_specific_backend
def logical_xor(
    x1: torch.Tensor, x2: torch.Tensor, /, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Performs logical XOR on two boolean tensors x1 and x2.
    
    This function computes the truth value of x1 XOR x2.
    
    Parameters
    ----------
    x1 : tensor
        First input tensor. Must be castable to boolean tensor. 
    
    x2: tensor
        Second input tensor. Must be castable to boolean tensor.
    
    out : tensor, optional
        Optional output tensor to store the result. 
    
    Returns
    -------
    ret : tensor
        Boolean tensor of x1 XOR x2.
    
    Examples
    --------
    >>> x1 = ivy.array([True, False])
    >>> x2 = ivy.array([False, True])
    >>> ivy.logical_xor(x1, x2) 
    ivy.array([True, True])
    """
    return torch.logical_xor(x1.type(torch.bool), x2.type(torch.bool), out=out)


logical_xor.support_native_out = True


@handle_numpy_arrays_in_specific_backend
def logical_and(
    x1: torch.Tensor, x2: torch.Tensor, /, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Computes the logical AND of two arrays element-wise.
    
    Parameters
    ----------
    x1 : array_like
        First input array. Must be able to convert to boolean tensor.
    x2 : array_like
        Second input array. Must be able to convert to boolean tensor.
    out : optional
        Output tensor.
    
    Returns
    -------
    ret : array_like
        Element-wise logical AND of input arrays.
    
    Raises
    ------
    ValueError
        If x1 and x2 cannot be cast to boolean tensors. 
    
    """
    return torch.logical_and(x1.type(torch.bool), x2.type(torch.bool), out=out)


logical_and.support_native_out = True


@handle_numpy_arrays_in_specific_backend
def logical_or(
    x1: torch.Tensor, x2: torch.Tensor, /, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute the truth value of x1 OR x2 element-wise.
    
    Parameters
    ----------
    x1 : array_like
        First input array.
    x2 : array_like
        Second input array of the same shape as x1.
        
    out : array_like, optional
        Output array with the same shape as x1 and x2.
    
    Returns
    -------
    ret : array_like
        An array with the same shape as x1 and x2 containing the truth value of x1 OR x2.
    
    Examples
    --------
    >>> x1 = np.array([True, False])
    >>> x2 = np.array([False, True])
    >>> np.logical_or(x1, x2)
    array([ True, True])
    """
    return torch.logical_or(x1.type(torch.bool), x2.type(torch.bool), out=out)


logical_or.support_native_out = True


@with_unsupported_dtypes({"2.1.0 and below": ("float16",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def acosh(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Calculates the hyperbolic cosine of the input array.
    
    Parameters
    ----------
    x : array_like
        Input array.
    out : Tensor, optional
        Output tensor. 
    
    Returns
    -------
    ret : Tensor
        The hyperbolic cosine of the input tensor computed element-wise.
    
    Examples
    --------
    >>> x = torch.tensor([1.0, 2.0])
    >>> cosh(x)
    tensor([1.54308063, 3.76219569])
    """
    x = _cast_for_unary_op(x)
    return torch.acosh(x, out=out)


acosh.support_native_out = True


@with_unsupported_dtypes({"2.1.0 and below": ("float16",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def sin(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Calculates the trigonometric sine, element-wise.
    
    Parameters
    ----------
    x : Tensor
        The input tensor.
    out : Tensor, optional
        The output tensor.
    
    Returns
    -------
    ret : Tensor
        The sine of each element in x.
        This is a tensor of the same shape as x.
    
    Examples
    --------
    >>> x = torch.tensor([0., np.pi/2, np.pi])
    >>> ivy.sin(x)
    tensor([0., 1., 0.])
    """
    x = _cast_for_unary_op(x)
    return torch.sin(x, out=out)


sin.support_native_out = True


@handle_numpy_arrays_in_specific_backend
def negative(
    x: Union[float, torch.Tensor], /, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Negates an array element-wise. 
    
    Parameters
    ----------
    x: Union[float, torch.Tensor]
        Input array or scalar whose elements will be negated.
        
    out: Optional[torch.Tensor]
        Optional output array, for writing the result to. It must have a shape that the 
        inputs broadcast to.
        
    Returns
    -------
    ret: torch.Tensor
        An array with the same shape and type as x, with all elements negated.
    
    """
    x = _cast_for_unary_op(x)
    return torch.neg(x, out=out)


negative.support_native_out = True


@handle_numpy_arrays_in_specific_backend
def not_equal(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compares two arrays element-wise for equality.
    
    Parameters
    ----------
    x1 : array_like
        First array to compare.
    x2 : array_like
        Second array to compare. 
    
    out : ndarray, optional
        Output array. If not provided, a new array will be created.
    
    Returns
    -------
    ret : ndarray
        An array containing the element-wise comparisons.
        Values are True where x1[i] == x2[i] and False otherwise.
    
    Raises
    ------
    TypeError
        If x1 and x2 have incompatible shapes or data types. 
    """
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return torch.not_equal(x1, x2, out=out)


not_equal.support_native_out = True


@with_unsupported_dtypes({"2.1.0 and below": ("float16",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def tanh(
    x: torch.Tensor, /, *, complex_mode="jax", out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Computes the element-wise hyperbolic tangent of the input tensor x. 
    
    This function returns a tensor of the same shape and dtype as the input x.
    
    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    out : torch.Tensor, optional
        Output tensor to store the result.
    
    Returns
    -------
    ret : torch.Tensor
        The hyperbolic tangent of the input tensor computed element-wise.
    
    """
    x = _cast_for_unary_op(x)
    return torch.tanh(x, out=out)


tanh.support_native_out = True


@with_unsupported_dtypes({"2.1.0 and below": ("float16", "complex")}, backend_version)
@handle_numpy_arrays_in_specific_backend
def floor_divide(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Rounds floating point array elements downward to the next lower integer value.
    
    Parameters
    ----------
    x : tensor
        Input array.
    out : tensor, optional
        Output tensor. If not provided, a new tensor will be created.
        
    Returns
    -------
    ret : tensor
        An array with the elements of `x` rounded downward to the nearest integer.
    
    """
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    if ivy.exists(out):
        if not ivy.is_float_dtype(out):
            return ivy.inplace_update(
                out, torch.floor(torch.div(x1, x2)).type(out.dtype)
            )
    return torch.floor(torch.div(x1, x2), out=out).type(x1.dtype)


floor_divide.support_native_out = True


@with_unsupported_dtypes({"2.1.0 and below": ("complex",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def bitwise_or(
    x1: Union[int, bool, torch.Tensor],
    x2: Union[int, bool, torch.Tensor],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Performs a bitwise OR operation between two arrays element-wise. 
    
    Bitwise OR between two binary numbers returns 1 if either of the bits is 1. For example, 
    1 OR 0 = 1, 1 OR 1 = 1, 0 OR 0 = 0. This function is applied to each element in the input arrays.
    
    Parameters
    ----------
    x1 : int, bool, or array_like
        First input operand for the bitwise OR operation. 
    x2 : int, bool, or array_like
        Second input operand for the bitwise OR operation.
        
    out : Tensor, optional
        Optional output tensor. Must be able to cast the inputs to this dtype.
       
    Returns
    -------
    ret : Tensor
        Element-wise bitwise OR of the input arrays x1 and x2.
    
    Examples
    --------
    >>> bitwise_or(1, 0)
    tensor(1)
    
    >>> x1 = torch.tensor([True, False])  
    >>> x2 = torch.tensor([False, True])
    >>> bitwise_or(x1, x2)
    tensor([ True,  True])
    """
    x1, x2 = ivy.promote_types_of_inputs(x1, x2, array_api_promotion=True)
    return torch.bitwise_or(x1, x2, out=out)


bitwise_or.support_native_out = True


@with_unsupported_dtypes({"2.1.0 and below": ("float16",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def sinh(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Calculates the trigonometric sine, element-wise.
    
    Parameters
    ----------
    x : Tensor
        The input tensor.
    out : Tensor, optional
        The output tensor.
    
    Returns
    -------
    ret : Tensor
        The sine of each element in x.
        This is a tensor of the same shape as x.
    
    Examples
    --------
    >>> x = torch.tensor([0., np.pi/2, np.pi])
    >>> ivy.sin(x)
    tensor([0., 1., 0.])
    """
    x = _cast_for_unary_op(x)
    return torch.sinh(x, out=out)


sinh.support_native_out = True


@handle_numpy_arrays_in_specific_backend
def positive(
    x: Union[float, torch.Tensor], /, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Element-wise calculation of the positive values of the input array.
    
    Replaces all negative values in x with zero.
    
    Parameters
    ----------
    x : array_like
        Input array containing negative and/or positive values.
    
    Returns
    -------  
    ret : ndarray
        An array with the same shape and type as x, with all negative
        values replaced by zero.
    
    Examples
    --------
    >>> x = [-1, 0, 2, -4]
    >>> positive(x)
    array([0, 0, 2, 0])
    """
    x = _cast_for_unary_op(x)
    return torch.positive(x)


@handle_numpy_arrays_in_specific_backend
def square(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Squares each element in the input tensor `x`. 
    
    This is equivalent to multiplying each element by itself.
    
    Parameters
    ----------
    x : torch.Tensor
        Input tensor to square.
    
    out : Optional[torch.Tensor], optional
        Output tensor. 
    
    Returns
    -------
    ret : torch.Tensor
        Tensor containing the squared values of `x`.
    
    """
    x = _cast_for_unary_op(x)
    return torch.square(x, out=out)


square.support_native_out = True


@handle_numpy_arrays_in_specific_backend
def pow(
    x1: torch.Tensor,
    x2: Union[int, float, torch.Tensor],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Computes x1 raised to the power of x2 element-wise.
    
    Supports broadcasting and in-place operations based on the input arrays.
    
    Handles edge cases like 0**negative number and complex dtypes correctly.
    
    Parameters
    ----------
    x1 : array_like
        The base array.
    x2 : array_like
        The exponent array.
    out : array_like
        Output array. Must be able to cast x1 and x2 arrays to this dtype.
    
    Returns
    -------
    ret : array_like
        An array the same shape as x1+x2 containing the element-wise results.
    
    Examples
    --------
    >>> x1 = [1, 2, 3]
    >>> x2 = [4, 5, 6]
    >>> pow(x1, x2)
    array([1, 32, 729])
    
    >>> import ivy
    >>> a = ivy.array([0, 1, 2])
    >>> ivy.pow(a, -1)  
    array([inf, 1., 0.5])
    """
    if ivy.is_complex_dtype(x1) and ivy.any(ivy.isinf(x2)):
        ret = torch.pow(x1, x2)
        x2 = torch.as_tensor(x2).to(torch.float64)
        return torch.where(
            ivy.isinf(x2), torch.nan + torch.nan * 1j if x2 < 0 else -0 * 1j, ret
        )
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    if ivy.any(x1 == 0):
        if ivy.is_complex_dtype(x2):
            x2 = torch.broadcast_to(x2, x1.shape)
            ret = torch.pow(x1, x2)
            return torch.where(x1 == 0, torch.nan + torch.nan * 1j, ret)
        elif (
            ivy.any(x2 < 0)
            and ivy.is_int_dtype(x2)
            and all(dtype not in str(x1.dtype) for dtype in ["int16", "int8"])
        ):
            if ivy.is_int_dtype(x1):
                fill_value = torch.iinfo(x1.dtype).min
            else:
                fill_value = torch.finfo(x1.dtype).min
            x2 = torch.broadcast_to(x2, x1.shape)
            ret = torch.pow(x1, x2)
            return torch.where(torch.bitwise_and(x1 == 0, x2 < 0), fill_value, ret)
    return torch.pow(x1, x2, out=out)


pow.support_native_out = True


@with_unsupported_dtypes({"2.1.0 and below": ("float16", "complex")}, backend_version)
@handle_numpy_arrays_in_specific_backend
def round(
    x: torch.Tensor, /, *, decimals: int = 0, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Rounds values element-wise to the nearest integer or specified number of decimals. 
    
    The rounding method depends on the data type:
    
    - For floating-point input, it rounds to the nearest integer, with ties (fractional values of 0.5) being rounded away from zero. This mimics the behavior of Python's standard round() function.
    
    - For integral input, the number is returned unchanged.
    
    Parameters
    ----------
    x : array_like
        Input data to round.
    decimals : int, optional
        Number of decimals to round to, by default 0. 
        If decimals is negative, integer inputs are rounded to the left of the decimal point.
    out : ndarray, optional
        Output array with the same shape as x.
    
    Returns
    -------
    ret : ndarray
        An array of rounded values with the same shape and data type as x.
    
    Examples
    --------
    >>> x = np.array([1.5, 2.2, -3.7])
    >>> ivy.round(x)
    array([ 2.,  2., -4.])
    
    >>> x = np.array([12.46, 97.28, 23.67])
    >>> ivy.round(x, 1)  
    array([12.5, 97.3, 23.7])
    
    """
    if "int" in str(x.dtype):
        if ivy.exists(out):
            return ivy.inplace_update(out, x)
        return x
    return torch.round(x, decimals=decimals, out=out)


round.support_native_out = True


def trapz(
    y: torch.Tensor,
    /,
    *,
    x: Optional[torch.Tensor] = None,
    dx: Optional[float] = None,
    axis: int = -1,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Integrate along the given axis using the composite trapezoidal rule.
    
    Parameters
    ----------
    y : array_like
        Input array to integrate.
    x : array_like, optional
        The sample points corresponding to the y values. If x is None,
        the sample points are assumed to be evenly spaced dx apart. The default is None.
    dx : scalar, optional
        The spacing between sample points when x is None. The default is 1.
    axis : int, optional
        The axis along which to integrate. Default is -1.
    
    Returns
    -------
    ret : float
        Definite integral as approximated by trapezoidal rule.
    
    """
    if x is None:
        dx = dx if dx is not None else 1
        return torch.trapezoid(y, dx=dx, dim=axis)
    else:
        if dx is not None:
            TypeError(
                "trapezoid() received an invalid combination of arguments - got "
                "(Tensor, Tensor, int), but expected one of: *(Tensor "
                "y, Tensor x, *, int dim) * (Tensor y, *, Number dx, int dim)"
            )
        else:
            return torch.trapezoid(y, x=x, dim=axis)


trapz.support_native_out = False


@with_unsupported_dtypes({"2.1.0 and below": ("float16", "complex")}, backend_version)
@handle_numpy_arrays_in_specific_backend
def trunc(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Truncates the values in an array according to the precision specified by the decimals argument. 
    
    Values in the array are truncated to the specified number of decimal places.
    Truncation is done by discarding the decimals beyond the specified precision.
    
    Parameters
    ----------
    x : array_like
        Input array containing elements to truncate.
    out : ndarray, optional
        Output array. If not provided, a new array will be created.
    
    Returns
    -------
    truncated_array : ndarray
        An array containing the truncated values from the input array.
    
    Examples
    --------
    >>> trunc([2.567, 3.333], decimals=1)
    array([2.5, 3.3])
    
    """
    x = _cast_for_unary_op(x)
    if "int" not in str(x.dtype):
        return torch.trunc(x, out=out)
    ret = x
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


trunc.support_native_out = True


@handle_numpy_arrays_in_specific_backend
def abs(
    x: Union[float, torch.Tensor],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Calculates the absolute value element-wise for the input array. 
    
    Parameters
    ----------
    x : array_like
        Input array.
    
    out : array_like, optional
        Output array. Must be of the same shape and buffer length as the expected output.
    
    Returns
    -------
    ret : ndarray
        An array containing the absolute value of each element in x.
        This is a scalar if x is a scalar.
    
    Examples
    --------
    >>> x = np.array([-1, 2, -3])
    >>> abs(x)
    array([1, 2, 3])
    """
    x = _cast_for_unary_op(x)
    if x.dtype is torch.bool:
        if ivy.exists(out):
            return ivy.inplace_update(out, x)
        return x
    return torch.abs(x, out=out)


abs.support_native_out = True


@with_unsupported_dtypes({"2.1.0 and below": ("float16", "complex")}, backend_version)
@handle_numpy_arrays_in_specific_backend
def logaddexp(
    x1: torch.Tensor, x2: torch.Tensor, /, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Adds two arrays element-wise. 
    
    Supports both scalar and array inputs and optionally allows specifying a 
    scaling factor.
    
    Parameters
    ----------
    x1: array_like
        First input array to add.
    x2: array_like 
        Second input array to add.
    alpha: int or float, optional
        Scaling factor for the addition. Default is 1.
    out: tensor, optional
        Output tensor. 
    
    Returns
    -------
    tensor
        Element-wise sum of the input arrays, optionally scaled.
    
    """
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return torch.logaddexp(x1, x2, out=out)


logaddexp.support_native_out = True


@with_unsupported_dtypes({"2.1.0 and below": ("float16",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def logaddexp2(
    x1: Union[torch.Tensor, float, list, tuple],
    x2: Union[torch.Tensor, float, list, tuple],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Adds two arrays element-wise. 
    
    Supports both scalar and array inputs and optionally allows specifying a 
    scaling factor.
    
    Parameters
    ----------
    x1: array_like
        First input array to add.
    x2: array_like 
        Second input array to add.
    alpha: int or float, optional
        Scaling factor for the addition. Default is 1.
    out: tensor, optional
        Output tensor. 
    
    Returns
    -------
    tensor
        Element-wise sum of the input arrays, optionally scaled.
    
    """
    x1, x2 = promote_types_of_inputs(x1, x2)
    if not ivy.is_float_dtype(x1):
        x1 = x1.type(ivy.default_float_dtype(as_native=True))
        x2 = x2.type(ivy.default_float_dtype(as_native=True))
    return torch.logaddexp2(x1, x2, out=out)


logaddexp2.support_native_out = True


@with_unsupported_dtypes({"2.1.0 and below": ("float16",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def tan(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Computes tangent of x element-wise. 
    
    Parameters
    ----------
    x : tensor_like
        Input tensor.
    out : tensor_like or None, optional
        Optional output tensor to hold the result.
    
    Returns
    -------
    ret : tensor_like
        The tangent of each element of x.
    
    Raises
    ------
    TypeError
        If x is not a tensor. 
    
    Examples
    --------  
    >>> x = torch.tensor([0., np.pi/2, np.pi])
    >>> tan(x)
    tensor([0., 1.63312394e+16, -1.22464680e-16])
    """
    x = _cast_for_unary_op(x)
    return torch.tan(x, out=out)


tan.support_native_out = True


@with_unsupported_dtypes({"2.1.0 and below": ("float16",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def atan(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Computes tangent of x element-wise. 
    
    Parameters
    ----------
    x : tensor_like
        Input tensor.
    out : tensor_like or None, optional
        Optional output tensor to hold the result.
    
    Returns
    -------
    ret : tensor_like
        The tangent of each element of x.
    
    Raises
    ------
    TypeError
        If x is not a tensor. 
    
    Examples
    --------  
    >>> x = torch.tensor([0., np.pi/2, np.pi])
    >>> tan(x)
    tensor([0., 1.63312394e+16, -1.22464680e-16])
    """
    x = _cast_for_unary_op(x)
    return torch.atan(x, out=out)


atan.support_native_out = True


@with_unsupported_dtypes(
    {"2.1.0 and below": ("float16", "bfloat16", "complex")}, backend_version
)  # TODO Fixed in PyTorch 1.12.1 (this note excludes complex)
@handle_numpy_arrays_in_specific_backend
def atan2(
    x1: torch.Tensor, x2: torch.Tensor, /, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Computes tangent of x element-wise. 
    
    Parameters
    ----------
    x : tensor_like
        Input tensor.
    out : tensor_like or None, optional
        Optional output tensor to hold the result.
    
    Returns
    -------
    ret : tensor_like
        The tangent of each element of x.
    
    Raises
    ------
    TypeError
        If x is not a tensor. 
    
    Examples
    --------  
    >>> x = torch.tensor([0., np.pi/2, np.pi])
    >>> tan(x)
    tensor([0., 1.63312394e+16, -1.22464680e-16])
    """
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return torch.atan2(x1, x2, out=out)


atan2.support_native_out = True


@with_unsupported_dtypes({"2.1.0 and below": ("float16",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def log(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Calculates the natural logarithm of the input tensor, element-wise. 
    
    This function calculates :math:`out = \\ln(x)` for each element in the input tensor.
    
    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    out : torch.Tensor, optional
        Output tensor. 
    
    Returns
    -------
    ret : torch.Tensor
        The natural logarithm of the input tensor, calculated element-wise.
    
    Raises
    ------
    ValueError
        If input contains negative or zero values.
    """
    x = _cast_for_unary_op(x)
    return torch.log(x, out=out)


log.support_native_out = True


@with_unsupported_dtypes({"2.1.0 and below": ("float16",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def exp(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Calculates the exponential of all elements in the input tensor x.  
    
    This function performs element-wise exponential.
    
    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    out : torch.Tensor, optional
        Output tensor.
    
    Returns
    -------
    ret : torch.Tensor
        A new tensor containing the exponentials of the elements in x.
    
    Examples
    --------
    >>> x = torch.tensor([1.0, 2.0])
    >>> exp(x)
    tensor([2.7183, 7.3891])
    """
    x = _cast_for_unary_op(x)
    return torch.exp(x, out=out)


exp.support_native_out = True


@handle_numpy_arrays_in_specific_backend
def exp2(
    x: Union[torch.Tensor, float, list, tuple],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Calculates the exponential of all elements in the input tensor x.  
    
    This function performs element-wise exponential.
    
    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    out : torch.Tensor, optional
        Output tensor.
    
    Returns
    -------
    ret : torch.Tensor
        A new tensor containing the exponentials of the elements in x.
    
    Examples
    --------
    >>> x = torch.tensor([1.0, 2.0])
    >>> exp(x)
    tensor([2.7183, 7.3891])
    """
    return torch.exp2(x, out=out)


exp2.support_native_out = True


@handle_numpy_arrays_in_specific_backend
def subtract(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    /,
    *,
    alpha: Optional[Union[int, float]] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Subtracts two arrays element-wise.
    
    Parameters
    ----------
    x1 : float or array_like
        The array to subtract from.
    x2: float or array_like
        The array to subtract with.  
    alpha: int or float, optional
        Scaling factor for x2. Default is 1.
    out: ndarray, optional
        A location in which to store the results. If not provided, a new array will be created.
    
    Returns
    -------
    ret : ndarray
        The difference of `x1` and `x2`, element-wise.
    
    Raises
    ------
    TypeError
        If x1 and x2 have different data types and can't be promoted to a common type.
    """
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    if alpha not in (1, None):
        return torch.subtract(x1, x2, alpha=alpha, out=out)
    return torch.subtract(x1, x2, out=out)


subtract.support_native_out = True


@with_unsupported_dtypes({"2.1.0 and below": ("float16", "complex")}, backend_version)
@handle_numpy_arrays_in_specific_backend
def remainder(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    /,
    *,
    modulus: bool = True,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Computes the element-wise remainder of division.
    
    This is the remainder operation as prescribed by the IEEE 754 standard.
    The remainder has the same sign as the dividend `x1`.
    
    Parameters
    ----------
    x1 : array_like
        Dividend array.
    x2 : array_like
        Divisor array.
    modulus : bool, optional
        If True, the absolute value remainder will be returned instead of the 
        default python behavior. Default is True.
        
    out : optional
        Output array. If not provided, a new array will be created.
        
    Returns
    -------
    ret : ndarray
        The element-wise remainder of the quotient ``x1/x2``. This has the same
        sign as `x1`.
    
    Examples
    --------
    >>> x1 = [5, 7, -3]
    >>> x2 = [2, -2, 2]
    >>> remainder(x1, x2)
    [1, -1, -1]
    
    """
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    if not modulus:
        res = x1 / x2
        res_floored = torch.where(res >= 0, torch.floor(res), torch.ceil(res))
        diff = res - res_floored
        diff, x2 = ivy.promote_types_of_inputs(diff, x2)
        if ivy.exists(out):
            if out.dtype != x2.dtype:
                return ivy.inplace_update(
                    out, torch.round(torch.mul(diff, x2)).to(out.dtype)
                )
        return torch.round(torch.mul(diff, x2), out=out).to(x1.dtype)
    return torch.remainder(x1, x2, out=out).to(x1.dtype)


remainder.support_native_out = True


@with_unsupported_dtypes({"2.1.0 and below": ("float16",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def atanh(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Computes the element-wise hyperbolic tangent of the input tensor x. 
    
    This function returns a tensor of the same shape and dtype as the input x.
    
    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    out : torch.Tensor, optional
        Output tensor to store the result.
    
    Returns
    -------
    ret : torch.Tensor
        The hyperbolic tangent of the input tensor computed element-wise.
    
    """
    x = _cast_for_unary_op(x)
    return torch.atanh(x, out=out)


atanh.support_native_out = True


@with_unsupported_dtypes({"2.1.0 and below": ("complex",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def bitwise_right_shift(
    x1: Union[int, bool, torch.Tensor],
    x2: Union[int, bool, torch.Tensor],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Performs a bitwise right shift operation on two inputs. 
    
    Shifts the bits of x1 right by x2 number of bits, filling the new left bits with zeros.
    
    Parameters
    ----------
    x1 : int, bool or tensor
        Input to shift bits from.
    x2 : int, bool or tensor
        Number of bits to shift x1 right by.  
    out : tensor, optional
        Optional output tensor, by default None.
    
    Returns
    -------
    tensor
        Tensor result of bitwise right shifting x1 by x2.
    
    Examples
    --------
    >>> x1 = 5 
    >>> x2 = 2
    >>> bitwise_right_shift(x1, x2)
    1
    
    """
    x1, x2 = ivy.promote_types_of_inputs(x1, x2, array_api_promotion=True)
    x2 = torch.clamp(x2, min=0, max=torch.iinfo(x2.dtype).bits - 1)
    return torch.bitwise_right_shift(x1, x2, out=out)


bitwise_right_shift.support_native_out = True


@with_unsupported_dtypes({"2.1.0 and below": ("complex",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def bitwise_left_shift(
    x1: Union[int, bool, torch.Tensor],
    x2: Union[int, bool, torch.Tensor],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Performs a bitwise left shift operation on two inputs. 
    
    Shifts the bits of x1 to the left by x2 number of bits. Vacated bits are zero-filled.
    
    Parameters
    ----------
    x1 : int, bool or array_like
        First input array containing integers.
    x2: int, bool or array_like
        Number of bits to shift x1. Must be non-negative.
    
    Returns
    -------
    ret : ndarray
        Output array containing the left shifted elements of x1.
    
    Examples
    --------
    >>> bitwise_left_shift(5, 2)
    20
    
    """
    x1, x2 = ivy.promote_types_of_inputs(x1, x2, array_api_promotion=True)
    return torch.bitwise_left_shift(x1, x2, out=out)


bitwise_left_shift.support_native_out = True


# Extra #
# ------#


@with_unsupported_dtypes({"2.1.0 and below": ("float16", "complex")}, backend_version)
@handle_numpy_arrays_in_specific_backend
def erf(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    The error function, or erf, calculates the integral of the Gaussian 
    distribution from 0 to x.
    
    Parameters
    ----------
    x: torch.Tensor
        Input tensor.
    
    out: Optional[torch.Tensor], optional
        Output tensor.
    
    Returns
    -------
    torch.Tensor
        The error function of each element of input `x`.
        
    Raises
    ------
    
    """
    x = _cast_for_unary_op(x)
    return torch.erf(x, out=out)


erf.support_native_out = True


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