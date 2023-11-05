# global
from typing import Union, Optional
from math import pi
import torch

# local
import ivy
from ivy.func_wrapper import (
    with_unsupported_dtypes,
    handle_numpy_arrays_in_specific_backend,
)
from ivy import promote_types_of_inputs
from . import backend_version


def _cast_for_unary_op(x):
    """Casts the input to a tensor for unary operations.

    This internal utility function ensures the input is a torch tensor, casting
    it to one if necessary. This allows the unary operations to easily
    process arguments of various types.

    Parameters
    ----------
    x : array_like
        The input data, which can be a tensor, numpy array, python scalar, etc.

    Returns
    -------
    ret : torch.Tensor
        The input cast to a torch tensor.

    Examples
    --------
    >>> x = 1.5
    >>> x_tensor = _cast_for_unary_op(x)
    tensor(1.5000)
    """
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
    """Adds two tensors element-wise.

    This function supports broadcasting.

    Parameters
    ----------
    x1 : Tensor
        The first tensor to be added.
    x2 : Tensor
        The second tensor to be added.

    Keyword Arguments
    -----------------
    alpha : float or int, optional
        A scaling factor for the second tensor `x2`. Default is 1.
    out : Tensor, optional
        Output tensor. The result will be placed in this tensor.

    Returns
    -------
    Tensor
        The sum of the two input tensors.

    Examples
    --------
    >>> x = torch.tensor([1, 2])
    >>> y = torch.tensor([3, 4])
    >>> torch.add(x, y)
    tensor([4., 6.])

    >>> x = torch.tensor([1, 2])
    >>> y = torch.tensor([3, 4])
    >>> torch.add(x, y, alpha=2)
    tensor([7., 10.])
    """
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    if alpha not in (1, None):
        return torch.add(x1, x2, alpha=alpha, out=out)
    return torch.add(x1, x2, out=out)


add.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("complex",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def bitwise_xor(
    x1: Union[int, bool, torch.Tensor],
    x2: Union[int, bool, torch.Tensor],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Performs a bitwise XOR operation on two input tensors x1 and x2.

    Bitwise XOR returns 1 in each bit position where the corresponding bits of x1 and x2 are different.

    Parameters
    ----------
    x1 : int, bool or tensor
        First input tensor.
    x2 : int, bool or tensor
        Second input tensor. Must be able to broadcast with x1.

    Returns
    -------
    ret : tensor
        Output tensor of bitwise XOR operation.

    Examples
    --------
    >>> x = torch.tensor([1, 0, 1])
    >>> y = torch.tensor([0, 1, 0])
    >>> torch.bitwise_xor(x, y)
    tensor([1, 1, 1])

    This shows that where the input tensors have different bits, the output is 1.
    """
    x1, x2 = ivy.promote_types_of_inputs(x1, x2, array_api_promotion=True)
    return torch.bitwise_xor(x1, x2, out=out)


bitwise_xor.support_native_out = True


def imag(
    val: torch.Tensor,
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Returns the imaginary part of the complex tensor.

    For real tensors, this returns a tensor of zeros. For complex tensors, this
    returns the imaginary component.

    Parameters
    ----------
    val : torch.Tensor
        Tensor for which to take the imaginary component.

    Returns
    -------
    ret : torch.Tensor
        Imaginary component of the input tensor.

    Examples
    --------
    >>> x = torch.tensor([1+2j, 3+4j])
    >>> imag(x)
    tensor([2., 4.])

    >>> x = torch.tensor([1., 2.])
    >>> imag(x)
    tensor([0., 0.])
    """
    if val.dtype not in (torch.complex64, torch.complex128):
        return torch.zeros_like(val, dtype=val.dtype)
    return torch.imag(val)


imag.support_native_out = False


@with_unsupported_dtypes({"2.0.1 and below": ("float16", "complex")}, backend_version)
@handle_numpy_arrays_in_specific_backend
def expm1(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Computes exponential of x minus 1 element-wise.

    This function provides greater precision than exp(x) - 1 for small values of x.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    out : Optional[torch.Tensor], optional
        Output tensor.

    Returns
    -------
    ret : torch.Tensor
        A tensor containing the exponential of the input tensor x minus 1 computed element-wise.

    Examples
    --------
    >>> x = torch.tensor([-1., 0., 1.])
    >>> torch.expm1(x)
    tensor([-0.6321,  0.0000,  1.7181])

    This illustrates that expm1(x) provides greater precision than exp(x) - 1
    for small values of x:

    >>> torch.exp(x) - 1
    tensor([-0.3679,  0.0000,  1.7181])
    """
    x = _cast_for_unary_op(x)
    return torch.expm1(x, out=out)


expm1.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("complex",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def bitwise_invert(
    x: Union[int, bool, torch.Tensor], /, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Performs a bitwise inversion on an integer or boolean input tensor x.

    This function flips all the bits in the binary representation of each element in x.
    For example, 0110 would become 1001.

    Parameters
    ----------
    x : int, bool, or tensor
        Input tensor to invert.

    Returns
    -------
    tensor
        Tensor containing the bitwise inverted elements of x.

    Examples
    --------
    >>> x = torch.tensor([2, 5])
    >>> torch.bitwise_invert(x)
    tensor([-3, -6])

    >>> x = torch.tensor([True, False])
    >>> torch.bitwise_invert(x)
    tensor([False, True])
    """
    x = _cast_for_unary_op(x)
    return torch.bitwise_not(x, out=out)


bitwise_invert.support_native_out = True


@handle_numpy_arrays_in_specific_backend
def isfinite(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Checks element-wise if input tensors are finite, i.e. not NaN, not
    infinity and not -infinity.

    Parameters
    ----------
    x : tensor
        The input tensor.

    Returns
    -------
    Boolean tensor
        A boolean tensor with the same shape as x, True where x is finite (not NaN, infinity or -infinity).

    Examples
    --------
    >>> x = torch.tensor([1, float('inf'), 2, float('nan'), -float('inf')])
    >>> torch.isfinite(x)
    tensor([True, False, True, False, False])

    This function can also be used on complex numbers:

    >>> z = torch.complex(1+1j, float('inf') + 1j, float('nan') + 3j)
    >>> torch.isfinite(z)
    tensor([ True, False, False])
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
    """Check if input contains infinite values.

    Parameters
    ----------
    x : Tensor
        The input tensor.
    detect_positive : bool, optional
        Whether to check for positive infinity (True) or negative infinity (False). Default: True.
    detect_negative : bool, optional
        Whether to check for negative infinity (True) or positive infinity (False). Default: True

    Returns
    -------
    Tensor
        A boolean tensor with the same shape as x indicating if each element is positive or negative infinity.

    Examples
    --------
    >>> x = torch.tensor([1.0, float('inf'), 2.0])
    >>> torch.isinf(x)
    tensor([False, True, False])

    >>> x = torch.tensor([1.0, float('-inf'), 2.0])
    >>> torch.isinf(x, detect_positive=False)
    tensor([False, False, False])
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
    """Checks element-wise equality between two arrays.

    Parameters
    ----------
    x1 : array_like
        First input array to compare.
    x2 : array_like
        Second input array to compare.

    Returns
    -------
    ret : ndarray
        Boolean array of same shape as inputs indicating equality.

    Examples
    --------
    >>> x = torch.tensor([1, 2, 3])
    >>> y = torch.tensor([1, 2, 3])
    >>> torch.equal(x, y)
    tensor([True, True, True])

    >>> x = torch.tensor([1, 2, 3])
    >>> y = torch.tensor([1, 4, 3])
    >>> torch.equal(x, y)
    tensor([True, False, True])
    """
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return torch.eq(x1, x2, out=out)


equal.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("complex",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def less_equal(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Checks element-wise equality between two arrays.

    Parameters
    ----------
    x1 : array_like
        First input array to compare.
    x2 : array_like
        Second input array to compare.

    Returns
    -------
    ret : ndarray
        Boolean array of same shape as inputs indicating equality.

    Examples
    --------
    >>> x = torch.tensor([1, 2, 3])
    >>> y = torch.tensor([1, 2, 3])
    >>> torch.equal(x, y)
    tensor([True, True, True])

    >>> x = torch.tensor([1, 2, 3])
    >>> y = torch.tensor([1, 4, 3])
    >>> torch.equal(x, y)
    tensor([True, False, True])
    """
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return torch.less_equal(x1, x2, out=out)


less_equal.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("complex",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def bitwise_and(
    x1: Union[int, bool, torch.Tensor],
    x2: Union[int, bool, torch.Tensor],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Performs a bitwise AND operation elementwise on the input arrays.

    Bitwise AND computes the bitwise AND of the underlying binary representation of
    the integers in the input arrays. This ufunc implements the C/Python operator '&'.

    Parameters
    ----------
    x1 : int, bool, or torch.Tensor
        First input operand.
    x2 : int, bool, or torch.Tensor
        Second input operand of the same shape and type as x1.

    Returns
    -------
    torch.Tensor
        The bitwise AND of x1 and x2, element-wise. Returns a tensor of the same
        shape and type as x1 and x2.

    Examples
    --------
    >>> x1 = torch.tensor([1, 0, 1])
    >>> x2 = torch.tensor([0, 1, 0])
    >>> torch.bitwise_and(x1, x2)
    tensor([0, 0, 0])

    >>> x1 = True
    >>> x2 = False
    >>> torch.bitwise_and(x1, x2)
    False
    """
    x1, x2 = ivy.promote_types_of_inputs(x1, x2, array_api_promotion=True)
    return torch.bitwise_and(x1, x2, out=out)


bitwise_and.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("float16", "complex")}, backend_version)
@handle_numpy_arrays_in_specific_backend
def ceil(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Ceiling of the input, element-wise.

    The ceil of the scalar x is the smallest integer i, such that i >= x.

    Parameters
    ----------
    x : array_like
        Input data.

    out : Tensor, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned.

    Returns
    -------
    y : Tensor or scalar
        The ceiling of each element in x, with float dtype. This is a scalar
        if x is a scalar.

    Examples
    --------
    >>> a = torch.tensor([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7])
    >>> torch.ceil(a)
    tensor([-1., -1., -0.,  1.,  2.,  2.])
    """
    x = _cast_for_unary_op(x)
    if "int" in str(x.dtype):
        if ivy.exists(out):
            return ivy.inplace_update(out, x)
        return x
    return torch.ceil(x, out=out)


ceil.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("float16", "complex")}, backend_version)
@handle_numpy_arrays_in_specific_backend
def floor(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Computes the floor of each element in the input tensor.

    The floor of the scalar x is the largest integer i, such that i <= x.

    Parameters
    ----------
    x : Tensor
        The input tensor.

    out : Optional Tensor, optional
        A tensor into which the result will be placed. If specified, the input
        tensor is cast to out.dtype before the operation is performed. This is useful
        for preventing data type overflows. Default is None.

    Returns
    -------
    ret : Tensor
        A tensor of the same shape as x, containing floor values.

    Examples
    --------
    >>> x = torch.tensor([1.7, -2.4, 3.5])
    >>> ivy.floor(x)
    tensor([ 1., -3.,  3.])

    >>> x = torch.tensor([3, -2, -1.5])
    >>> ivy.floor(x, out=torch.tensor([0., 0., 0.], dtype=torch.float32))
    tensor([3., -2., -2.], dtype=torch.float32)
    """
    x = _cast_for_unary_op(x)
    if "int" in str(x.dtype):
        if ivy.exists(out):
            return ivy.inplace_update(out, x)
        return x
    return torch.floor(x, out=out)


floor.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("complex",)}, backend_version)
def fmin(
    x1: torch.Tensor,
    x2: torch.Tensor,
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Calculates the element-wise minimum of two tensors.

    This function compares two tensors element-wise and returns a new tensor containing the minimum value from each pair.

    Parameters
    ----------
    x1 : torch.Tensor
        The first input tensor.
    x2 : torch.Tensor
        The second input tensor. Must be able to broadcast with x1.

    Returns
    -------
    torch.Tensor
        The element-wise minimum of x1 and x2.

    Examples
    --------
    >>> x = torch.tensor([1, 2, 3])
    >>> y = torch.tensor([3, 2, 1])
    >>> fmin(x, y)
    tensor([1, 2, 1])
    """
    return torch.fmin(x1, x2, out=None)


fmin.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def asin(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Computes the inverse sine of each element in the input tensor x.

    Parameters
    ----------
    x : torch.Tensor
        Input array containing elements to compute the inverse sine of.
    out : Optional[torch.Tensor], optional
        Output tensor.

    Returns
    -------
    ret : torch.Tensor
        An array containing the inverse sine of each element in x.
        This has the same shape as the input x.

    Examples
    --------
    >>> x = torch.tensor([0.5, 1])
    >>> y = asin(x)
    >>> y
    tensor([0.5236, 1.5708])
    """
    x = _cast_for_unary_op(x)
    return torch.asin(x, out=out)


asin.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def asinh(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Computes the inverse sine of each element in the input tensor x.

    Parameters
    ----------
    x : torch.Tensor
        Input array containing elements to compute the inverse sine of.
    out : Optional[torch.Tensor], optional
        Output tensor.

    Returns
    -------
    ret : torch.Tensor
        An array containing the inverse sine of each element in x.
        This has the same shape as the input x.

    Examples
    --------
    >>> x = torch.tensor([0.5, 1])
    >>> y = asin(x)
    >>> y
    tensor([0.5236, 1.5708])
    """
    x = _cast_for_unary_op(x)
    return torch.asinh(x, out=out)


asinh.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def sign(
    x: torch.Tensor,
    /,
    *,
    np_variant: Optional[bool] = True,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Computes an element-wise indication of the sign of a number.

    The `sign` function returns -1 if x < 0, 0 if x==0, 1 if x > 0.
    nan is returned for nan inputs.

    For complex inputs, the `sign` is calculated on both the real and imaginary parts.

    Parameters
    ----------
    x : array_like
        Input array.
    np_variant : bool, optional
        If True, uses the numpy implementation of sign, otherwise uses default
        torch implementation. Default is True.
    out : Tensor, optional
        A location in which to store the results. If provided, it must have a shape
        that the inputs broadcast to.

    Returns
    -------
    ret : Tensor
        An array with the same shape and type as x that contains the signed value of
        each element in x.

    Examples
    --------
    >>> x = torch.tensor([-5., 4.5])
    >>> np.sign(x)
    tensor([-1.,  1.])

    >>> x = torch.tensor(3 - 4j)
    >>> np.sign(x)
    tensor(0.6389 - 0.7692j)
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


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def sqrt(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Computes the square root of the input tensor element-wise.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    out : torch.Tensor, optional
        Output tensor.

    Returns
    -------
    ret : torch.Tensor
        An tensor containing the square roots of the elements of x.
        This is a float tensor of the same shape as x.

    Examples
    --------
    >>> x = torch.tensor([4., 9., 16.])
    >>> y = sqrt(x)
    >>> y
    tensor([2., 3., 4.])

    This computes the element-wise square root:

    >>> x = torch.tensor([[4., 9., 16.], [25., 36., 49.]])
    >>> sqrt(x)
    tensor([[ 2.,  3.,  4.],
            [ 5.,  6.,  7.]])
    """
    x = _cast_for_unary_op(x)
    return torch.sqrt(x, out=out)


sqrt.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def cosh(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Computes the hyperbolic cosine of the input array.

    Parameters
    ----------
    x : torch.Tensor
        Input array.
    out : torch.Tensor, optional
        Output tensor to store the result.

    Returns
    -------
    ret : torch.Tensor
        The hyperbolic cosine of the input array computed element-wise.

    Examples
    --------
    >>> x = torch.tensor([1., 2., 3.])
    >>> y = cosh(x)
    >>> print(y)
    tensor([1.5431, 3.7622, 10.0677])
    """
    x = _cast_for_unary_op(x)
    return torch.cosh(x, out=out)


cosh.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def log10(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Calculates the base 10 logarithm of the input tensor x.

    This function is the inverse of torch.pow(x, 10).

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    out : torch.Tensor, optional
        Output tensor.

    Returns
    -------
    ret : torch.Tensor
        Base 10 logarithm of x.

    Examples
    --------
    >>> x = torch.tensor([10., 100.])
    >>> y = torch.log10(x)
    >>> y
    tensor([1., 2.])

    >>> x = torch.tensor([1., 0.1])
    >>> torch.log10(x)
    tensor([0., -1.])
    """
    x = _cast_for_unary_op(x)
    return torch.log10(x, out=out)


log10.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def log2(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Computes the base 2 logarithm of the input tensor.

    This function calculates the base 2 log of each element in x.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    out : Tensor, optional
        Output tensor.

    Returns
    -------
    ret : Tensor
        Base 2 logarithm of each element in x.

    Examples
    --------
    >>> x = torch.tensor([4.0])
    >>> y = log2(x)
    >>> y
    tensor([2.])

    This computes log2(4) which is 2.
    """
    x = _cast_for_unary_op(x)
    return torch.log2(x, out=out)


@with_unsupported_dtypes({"2.0.1 and below": ("float16", "complex")}, backend_version)
@handle_numpy_arrays_in_specific_backend
def log1p(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Computes the natural logarithm of one plus the input tensor,
    elementwise.

    This function calculates the natural log of each element in the input tensor x
    plus 1. This function is more accurate than torch.log(torch.add(x, 1)) for small x so
    that 1+x/x ≈ 1.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    out : Tensor, optional
        Output tensor.

    Returns
    -------
    Tensor
        Natural log of one plus the input tensor computed element-wise.

    Examples
    --------
    >>> x = torch.tensor([0., 1., 2.])
    >>> torch.log1p(x)
    tensor([0.00000000e+00, 6.93147212e-01, 1.09861229e+00])
    """
    x = _cast_for_unary_op(x)
    return torch.log1p(x, out=out)


log1p.support_native_out = True


@handle_numpy_arrays_in_specific_backend
def isnan(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Checks element-wise if the values are NaN or not.

    Parameters
    ----------
    x : Tensor
        The input tensor.

    Returns
    -------
    Boolean tensor
        A boolean tensor with the same shape as x indicating if each element is NaN.

    Examples
    --------
    >>> x = torch.tensor([1.0, float('nan'), 2.0])
    >>> torch.isnan(x)
    tensor([False, True, False])
    """
    x = _cast_for_unary_op(x)
    return torch.isnan(x)


@with_unsupported_dtypes({"2.0.1 and below": ("complex",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def less(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compares two values element-wise and returns a boolean tensor indicating
    where x1 is less than x2.

    Parameters
    ----------
    x1 : float or array_like
        The first input array.
    x2 : float or array_like
        The second input array. Must be able to broadcast with x1.

    Returns
    -------
    result : bool Tensor
        A boolean tensor containing True where x1 is less than x2 and False otherwise.

    Examples
    --------
    >>> x1 = torch.tensor([1, 2, 3])
    >>> x2 = torch.tensor([3, 2, 1])
    >>> torch.less(x1, x2)
    tensor([True, False, False])
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
    """Multiplies two arrays or scalars elementwise.

    This function promotes inputs to a common data type and returns the
    multiplication of the inputs.

    Parameters
    ----------
    x1 : array_like
        The first input array or scalar.
    x2: array_like
        The second input array or scalar.

    Returns
    -------
    ret : ndarray
        An array containing the elementwise multiplication.

    Examples
    --------
    >>> x = torch.tensor([1, 2, 3])
    >>> y = torch.tensor([3, 4, 5])
    >>> torch.multiply(x, y)
    tensor([ 3,  8, 15])
    """
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return torch.multiply(x1, x2, out=out)


multiply.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def cos(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Computes the cosine of the input array, element-wise.

    Parameters
    ----------
    x : Tensor
        Input array containing elements to compute the cosine of.

    out : Tensor, optional
        Output tensor.

    Returns
    -------
    ret : Tensor
        The cosine of each element in x.

    Examples
    --------
    >>> import torch
    >>> x = torch.tensor([0, math.pi/2, math.pi])
    >>> torch.cos(x)
    tensor([ 1.0000,  6.1232e-17, -1.0000])
    """
    x = _cast_for_unary_op(x)
    return torch.cos(x, out=out)


cos.support_native_out = True


@handle_numpy_arrays_in_specific_backend
def logical_not(
    x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Logical NOT function.

    Computes the logical NOT of the input tensor element-wise.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    ret : torch.Tensor
        Output tensor containing element-wise NOT of x. Same shape and dtype as x.

    Examples
    --------
    >>> x = torch.tensor([True, False])
    >>> ivy.logical_not(x)
    tensor([False, True])
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
    """Divides x1 by x2 elementwise.

    Promotes inputs to a common dtype and divides x1 by x2.

    Parameters
    ----------
    x1 : float or array_like
        The dividend.
    x2 : float or array_like
        The divisor.
    out : optional array_like
        Output tensor. Must be able to cast x1 and x2 dtypes to out dtype.

    Returns
    -------
    ret : Tensor
        The quotient x1/x2, promoted to a common dtype.

    Examples
    --------
    >>> x1 = torch.tensor([3., 6.])
    >>> x2 = torch.tensor([6., 3.])
    >>> divide(x1, x2)
    tensor([0.5000, 2.0000])

    >>> x1 = torch.tensor([3, 6], dtype=torch.int32)
    >>> x2 = torch.tensor([6, 3], dtype=torch.int32)
    >>> divide(x1, x2)
    tensor([0., 2.], dtype=torch.float32)
    """
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    ret = torch.div(x1, x2)
    if ivy.is_float_dtype(x1.dtype) or ivy.is_complex_dtype(x1.dtype):
        ret = ivy.astype(ret, x1.dtype, copy=False)
    else:
        ret = ivy.astype(ret, ivy.default_float_dtype(as_native=True), copy=False)
    return ret


divide.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("complex",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def greater(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return torch.greater(x1, x2, out=out)


greater.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("complex",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def greater_equal(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Checks element-wise equality between two arrays.

    Parameters
    ----------
    x1 : array_like
        First input array to compare.
    x2 : array_like
        Second input array to compare.

    Returns
    -------
    ret : ndarray
        Boolean array of same shape as inputs indicating equality.

    Examples
    --------
    >>> x = torch.tensor([1, 2, 3])
    >>> y = torch.tensor([1, 2, 3])
    >>> torch.equal(x, y)
    tensor([True, True, True])

    >>> x = torch.tensor([1, 2, 3])
    >>> y = torch.tensor([1, 4, 3])
    >>> torch.equal(x, y)
    tensor([True, False, True])
    """
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return torch.greater_equal(x1, x2, out=out)


greater_equal.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def acos(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Computes the cosine of the input array, element-wise.

    Parameters
    ----------
    x : Tensor
        Input array containing elements to compute the cosine of.

    out : Tensor, optional
        Output tensor.

    Returns
    -------
    ret : Tensor
        The cosine of each element in x.

    Examples
    --------
    >>> import torch
    >>> x = torch.tensor([0, math.pi/2, math.pi])
    >>> torch.cos(x)
    tensor([ 1.0000,  6.1232e-17, -1.0000])
    """
    x = _cast_for_unary_op(x)
    return torch.acos(x, out=out)


acos.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("float",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def lcm(
    x1: torch.Tensor,
    x2: torch.Tensor,
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Computes the element-wise least common multiple (LCM) of input tensors
    x1 and x2.

    This function calculates the LCM for each corresponding pair of elements in x1 and x2.
    The LCM is the smallest positive integer that is divisible by both x1 and x2.

    Parameters
    ----------
    x1 : Tensor
        The first input tensor.
    x2 : Tensor
        The second input tensor. Must be broadcastable with x1.

    out : Tensor, optional
        Optional output tensor to store the result.

    Returns
    -------
    Tensor
        The element-wise LCM of x1 and x2.

    Examples
    --------
    >>> x1 = torch.tensor([5, 10, 15])
    >>> x2 = torch.tensor([3, 5, 10])
    >>> torch.lcm(x1, x2)
    tensor([15, 10, 30])

    >>> x1 = torch.tensor([2, 4, 6])
    >>> x2 = torch.tensor(3)
    >>> torch.lcm(x1, x2)
    tensor([6, 12, 6])
    """
    x1, x2 = promote_types_of_inputs(x1, x2)
    return torch.lcm(x1, x2, out=out)


lcm.support_native_out = True


@handle_numpy_arrays_in_specific_backend
def logical_xor(
    x1: torch.Tensor, x2: torch.Tensor, /, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Performs logical XOR on two input tensors x1 and x2.

    This function computes the truth value of x1 XOR x2, performing an element-wise
    logical XOR operation.

    Parameters
    ----------
    x1 : torch.Tensor
        The first input tensor.
    x2 : torch.Tensor
        The second input tensor, must be able to broadcast with x1.

    Returns
    -------
    ret : torch.Tensor
        A tensor containing the result of x1 XOR x2.

    Examples
    --------
    >>> x = torch.tensor([True, False, True])
    >>> y = torch.tensor([False, True, False])
    >>> logical_xor(x, y)
    tensor([ True,  True,  True])

    >>> x = torch.tensor([1, 0, 1])
    >>> y = torch.tensor([0, 1, 0])
    >>> logical_xor(x, y)
    tensor([ True,  True,  True])
    """
    return torch.logical_xor(x1.type(torch.bool), x2.type(torch.bool), out=out)


logical_xor.support_native_out = True


@handle_numpy_arrays_in_specific_backend
def logical_and(
    x1: torch.Tensor, x2: torch.Tensor, /, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Computes the natural logarithm of the input tensor x element-wise.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    out : torch.Tensor, optional
        Output tensor.

    Returns
    -------
    ret : torch.Tensor
        The natural logarithm of x, element-wise.

    Examples
    --------
    >>> x = torch.tensor([1., 2., math.e])
    >>> y = torch.log(x)
    >>> y
    tensor([0., 0.6931, 1.])

    This computes the natural logarithm (base e) of each element in x.
    """
    return torch.logical_and(x1.type(torch.bool), x2.type(torch.bool), out=out)


logical_and.support_native_out = True


@handle_numpy_arrays_in_specific_backend
def logical_or(
    x1: torch.Tensor, x2: torch.Tensor, /, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Performs logical OR operation on input arrays element-wise.

    Conceptually this is equivalent to calling `bool(x1) | bool(x2)`, but it is
    implemented more efficiently using bitwise operations.

    Parameters
    ----------
    x1 : array_like
        First input array.
    x2 : array_like
        Second input array. Must have the same shape as x1.

    out : ndarray, optional
        Output array with the result of the logical OR operation. Must have a shape
        that can broadcast with both x1 and x2.

    Returns
    -------
    ret : ndarray
        Output array containing the element-wise results. The values True and False
        are cast to 1 and 0 respectively.

    Examples
    --------
    >>> x1 = np.array([True, False, True])
    >>> x2 = np.array([False, True, False])
    >>> np.logical_or(x1, x2)
    array([ True, True, True])

    >>> x1 = np.array([1, 0, 1], dtype=bool)
    >>> x2 = np.array([0, 1, 0], dtype=bool)
    >>> np.logical_or(x1, x2)
    array([ True, True, True])
    """
    return torch.logical_or(x1.type(torch.bool), x2.type(torch.bool), out=out)


logical_or.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def acosh(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Computes the hyperbolic cosine of the input array.

    Parameters
    ----------
    x : torch.Tensor
        Input array.
    out : torch.Tensor, optional
        Output tensor to store the result.

    Returns
    -------
    ret : torch.Tensor
        The hyperbolic cosine of the input array computed element-wise.

    Examples
    --------
    >>> x = torch.tensor([1., 2., 3.])
    >>> y = cosh(x)
    >>> print(y)
    tensor([1.5431, 3.7622, 10.0677])
    """
    x = _cast_for_unary_op(x)
    return torch.acosh(x, out=out)


acosh.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def sin(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Computes the sine of the input tensor.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor, in radians.
    out : torch.Tensor, optional
        Output tensor to store the result.

    Returns
    -------
    torch.Tensor
        The sine of each element of the input tensor.

    Examples
    --------
    >>> x = torch.tensor([0., math.pi/2])
    >>> torch.sin(x)
    tensor([0., 1.])
    """
    x = _cast_for_unary_op(x)
    return torch.sin(x, out=out)


sin.support_native_out = True


@handle_numpy_arrays_in_specific_backend
def negative(
    x: Union[float, torch.Tensor], /, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Negates the input tensor elementwise.

    Parameters
    ----------
    x : float or torch.Tensor
        Input tensor to negate.

    out : torch.Tensor, optional
        Optional output tensor to write the result to.

    Returns
    -------
    ret : torch.Tensor
        Negated input tensor.

    Examples
    --------
    >>> x = torch.tensor([1., 2., -3.])
    >>> negative(x)
    tensor([-1., -2., 3.])
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
    """Checks element-wise equality between two arrays.

    Parameters
    ----------
    x1 : array_like
        First input array to compare.
    x2 : array_like
        Second input array to compare.

    Returns
    -------
    ret : ndarray
        Boolean array of same shape as inputs indicating equality.

    Examples
    --------
    >>> x = torch.tensor([1, 2, 3])
    >>> y = torch.tensor([1, 2, 3])
    >>> torch.equal(x, y)
    tensor([True, True, True])

    >>> x = torch.tensor([1, 2, 3])
    >>> y = torch.tensor([1, 4, 3])
    >>> torch.equal(x, y)
    tensor([True, False, True])
    """
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return torch.not_equal(x1, x2, out=out)


not_equal.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def tanh(
    x: torch.Tensor, /, *, complex_mode="jax", out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Computes the hyperbolic tangent of the input element-wise.

    The hyperbolic tangent function is defined as `tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))`.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    out : Tensor, optional
        Output tensor to store the result.

    Returns
    -------
    ret : Tensor
        The hyperbolic tangent of `x`. This will have the same shape and dtype as the input `x`.

    Examples
    --------
    >>> x = torch.tensor([-1., 0., 1.])
    >>> torch.tanh(x)
    tensor([-0.7616,  0.0000,  0.7616])

    >>> x = torch.randn(2, 3)
    >>> y = torch.tanh(x)
    >>> y.shape
    torch.Size([2, 3])
    """
    x = _cast_for_unary_op(x)
    return torch.tanh(x, out=out)


tanh.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("float16", "complex")}, backend_version)
@handle_numpy_arrays_in_specific_backend
def floor_divide(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Computes the floor of each element in the input tensor.

    The floor of the scalar x is the largest integer i, such that i <= x.

    Parameters
    ----------
    x : Tensor
        The input tensor.

    out : Optional Tensor, optional
        A tensor into which the result will be placed. If specified, the input
        tensor is cast to out.dtype before the operation is performed. This is useful
        for preventing data type overflows. Default is None.

    Returns
    -------
    ret : Tensor
        A tensor of the same shape as x, containing floor values.

    Examples
    --------
    >>> x = torch.tensor([1.7, -2.4, 3.5])
    >>> ivy.floor(x)
    tensor([ 1., -3.,  3.])

    >>> x = torch.tensor([3, -2, -1.5])
    >>> ivy.floor(x, out=torch.tensor([0., 0., 0.], dtype=torch.float32))
    tensor([3., -2., -2.], dtype=torch.float32)
    """
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    if ivy.exists(out):
        if not ivy.is_float_dtype(out):
            return ivy.inplace_update(
                out, torch.floor(torch.div(x1, x2)).type(out.dtype)
            )
    return torch.floor(torch.div(x1, x2), out=out).type(x1.dtype)


floor_divide.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("complex",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def bitwise_or(
    x1: Union[int, bool, torch.Tensor],
    x2: Union[int, bool, torch.Tensor],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Performs a bitwise OR operation element-wise on input arrays x1 and x2.

    Bits that are set in either x1 or x2 will be set in the output.

    Parameters
    ----------
    x1 : int or bool or tensor
        First input array for bitwise OR operation.
    x2 : int or bool or tensor
        Second input array for bitwise OR operation, must be same shape as x1.

    Returns
    -------
    tensor
        Tensor containing the bitwise OR between x1 and x2.

    Examples
    --------
    >>> x1 = torch.tensor([1, 0, 1])
    >>> x2 = torch.tensor([0, 1, 0])
    >>> torch.bitwise_or(x1, x2)
    tensor([1, 1, 1])

    >>> x1 = True
    >>> x2 = False
    >>> torch.bitwise_or(x1, x2)
    tensor(True)
    """
    x1, x2 = ivy.promote_types_of_inputs(x1, x2, array_api_promotion=True)
    return torch.bitwise_or(x1, x2, out=out)


bitwise_or.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def sinh(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Computes the sine of the input tensor.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor, in radians.
    out : torch.Tensor, optional
        Output tensor to store the result.

    Returns
    -------
    torch.Tensor
        The sine of each element of the input tensor.

    Examples
    --------
    >>> x = torch.tensor([0., math.pi/2])
    >>> torch.sin(x)
    tensor([0., 1.])
    """
    x = _cast_for_unary_op(x)
    return torch.sinh(x, out=out)


sinh.support_native_out = True


@handle_numpy_arrays_in_specific_backend
def positive(
    x: Union[float, torch.Tensor], /, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Returns a new tensor with only the positive values from the input
    tensor.

    Parameters
    ----------
    x : array_like
        The input array.

    Returns
    -------
    ret : ndarray
        An array with the same shape and type as x, containing only the
        positive values from x.

    Examples
    --------
    >>> x = torch.tensor([-1., 0., 1.])
    >>> positive(x)
    tensor([0., 1.])
    """
    x = _cast_for_unary_op(x)
    return torch.positive(x)


@handle_numpy_arrays_in_specific_backend
def square(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Computes the element-wise square of the input.

    Squares each element in the input `x`.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor to square.

    out : torch.Tensor, optional
        Output tensor to save the result in, by default None.

    Returns
    -------
    ret : torch.Tensor
        An array of the same shape as `x` containing the squared values.

    Examples
    --------
    >>> x = torch.tensor([1, 2, 3])
    >>> square(x)
    tensor([1, 4, 9])
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
    """Computes the power of one tensor to another.

    This function takes two tensors `x1` and `x2` and returns a tensor containing `x1` raised to the power of `x2`.

    Parameters
    ----------
    x1 : torch.Tensor
        The base tensor to be raised.
    x2 : Union[int, float, torch.Tensor]
        The exponent tensor. Must be able to broadcast to the shape of `x1`.

    out : Optional[torch.Tensor]
        Optional output tensor to store the result.

    Returns
    -------
    ret : torch.Tensor
        A tensor containing the values of `x1` to the power of `x2`.

    Examples
    --------
    >>> x = torch.tensor([1, 2])
    >>> pow(x, 2)
    tensor([1, 4])

    >>> x = torch.tensor([1, 2])
    >>> y = torch.tensor([3, 4])
    >>> pow(x, y)
    tensor([1, 16])
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


@with_unsupported_dtypes({"2.0.1 and below": ("float16", "complex")}, backend_version)
@handle_numpy_arrays_in_specific_backend
def round(
    x: torch.Tensor, /, *, decimals: int = 0, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Rounds the values in input tensor x to the nearest integer.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor to be rounded.
    decimals : int, optional
        Number of decimals to round to, by default 0
        If decimals is negative, it specifies the number of positions to the left of the decimal point.
    out : torch.Tensor, optional
        Output tensor.

    Returns
    -------
    torch.Tensor
        A tensor of the same shape as x, containing the rounded values.

    Examples
    --------
    >>> x = torch.tensor([1.2, 2.7])
    >>> round(x)
    tensor([1., 3.])

    >>> x = torch.tensor([12.352, 27.648])
    >>> round(x, 1)
    tensor([12.4, 27.6])

    >>> x = torch.tensor([123.456, 789.012])
    >>> round(x, -1)
    tensor([120., 790.])
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


@with_unsupported_dtypes({"2.0.1 and below": ("float16", "complex")}, backend_version)
@handle_numpy_arrays_in_specific_backend
def trunc(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Truncates each element in the input to the nearest integer not smaller
    in magnitude.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    out : optional Tensor, None
        Output tensor.

    Returns
    -------
    Tensor
        A new tensor with the truncated values.

    Examples
    --------
    >>> a = torch.tensor([1.2, 2.3])
    >>> torch.trunc(a)
    tensor([1., 2.])

    This truncates each value, so 1.2 becomes 1 and 2.3 becomes 2.
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
    """Computes the absolute value element-wise for the input array.

    Replaces all negative values in x with their corresponding positive values.

    Parameters
    ----------
    x : array_like
        Input array containing numbers whose absolute values are required.

    out : ndarray, optional
        Alternate output array in which to place the result. Must be of the same shape and buffer length as the expected output.

    Returns
    -------
    ret : ndarray
        An array containing the absolute values of the elements in x.
        This is a scalar if x is a scalar.

    Examples
    --------
    >>> x = np.array([-1.2, 3.4, -5.6])
    >>> np.abs(x)
    array([1.2, 3.4, 5.6])

    >>> x = np.abs(-1.2)
    >>> x
    1.2
    """
    x = _cast_for_unary_op(x)
    if x.dtype is torch.bool:
        if ivy.exists(out):
            return ivy.inplace_update(out, x)
        return x
    return torch.abs(x, out=out)


abs.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("float16", "complex")}, backend_version)
@handle_numpy_arrays_in_specific_backend
def logaddexp(
    x1: torch.Tensor, x2: torch.Tensor, /, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Adds two tensors element-wise.

    This function supports broadcasting.

    Parameters
    ----------
    x1 : Tensor
        The first tensor to be added.
    x2 : Tensor
        The second tensor to be added.

    Keyword Arguments
    -----------------
    alpha : float or int, optional
        A scaling factor for the second tensor `x2`. Default is 1.
    out : Tensor, optional
        Output tensor. The result will be placed in this tensor.

    Returns
    -------
    Tensor
        The sum of the two input tensors.

    Examples
    --------
    >>> x = torch.tensor([1, 2])
    >>> y = torch.tensor([3, 4])
    >>> torch.add(x, y)
    tensor([4., 6.])

    >>> x = torch.tensor([1, 2])
    >>> y = torch.tensor([3, 4])
    >>> torch.add(x, y, alpha=2)
    tensor([7., 10.])
    """
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return torch.logaddexp(x1, x2, out=out)


logaddexp.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def logaddexp2(
    x1: Union[torch.Tensor, float, list, tuple],
    x2: Union[torch.Tensor, float, list, tuple],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Adds two tensors element-wise.

    This function supports broadcasting.

    Parameters
    ----------
    x1 : Tensor
        The first tensor to be added.
    x2 : Tensor
        The second tensor to be added.

    Keyword Arguments
    -----------------
    alpha : float or int, optional
        A scaling factor for the second tensor `x2`. Default is 1.
    out : Tensor, optional
        Output tensor. The result will be placed in this tensor.

    Returns
    -------
    Tensor
        The sum of the two input tensors.

    Examples
    --------
    >>> x = torch.tensor([1, 2])
    >>> y = torch.tensor([3, 4])
    >>> torch.add(x, y)
    tensor([4., 6.])

    >>> x = torch.tensor([1, 2])
    >>> y = torch.tensor([3, 4])
    >>> torch.add(x, y, alpha=2)
    tensor([7., 10.])
    """
    x1, x2 = promote_types_of_inputs(x1, x2)
    if not ivy.is_float_dtype(x1):
        x1 = x1.type(ivy.default_float_dtype(as_native=True))
        x2 = x2.type(ivy.default_float_dtype(as_native=True))
    return torch.logaddexp2(x1, x2, out=out)


logaddexp2.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def tan(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Computes the tangent of the input array or tensor, element-wise.

    This function wraps `torch.tan()` and handles list/tuple inputs as well as
    Numpy array and PyTorch tensor inputs.

    Parameters
    ----------
    x : torch.Tensor
        The input tensor or array.
    out : torch.Tensor, optional
        An optional output tensor to hold the result.

    Returns
    -------
    torch.Tensor
        The tangent of each element of the input.

    Examples
    --------
    >>> x = torch.tensor([0, 0.5*pi, pi])
    >>> tan(x)
    tensor([0.0000, -1.6331, -0.0000])

    >>> tan(0)
    0.0
    """
    x = _cast_for_unary_op(x)
    return torch.tan(x, out=out)


tan.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def atan(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Computes the tangent of the input array or tensor, element-wise.

    This function wraps `torch.tan()` and handles list/tuple inputs as well as
    Numpy array and PyTorch tensor inputs.

    Parameters
    ----------
    x : torch.Tensor
        The input tensor or array.
    out : torch.Tensor, optional
        An optional output tensor to hold the result.

    Returns
    -------
    torch.Tensor
        The tangent of each element of the input.

    Examples
    --------
    >>> x = torch.tensor([0, 0.5*pi, pi])
    >>> tan(x)
    tensor([0.0000, -1.6331, -0.0000])

    >>> tan(0)
    0.0
    """
    x = _cast_for_unary_op(x)
    return torch.atan(x, out=out)


atan.support_native_out = True


@with_unsupported_dtypes(
    {"2.0.1 and below": ("float16", "bfloat16", "complex")}, backend_version
)  # TODO Fixed in PyTorch 1.12.1 (this note excludes complex)
@handle_numpy_arrays_in_specific_backend
def atan2(
    x1: torch.Tensor, x2: torch.Tensor, /, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Computes the tangent of the input array or tensor, element-wise.

    This function wraps `torch.tan()` and handles list/tuple inputs as well as
    Numpy array and PyTorch tensor inputs.

    Parameters
    ----------
    x : torch.Tensor
        The input tensor or array.
    out : torch.Tensor, optional
        An optional output tensor to hold the result.

    Returns
    -------
    torch.Tensor
        The tangent of each element of the input.

    Examples
    --------
    >>> x = torch.tensor([0, 0.5*pi, pi])
    >>> tan(x)
    tensor([0.0000, -1.6331, -0.0000])

    >>> tan(0)
    0.0
    """
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return torch.atan2(x1, x2, out=out)


atan2.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def log(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Computes the natural logarithm of the input tensor x element-wise.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    out : torch.Tensor, optional
        Output tensor.

    Returns
    -------
    ret : torch.Tensor
        The natural logarithm of x, element-wise.

    Examples
    --------
    >>> x = torch.tensor([1., 2., math.e])
    >>> y = torch.log(x)
    >>> y
    tensor([0., 0.6931, 1.])

    This computes the natural logarithm (base e) of each element in x.
    """
    x = _cast_for_unary_op(x)
    return torch.log(x, out=out)


log.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def exp(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
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
    """Exponential squared.

    Calculates 2**x for each element in the input tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    out : Tensor, optional
        Output tensor.

    Returns
    -------
    ret : Tensor
        Element-wise 2 raised to the power of x.

    Examples
    --------
    >>> x = torch.tensor([1., 2., 3.])
    >>> ivy.exp2(x)
    tensor([2., 4., 8.])

    This is equivalent to:

    >>> torch.pow(2, x)
    tensor([2., 4., 8.])
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
    """Subtracts arguments element-wise.

    This function subtracts the two input arrays element-wise, optionally multiplying
    the difference by alpha before returning.

    Parameters
    ----------
    x1 : array_like
        The first input array.
    x2 : array_like
        The second input array.
    alpha : float or int, optional
        A scalar factor to multiply the difference by before returning. Default is 1.
    out : array_like, optional
        An output array to store the result in.

    Returns
    -------
    ret : array_like
        The difference of the two input arrays, multiplied by alpha.

    Examples
    --------
    >>> x = torch.tensor([1, 2, 3])
    >>> y = torch.tensor([3, 2, 1])
    >>> torch.subtract(x, y)
    tensor([-2, 0, 2])

    >>> x = torch.tensor([1, 2, 3])
    >>> y = torch.tensor([3, 2, 1])
    >>> torch.subtract(x, y, alpha=2)
    tensor([-4, 0, 4])
    """
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    if alpha not in (1, None):
        return torch.subtract(x1, x2, alpha=alpha, out=out)
    return torch.subtract(x1, x2, out=out)


subtract.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("float16", "complex")}, backend_version)
@handle_numpy_arrays_in_specific_backend
def remainder(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    /,
    *,
    modulus: bool = True,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Computes the remainder of division for each element in x1 by the
    corresponding element in x2.

    Parameters
    ----------
    x1 : array_like
        Dividend array. Must be broadcastable with x2.
    x2 : array_like
        Divisor array. Must be broadcastable with x1.
    modulus : bool, optional
        If True, the absolute value of the remainder is returned, else the raw
        remainder is returned. Default is True.
    out : ndarray, optional
        Output array for the result. Must be broadcastable with input arrays.

    Returns
    -------
    ret : array_like
        The element-wise remainder resulting from the floor division of x1 by x2.
        Its dtype is float64 or result dtype identified by input types.

    Examples
    --------
    >>> x1 = np.array([4, 7, -5, 2])
    >>> x2 = np.array([2, 3, 2, 3])
    >>> np.remainder(x1, x2)
    array([0, 1, -1,  2])

    >>> x1 = np.array([4, 7, -5, 2])
    >>> x2 = np.array([2, 3, 2, 3])
    >>> np.remainder(x1, x2, modulus=False)
    array([0, 1, 1, -1])
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


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def atanh(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Computes the hyperbolic tangent of the input element-wise.

    The hyperbolic tangent function is defined as `tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))`.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    out : Tensor, optional
        Output tensor to store the result.

    Returns
    -------
    ret : Tensor
        The hyperbolic tangent of `x`. This will have the same shape and dtype as the input `x`.

    Examples
    --------
    >>> x = torch.tensor([-1., 0., 1.])
    >>> torch.tanh(x)
    tensor([-0.7616,  0.0000,  0.7616])

    >>> x = torch.randn(2, 3)
    >>> y = torch.tanh(x)
    >>> y.shape
    torch.Size([2, 3])
    """
    x = _cast_for_unary_op(x)
    return torch.atanh(x, out=out)


atanh.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("complex",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def bitwise_right_shift(
    x1: Union[int, bool, torch.Tensor],
    x2: Union[int, bool, torch.Tensor],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Performs a bitwise right shift on x1 by x2 number of bits.

    Shifts the bits of x1 to the right by x2 places. This is equivalent to
    floor dividing x1 by 2**x2 without casting x1 and x2 to a different type.

    Parameters
    ----------
    x1 : int, bool, or array_like
        The value to shift right.
    x2 : int, bool, or array_like
        The number of bits to shift x1 to the right.

    Returns
    -------
    ret : tensor
        x1 bitwise right shifted by x2.

    Examples
    --------
    >>> x1 = 60   # binary: 0011 1100
    >>> x2 = 2
    >>> torch.bitwise_right_shift(x1, x2)
    15 # binary: 0000 1111
    """
    x1, x2 = ivy.promote_types_of_inputs(x1, x2, array_api_promotion=True)
    x2 = torch.clamp(x2, min=0, max=torch.iinfo(x2.dtype).bits - 1)
    return torch.bitwise_right_shift(x1, x2, out=out)


bitwise_right_shift.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("complex",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def bitwise_left_shift(
    x1: Union[int, bool, torch.Tensor],
    x2: Union[int, bool, torch.Tensor],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Performs a bitwise left shift operation.

    Shifts the bits of x1 to the left by x2 number of bits. Vacated bits are
    filled with zeros.

    Parameters
    ----------
    x1 : int, bool or array_like
        Input array containing integers.
    x2 : int, bool or array_like
        Number of bits to shift x1. Must be non-negative.

    Returns
    -------
    ret : ndarray
        Return x1 with bits shifted left by x2.

    Examples
    --------
    >>> x1 = 5
    >>> x2 = 2
    >>> bitwise_left_shift(x1, x2)
    20

    >>> x1 = [1, 2, 4]
    >>> x2 = 1
    >>> bitwise_left_shift(x1, x2)
    [2, 4, 8]
    """
    x1, x2 = ivy.promote_types_of_inputs(x1, x2, array_api_promotion=True)
    return torch.bitwise_left_shift(x1, x2, out=out)


bitwise_left_shift.support_native_out = True


# Extra #
# ------#


@with_unsupported_dtypes({"2.0.1 and below": ("float16", "complex")}, backend_version)
@handle_numpy_arrays_in_specific_backend
def erf(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Computes the Gauss error function of each element of the input tensor.

    The Gauss error function is defined as:

        erf(x) = (2/sqrt(pi))*integral_from_0_to_x exp(-t**2) dt

    Parameters
    ----------
    x : Tensor
        The input tensor.

    out : Tensor, optional
        Output tensor.

    Returns
    -------
    Tensor
        The Gauss error function of each element of x.

    Examples
    --------
    >>> x = torch.tensor([0., -1., 2.])
    >>> y = erf(x)
    >>> y
    tensor([0.0000, -0.8427, 0.9953])
    """
    x = _cast_for_unary_op(x)
    return torch.erf(x, out=out)


erf.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("complex",)}, backend_version)
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


@with_unsupported_dtypes({"2.0.1 and below": ("complex",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def maximum(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    /,
    *,
    use_where: bool = True,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Determines element-wise maximum of two arrays.

    Compares two arrays and returns a new array containing the element-wise maxima.
    If one of the elements being compared is a NaN, then that element is returned.
    If both elements are NaNs then the first is returned.
    The latter distinction is important for complex NaNs, which are defined as
    at least one of the real or imaginary parts being a NaN.

    Parameters
    ----------
    x1 : array_like
        First input array. Can be any array-like object.
    x2 : array_like
        Second input array. Must be broadcastable to the same shape as `x1`.
    use_where : bool, optional
        Whether to use `torch.where` or `torch.maximum`. Default is True.
        If True, `torch.where(x1 >= x2, x1, x2)` is returned.
        If False, `torch.maximum(x1, x2)` is returned.
    out : ndarray, optional
        Alternate array to store the output. Must have the same shape as the expected output.

    Returns
    -------
    maxima : ndarray
        The element-wise maxima of `x1` and `x2`.

    Examples
    --------
    >>> x1 = torch.tensor([1, 2, 3])
    >>> x2 = torch.tensor([2, 1, 4])
    >>> torch.maximum(x1, x2)
    tensor([2, 2, 4])

    >>> x1 = torch.tensor([1, float('nan'), 3])
    >>> x2 = torch.tensor([2, float('nan'), 4])
    >>> torch.maximum(x1, x2)
    tensor([ 2., nan,  4.])
    """
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    if use_where:
        return torch.where(x1 >= x2, x1, x2, out=out)
    return torch.maximum(x1, x2, out=out)


maximum.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def reciprocal(
    x: Union[float, torch.Tensor], /, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Computes the reciprocal of the input tensor.

    This function computes 1/x for each element x in the input tensor.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    out : torch.Tensor, optional
        Alternate output tensor.

    Returns
    -------
    ret : torch.Tensor
        A new tensor holding the reciprocal values.

    Examples
    --------
    >>> x = torch.tensor([4.0, 6.0])
    >>> torch.reciprocal(x)
    tensor([0.2500, 0.1667])

    This computes the reciprocal of x.
    """
    x = _cast_for_unary_op(x)
    return torch.reciprocal(x, out=out)


reciprocal.support_native_out = True


@with_unsupported_dtypes(
    {"2.0.1 and below": ("complex64", "complex128")}, backend_version
)
@handle_numpy_arrays_in_specific_backend
def deg2rad(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Converts angle from degrees to radians element-wise.

    Parameters
    ----------
    x : torch.Tensor
        Input array in degrees.
    out : torch.Tensor, optional
        Output tensor for the result.

    Returns
    -------
    ret : torch.Tensor
        The converted angle in radians with same shape as input x.

    Examples
    --------
    >>> x = torch.tensor([180., 270, 360])
    >>> y = deg2rad(x)
    >>> y
    tensor([3.1416, 4.7124, 6.2832])
    """
    return torch.deg2rad(x, out=out)


deg2rad.support_native_out = True


@with_unsupported_dtypes(
    {"2.0.1 and below": ("complex64", "complex128")}, backend_version
)
@handle_numpy_arrays_in_specific_backend
def rad2deg(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Converts angles from radians to degrees.

    Parameters
    ----------
    x : torch.Tensor
        Angle in radians.

    out : Optional[torch.Tensor]
        Optional output tensor to hold the result.

    Returns
    -------
    ret : torch.Tensor
        Angle in degrees.

    Examples
    --------
    >>> import torch
    >>> x = torch.tensor([np.pi/2, np.pi])
    >>> y = rad2deg(x)
    >>> y
    tensor([90., 180.])
    """
    return torch.rad2deg(x, out=out)


rad2deg.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("complex",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def trunc_divide(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Divides x1 by x2 elementwise.

    Promotes inputs to a common dtype and divides x1 by x2.

    Parameters
    ----------
    x1 : float or array_like
        The dividend.
    x2 : float or array_like
        The divisor.
    out : optional array_like
        Output tensor. Must be able to cast x1 and x2 dtypes to out dtype.

    Returns
    -------
    ret : Tensor
        The quotient x1/x2, promoted to a common dtype.

    Examples
    --------
    >>> x1 = torch.tensor([3., 6.])
    >>> x2 = torch.tensor([6., 3.])
    >>> divide(x1, x2)
    tensor([0.5000, 2.0000])

    >>> x1 = torch.tensor([3, 6], dtype=torch.int32)
    >>> x2 = torch.tensor([6, 3], dtype=torch.int32)
    >>> divide(x1, x2)
    tensor([0., 2.], dtype=torch.float32)
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
    """Checks whether each element of x is real-valued or not.

    Parameters
    ----------
    x : Tensor
        The tensor to check.

    Returns
    -------
    Tensor
        A boolean tensor with the same shape as x, True where x is real.

    Examples
    --------
    >>> x = torch.tensor([1+2j, 3+0j])
    >>> torch.isreal(x)
    tensor([False, True])
    """
    return torch.isreal(x)


@with_unsupported_dtypes(
    {"2.0.1 and below": ("bfloat16", "complex")},
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
    """Computes the element-wise remainder of division.

    This is the NumPy implementation of the C library function fmod, the remainder has the same sign as the dividend `x1`. It is equivalent to the Matlab(TM) rem function and should not be confused with the MATLAB(TM) mod function.

    Parameters
    ----------
    x1 : tensor_like
        Dividend tensor.
    x2 : tensor_like
        Divisor tensor.
    out : tensor, optional
        Output tensor.

    Returns
    -------
    tensor
        The remainder of the division element-wise.

    Examples
    --------
    >>> x1 = torch.tensor([-3., -2, -1, 1, 2, 3])
    >>> x2 = 2
    >>> torch.fmod(x1, x2)
    tensor([-1.,  0., -1.,  1.,  0.,  1.])

    >>> x1 = torch.tensor([5, 3], dtype=torch.float32)
    >>> x2 = torch.tensor([2], dtype=torch.float32)
    >>> torch.fmod(x1, x2)
    tensor([ 1.,  1.], dtype=torch.float32)
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
    """Calculates the greatest common divisor of the inputs `x1` and `x2`.

    The GCD is the largest integer that divides both `x1` and `x2` without remainder.

    Parameters
    ----------
    x1 : array_like
        First input array.
    x2 : array_like
        Second input array. Must be broadcastable with `x1`.

    out : Tensor, optional
        Output tensor.

    Returns
    -------
    ret : Tensor
        Element-wise gcd of `x1` and `x2`.

    Examples
    --------
    >>> x1 = torch.tensor([12, 18])
    >>> x2 = torch.tensor([4, 9])
    >>> gcd(x1, x2)
    tensor([4, 9])

    >>> x1 = 10
    >>> x2 = 15
    >>> gcd(x1, x2)
    5
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
    """Computes the angle of the complex argument.

    Parameters
    ----------
    input : torch.Tensor
        The complex input tensor.
    deg : bool, optional
        If True, returns the angle in degrees, otherwise in radians. Default is radians.
    out : torch.Tensor, optional
        The output tensor.

    Returns
    -------
    ret : torch.Tensor
        A tensor with the angles of the elements in input.

    Examples
    --------
    >>> x = torch.tensor([1+1j, -1+1j])
    >>> angle(x)
    tensor([ 0.78539816,  2.35619449])

    >>> angle(x, deg=True)
    tensor([ 45., 135.])
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
    """Replaces nan with zero and inf with finite numbers.

    This function replaces nan values in x with zero, and replaces
    posinf and neginf values with large finite numbers.

    Parameters
    ----------
    x : torch.Tensor
        Input data containing nan, inf or -inf values to replace.
    copy : bool, optional
        Whether to create a copy of x or modify x in-place. Default is True.
    nan : float, optional
        The value to replace nan values with. Default is 0.0
    posinf : float or int, optional
        The value to replace posinf values with. Default is None.
    neginf : float or int, optional
        The value to replace neginf values with. Default is None.
    out : torch.Tensor, optional
        Output tensor. Must have the same shape as x.

    Returns
    -------
    torch.Tensor
        A tensor of the same shape as x, with nan, inf and -inf values replaced.

    Examples
    --------
    >>> x = torch.tensor([1.0, 2.0, nan, -inf, inf])
    >>> ivy.nan_to_num(x)
    tensor([ 1.,  2.,  0., -3.4028235e+38,  3.4028235e+38])
    """
    if copy:
        return torch.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf, out=out)
    else:
        x = torch.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)
        return x


def real(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Returns the real component of the complex argument.

    Parameters
    ----------
    x : torch.Tensor
        Tensor to extract real component from.

    out : Optional[torch.Tensor], optional
        Output tensor.

    Returns
    -------
    ret : torch.Tensor
        Real component of x.

    Examples
    --------
    >>> x = torch.tensor(1 + 2j)
    >>> real(x)
    tensor(1.)
    """
    return torch.real(x)


@with_unsupported_dtypes({"2.0.1 and below": ("complex",)}, backend_version)
def fmax(
    x1: torch.Tensor,
    x2: torch.Tensor,
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Computes the element-wise maximum of two tensors.

    This compares two tensors element-wise and returns a new tensor
    containing the maximum value from each pair of elements.

    Parameters
    ----------
    x1 : torch.Tensor
        The first input tensor.
    x2 : torch.Tensor
        The second input tensor. Must be able to broadcast with x1.

    out : Optional[torch.Tensor], optional
        Output tensor. By default None.

    Returns
    -------
    torch.Tensor
        The element-wise maximum of x1 and x2.

    Examples
    --------
    >>> x = torch.tensor([1, 2, 3])
    >>> y = torch.tensor([3, 2, 1])
    >>> fmax(x, y)
    tensor([3, 2, 3])
    """
    x1, x2 = promote_types_of_inputs(x1, x2)
    return torch.fmax(x1, x2, out=None)


@with_unsupported_dtypes({"2.0.1 and below": ("complex",)}, backend_version)
def nansum(
    x: torch.Tensor,
    /,
    *,
    axis: Optional[Union[Tuple[int, ...], int]] = None,
    dtype: Optional[torch.dtype] = None,
    keepdims: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Computes the sum of array elements over a given axis treating Not a
    Numbers (NaNs) as zero.

    Description: This function performs a reduction operation along the given axis, computing the sum while treating NaN (Not a Number) values as zero.

    Parameters
    ----------
    x : Tensor
        Input tensor containing elements to sum.
    axis : int or tuple of ints, optional
        Axis or axes along which the sum is performed. By default, the sum is calculated over the entire array (None).
    dtype : dtype, optional
        The data type of the output tensor. By default, the data type is inferred from the input tensor.
    keepdims : bool, optional
        If this is set to True, the reduced axes are retained as dimensions with size one in the result. By default, False.
    out : Tensor, optional
        Output tensor to store the result. Must have appropriate shape and dtype.

    Returns
    -------
    Tensor
        The sum of the input tensor with NaN values treated as zero.

    Examples
    --------
    >>> x = torch.tensor([1, np.nan, 2, np.nan])
    >>> torch.nansum(x)
    tensor(3.)

    >>> x = torch.tensor([[1, 2], [np.nan, 4]])
    >>> torch.nansum(x, axis=0)
    tensor([1., 4.])
    """
    dtype = ivy.as_native_dtype(dtype)
    return torch.nansum(x, dim=axis, keepdim=keepdims, dtype=dtype)


def diff(
    x: Union[torch.Tensor, list, tuple],
    /,
    *,
    n: int = 1,
    axis: int = -1,
    prepend: Optional[Union[torch.Tensor, int, float, list, tuple]] = None,
    append: Optional[Union[torch.Tensor, int, float, list, tuple]] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Computes the n-th discrete difference along the given axis.

    The n-th differences are calculated by shifting the array by n elements along the
    given axis and subtracting the shifted values from the original array.

    Parameters
    ----------
    x : array_like
        Input array
    n : int, optional
        The number of times values are differenced. If zero, the input is returned as-is.
        Default is 1.
    axis : int, optional
        The axis along which the differences are taken, default is the last axis.
    prepend, append : array_like, scalar, or None
        Values to prepend or append to `x` along axis prior to performing the difference.
        Scalar values are expanded to arrays with length 1 in the direction of axis and the shape of the input array in along all other axes. Otherwise, the dimensions of `prepend` and `append` along axis must match that of `x`.
    out : ndarray, optional
        Alternative output array in which to place the result. Must be of the same shape and buffer length as the expected output.

    Returns
    -------
    diff : ndarray
        The n-th differences. The shape of the output is the same as `x` except along `axis` where the dimension is smaller by `n`.

    Examples
    --------
    >>> x = torch.tensor([1, 2, 4, 7, 0])
    >>> torch.diff(x)
    tensor([1, 2, 3, -7])

    >>> x = torch.tensor([[1, 3, 6, 10], [0, 5, 6, 8]])
    >>> torch.diff(x, axis=0)
    tensor([[ -1,   2,  0,  -2]])

    >>> x = torch.tensor([1, 4, np.nan, 2, 3])
    >>> torch.diff(x, prepend=0)
    tensor([3., nan, -2.,  1.])
    """
    x = x if isinstance(x, torch.Tensor) else torch.tensor(x)
    prepend = (
        prepend
        if isinstance(prepend, torch.Tensor) or prepend is None
        else torch.tensor(prepend)
    )
    append = (
        append
        if isinstance(append, torch.Tensor) or append is None
        else torch.tensor(append)
    )
    return torch.diff(x, n=n, dim=axis, prepend=prepend, append=append)
