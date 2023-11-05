def hello_world():
    """
    Prints 'Hello world!' to the console.

    This simple function exists solely to print a friendly greeting
    to the user when called. It takes no arguments and returns nothing.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Raises
    ------
    None
    """
    pass


def inplace_decrement(
    x: Union[ivy.Array, paddle.Tensor],
    val: Union[ivy.Array, paddle.Tensor],
) -> ivy.Array:
    """
    Inplace subtract a value from an array.

    This function directly subtracts val from x in place, modifying x rather than returning a new array.

    Parameters
    ----------
    x : Union[ivy.Array, paddle.Tensor]
        Input array to decrement inplace.
    val : Union[ivy.Array, paddle.Tensor]
        Value to subtract from x. Must be broadcastable with x.

    Returns
    -------
    ivy.Array
        The decremented x array after being modified inplace.

    Raises
    ------
    ValueError
        If val is not broadcastable with x.
    """
    (x_native, val_native), _ = ivy.args_to_native(x, val)
    if ivy.is_ivy_array(x):
        target = x.data
    else:
        target = x
    return paddle.assign(paddle_backend.subtract(x_native, val_native), target)


def scatter_flat(
    indices: paddle.Tensor,
    updates: paddle.Tensor,
    /,
    *,
    size: Optional[int] = None,
    reduction: str = "sum",
    out: Optional[paddle.Tensor] = None,
):
    """
    Scatters a tensor into a destination tensor according to indices.

    This functions scatters the `updates` tensor into the `out` tensor according
    to the indices specified in `indices`. The updates are added to the indices
    of the `out` tensor specified by `indices`.

    Parameters
    ----------
    indices : Tensor
        The indices into the output tensor `out` where the updates will be scattered.
    updates: Tensor
        The updates to scatter into the output tensor.
    size: int, optional
        The size of the output tensor. Default is None.
    reduction: {'sum', 'mean', 'min', 'max'}, optional
        The reduction method for the scatter, by default 'sum'.
    out: Tensor, optional
        The output tensor to scatter into. Default is None.

    Returns
    -------
    Tensor
        The tensor `out` after the scatter operation.

    Raises
    ------
    ValueError
        If reduction is not one of 'sum', 'mean', 'min' or 'max'.
    """
    if indices.dtype not in [paddle.int32, paddle.int64]:
        indices = indices.cast("int64")
    if ivy.exists(size) and ivy.exists(out):
        ivy.utils.assertions.check_equal(out.ndim, 1, as_array=False)
        ivy.utils.assertions.check_equal(out.shape[0], size, as_array=False)
    return paddle_backend.scatter_nd(
        indices.unsqueeze(-1), updates, shape=[size], reduction=reduction, out=out
    )
