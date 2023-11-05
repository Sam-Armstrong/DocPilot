import numpy as np


def one_hot(
    indices,
    depth,
    on_value=None,
    off_value=None,
    axis=None,
    dtype=None,
):
    """Convert a vector of indices into a matrix with one-hot encodings as
    columns.

    Parameters
    ----------
    indices : array_like
        Indices to convert to one-hot encodings.
    depth : int
        Size of one-hot dimension.
    on_value : float, optional
        Value to fill in output when class is hot, by default 1.
    off_value : float, optional
        Value to fill in output when class is not hot, by default 0.
    axis : int, optional
        Axis along which one-hot encodings are added.
    dtype : data-type, optional
        Data-type of the one-hot matrix.

    Returns
    -------
    ret : ndarray
        One-hot matrix corresponding to indices.

    Examples
    --------
    >>> indices = [0, 1, 2]
    >>> one_hot(indices, depth=3)
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])
    """
    on_none = on_value is None
    off_none = off_value is None

    if dtype is None:
        if on_none and off_none:
            dtype = np.float32
        else:
            if not on_none:
                dtype = np.array(on_value).dtype
            elif not off_none:
                dtype = np.array(off_value).dtype

    res = np.eye(depth, dtype=dtype)[np.array(indices, dtype="int64").reshape(-1)]
    res = res.reshape(list(indices.shape) + [depth])

    if not on_none and not off_none:
        res = np.where(res == 1, on_value, off_value)

    if axis is not None:
        res = np.moveaxis(res, -1, axis)

    return res
