import numpy as np


def one_hot(
    indices,
    depth,
    on_value=None,
    off_value=None,
    axis=None,
    dtype=None,
):
    """Convert indices into a one-hot representation.

    Parameters
    ----------
    indices : array_like
        Integer array containing indices to convert to one-hot.
    depth : int
        Size of one hot dimension for encoding indices.
    on_value : float, optional
        Value to fill in output when indices[i] = j. Default is None.
    off_value : float, optional
        Value to fill in output when indices[i] != j. Default is None.
    axis : int, optional
        Axis along which one-hot representation is concatenated. Default is None.
    dtype : numpy dtype, optional
        Data type of the one-hot array. Default is None.

    Returns
    -------
    res : ndarray
        One-hot representation of input indices.

    Examples
    --------
    >>> indices = [0, 1, 2]
    >>> one_hot(indices, depth=3)
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])

    >>> one_hot(indices, depth=3, on_value=5)
    array([[5., 0., 0.],
           [0., 5., 0.],
           [0., 0., 5.]])

    >>> one_hot(indices, depth=3, off_value=-1)
    array([[ 1., -1., -1.],
           [-1.,  1., -1.],
           [-1., -1.,  1.]])
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
