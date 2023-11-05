import numpy as np


def one_hot(
    indices,
    depth,
    on_value=None,
    off_value=None,
    axis=None,
    dtype=None,
):
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
