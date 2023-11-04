# global
from numbers import Number
import numpy as np
from typing import Union, Optional, List, Sequence, Tuple

import jax.dlpack
import jax.numpy as jnp
import jax._src as _src
import jaxlib.xla_extension

# local
import ivy
from ivy import as_native_dtype
from ivy.functional.backends.jax import JaxArray
from ivy.functional.ivy.creation import (
    _asarray_to_native_arrays_and_back,
    _asarray_infer_device,
    _asarray_infer_dtype,
    _asarray_handle_nestable,
    NestedSequence,
    SupportsBufferProtocol,
    _asarray_inputs_to_native_shapes,
)


# Array API Standard #
# ------------------ #

@_asarray_to_native_arrays_and_back
@_asarray_infer_device
@_asarray_handle_nestable
@_asarray_inputs_to_native_shapes
@_asarray_infer_dtype
def asarray(
    obj: Union[
        JaxArray,
        bool,
        int,
        float,
        tuple,
        NestedSequence,
        SupportsBufferProtocol,
        np.ndarray,
    ],
    /,
    *,
    copy: Optional[bool] = None,
    dtype: Optional[jnp.dtype] = None,
    device: jaxlib.xla_extension.Device = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    ivy.utils.assertions._check_jax_x64_flag(dtype)
    if copy is True:
        return jnp.array(obj, dtype=dtype, copy=True)
    else:
        return jnp.asarray(obj, dtype=dtype)