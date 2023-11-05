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
    if len(x.shape) > 0:
        x = tf.transpose(x)
    return tf.transpose(tf.reshape(x, shape[::-1]))