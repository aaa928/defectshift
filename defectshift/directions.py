"""Direction of a difference. A pair has the same direction only if both values are nonzero and have the same sign; a zero
(or a missing value) on either side is no direction, so two zeros do not count as agreeing. Works on scalars, arrays and
pandas objects."""
import numpy as np


def same_direction(a, b):
    """True where a and b are both positive or both negative."""
    return np.sign(a) * np.sign(b) > 0
