import numpy as np
import jax.numpy as jnp
from jax import Array

# Direction Cosine Matrix Function
def DCM(q: Array) -> Array:
    """Direction cosine matrix (inertial → body) from unit quaternion [q0, q1, q2, q3] (scalar first)."""
    return jnp.array(
        [
            [1 - 2 * (q[2] ** 2 + q[3] ** 2), 2 * (q[1] * q[2] + q[0] * q[3]), 2 * (q[1] * q[3] - q[0] * q[2])],
            [2 * (q[1] * q[2] - q[0] * q[3]), 1 - 2 * (q[1] ** 2 + q[3] ** 2), 2 * (q[2] * q[3] + q[0] * q[1])],
            [2 * (q[1] * q[3] + q[0] * q[2]), 2 * (q[2] * q[3] - q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2)],
        ],
    )

def omega(w: Array) -> Array:
    """Skew-symmetric quaternion kinematic matrix for angular velocity w."""
    return jnp.array(
        [
            [0, -w[0], -w[1], -w[2]],
            [w[0], 0, w[2], -w[1]],
            [w[1], -w[2], 0, w[0]],
            [w[2], w[1], -w[0], 0],
        ],
    )

def cr(v: Array) -> Array:
    """Skew-symmetric cross-product matrix for vector v."""
    return jnp.array(
        [
            [0, -v[2], v[1]], 
            [v[2], 0, -v[0]], 
            [-v[1], v[0], 0],
        ]
    )