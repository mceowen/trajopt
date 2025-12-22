import numpy as np
import cvxpy as cp
import jax 
import jax.numpy as jnp

class Mission:
    def __init__(self):
        pass

    # =============================================================================
    # nonconvex inequality constraint functions
    # (used with type: nonconvex_inequality constraints)
    # =============================================================================

    def height_triggered_pitch(self, t, z, nu, params):

        # f > 0 ==> g >= 0

        f = 2.0 - z[1]
        g = 1.0 - (params["cos_theta_max"] + 2 * jnp.sum(z[9:11] ** 2))

        return jnp.array([jnp.maximum(f, 0.0) * jnp.maximum(-g, 0.0)])

    def min_thrust_norm(self, t, z, nu, params):
        
        return jnp.array(1.0 - jnp.linalg.norm(nu[:3]) / params["min_thrust"])
