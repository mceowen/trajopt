import numpy as np
import cvxpy as cp
import jax
import jax.numpy as jnp
import trajopt.core.modules.method.convexify as convexify


# TODO: need to add affine approx of convex constraints for ctcs

# ===============================================================
# CONVEX CONSTRAINTS
# ===============================================================

# ---------------------------------------------------------------
# boundary conditions
# ---------------------------------------------------------------

class equality_bc:
    def __init__(self, set, x_idx, x, boundary, name):

        # parameters
        self.set = set
        self.x_idx = x_idx
        self.x = x
        self.idx = 0 if boundary == 'init' else -1 if boundary == 'final' else None
        self.name = name

class inequality_bc:
    def __init__(self, set, x_min_idx, x_max_idx, x_min, x_max, boundary, name):

        # parameters
        self.x_min_idx = x_min_idx
        self.x_max_idx = x_max_idx
        self.x_min = x_min
        self.x_max = x_max
        self.set = set
        self.idx = 0 if boundary == 'init' else -1 if boundary == 'final' else None
        self.name = name

# ---------------------------------------------------------------
# path inequality constraints
# ---------------------------------------------------------------
class box:
    def __init__(self, set, x_min_idx, x_max_idx, x_min, x_max, name):
        self.x_min_idx = x_min_idx
        self.x_max_idx = x_max_idx
        self.x_min = x_min
        self.x_max = x_max
        self.set = set
        self.name = name

# ---------------------------------------------------------------
# rate constraints
# ---------------------------------------------------------------
class rate_limit:
    def __init__(self, xdot_max, idx):
        self.xdot_max = xdot_max
        self.idx = idx

# ---------------------------------------------------------------
# Second-order cone cosntraints
# ---------------------------------------------------------------

class axis_angle_cone:
    def __init__(self, set, axis, theta_max, idx, name):
        self.cos_theta_max = np.cos(np.deg2rad(theta_max))
        self.axis = axis / np.linalg.norm(axis)
        self.idx = idx
        self.set = set
        self.name = name

class max_norm_cone:
    def __init__(self, set, max_val, idx, name):
        self.max_val = max_val
        self.idx = idx
        self.set = set
        self.name = name

class quaternion_cone:
    def __init__(self, quat_start_idx, theta_max, axis_num, name):
        self.quat_start_idx = quat_start_idx
        self.cos_theta_max = np.cos(np.deg2rad(theta_max))
        self.axis_num = axis_num
        self.rhs = np.sqrt((1.0 - self.cos_theta_max) * 0.5)
        self.name = name

# ===============================================================
# NONCONVEX CONSTRAINTS
# ===============================================================

# TODO: change to (func - max_val) / scale
class nonconvex_inequality:
    def __init__(self, func, params, ct, group, units, eps, dimension, name):
        self.ct = ct
        self.func = func
        self.params = params
        self.group = group
        self.units = units
        self.eps = eps
        self.dimension = dimension
        self.name = name

        self.fcn_jit, self.dfcn_dz_jit, self.dfcn_du_jit = convexify.linearize_jax(self.g, self.params)

    def g(self, t, z, nu):
        return self.func(t, z, nu, self.params)
    
    def g_aff(self, t, z, nu):
        return self.fcn_jit(t, z, nu), self.dfcn_dz_jit, self.dfcn_du_jit