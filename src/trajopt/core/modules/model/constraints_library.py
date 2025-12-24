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
    def __init__(self, name, set, x, x_idx, boundary, eps=np.array([])):

        # parameters
        self.name = name
        self.set = set
        self.x = x
        self.x_idx = x_idx
        self.boundary = boundary
        self.idx = 0 if boundary == 'init' else -1 if boundary == 'final' else None
        self.eps = eps
        self.dimension = len(x_idx)

class inequality_bc:
    def __init__(self, name, set, x_min, x_min_idx, x_max, x_max_idx, boundary, eps=np.array([])):

        # parameters
        self.name = name
        self.set = set
        self.x_min = x_min
        self.x_min_idx = x_min_idx
        self.x_max = x_max
        self.x_max_idx = x_max_idx
        self.idx = 0 if boundary == 'init' else -1 if boundary == 'final' else None
        self.eps = eps
        self.dimension = len(x_min_idx) + len(x_max_idx)
# ---------------------------------------------------------------
# path inequality constraints
# ---------------------------------------------------------------
class box:
    def __init__(self, name, set, x_min, x_min_idx, x_max, x_max_idx):
        self.name = name
        self.set = set
        self.x_min = x_min
        self.x_min_idx = x_min_idx
        self.x_max = x_max
        self.x_max_idx = x_max_idx
        self.dimension = len(x_min_idx) + len(x_max_idx)
# ---------------------------------------------------------------
# rate constraints
# ---------------------------------------------------------------
class control_rate_limit:
    def __init__(self, name, udot_max, udot_max_idx):
        self.name = name
        self.udot_max = udot_max
        self.udot_max_idx = udot_max_idx
        self.dimension = len(udot_max_idx)
# ---------------------------------------------------------------
# Second-order cone cosntraints
# ---------------------------------------------------------------

class axis_angle_cone:
    def __init__(self, name, set, axis, theta_max, x_idx):
        self.name = name
        self.set = set
        self.axis = axis / np.linalg.norm(axis)
        self.cos_theta_max = np.cos(np.deg2rad(theta_max))
        self.x_idx = x_idx
        self.dimension = 1

class max_norm_cone:
    def __init__(self, name, set, max_val, x_idx):
        self.name = name
        self.set = set
        self.max_val = max_val
        self.x_idx = x_idx
        self.dimension = 1

class quaternion_cone:
    def __init__(self, name, theta_max, axis_num, quat_start_idx):
        self.name = name
        self.quat_start_idx = quat_start_idx
        self.cos_theta_max = np.cos(np.deg2rad(theta_max))
        self.axis_num = axis_num
        self.rhs = np.sqrt((1.0 - self.cos_theta_max) * 0.5)
        self.dimension = 1

# ===============================================================
# NONCONVEX CONSTRAINTS
# ===============================================================

# TODO: change to (func - max_val) / scale
class nonconvex_inequality:
    def __init__(self, name, group, fcn, units, eps, dimension, ct, fcn_params={}, max_val=None):
        self.name = name
        self.group = group
        self.fcn_name = fcn
        self.fcn_params = fcn_params
        self.units = units
        self.eps = eps
        self.dimension = dimension
        self.ct = ct

        # the actual functions are resolved once mission and model are initialized
        self.fcn = None
        self.fcn_jit = None
        self.dfcn_dz_jit = None
        self.dfcn_du_jit = None

        if max_val is None:
            self.max_val = jnp.zeros(dimension)
        else:
            self.max_val = max_val

    def g(self, t, z, nu):
        return self.func(t, z, nu) - self.max_val
    
    def g_aff(self, t, z, nu):
        return self.fcn_jit(z, nu), self.dfcn_dz_jit(z, nu), self.dfcn_du_jit(z, nu)