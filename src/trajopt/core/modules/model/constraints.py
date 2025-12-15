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

class initial_state_equality:
    def __init__(self, zi_idx, zi):

        # parameters
        self.zi_idx = zi_idx
        self.zi = zi

class initial_state_inequality:
    def __init__(self, zi_min_idx, zi_max_idx, zi_min, zi_max):

        # parameters
        self.zi_min_idx = zi_min_idx
        self.zi_max_idx = zi_max_idx
        self.zi_min = zi_min
        self.zi_max = zi_max

class final_state_equality:
    def __init__(self, zf_idx, zf):

        # parameters
        self.zf_idx = zf_idx
        self.zf = zf

class final_state_inequality:
    def __init__(self, zf_min_idx, zf_max_idx, zf_min, zf_max):

        # parameters
        self.zf_min_idx = zf_min_idx
        self.zf_max_idx = zf_max_idx
        self.zf_min = zf_min
        self.zf_max = zf_max

class initial_control_equality:
    def __init__(self, ui_idx, ui):

        # parameters
        self.ui_idx = ui_idx
        self.ui = ui

class final_control_equality:
    def __init__(self, uf_idx, uf):

        # parameters
        self.uf_idx = uf_idx
        self.uf = uf

class final_control_inequality:
    def __init__(self, uf_min_idx, uf_max_idx, uf_min, uf_max):

        # parameters
        self.uf_min_idx = uf_min_idx
        self.uf_max_idx = uf_max_idx
        self.uf_min = uf_min
        self.uf_max = uf_max

# ---------------------------------------------------------------
# path inequality constraints
# ---------------------------------------------------------------
class state_inequality:
    def __init__(self, z_min_idx, z_max_idx, z_min, z_max):
        self.z_min_idx = z_min_idx
        self.z_max_idx = z_max_idx
        self.z_min = z_min
        self.z_max = z_max

class control_inequality:
    def __init__(self, u_min_idx, u_max_idx, u_min, u_max):
        self.u_min_idx = u_min_idx
        self.u_max_idx = u_max_idx
        self.u_min = u_min
        self.u_max = u_max

# ---------------------------------------------------------------
# rate constraints
# ---------------------------------------------------------------
class control_rate_limit:
    def __init__(self, udot_max, idx):
        self.udot_max = udot_max
        self.idx = idx

# ---------------------------------------------------------------
# Second-order cone cosntraints
# ---------------------------------------------------------------

class state_axis_angle_cone:
    def __init__(self, axis, theta_max, idx):
        self.cos_theta_max = np.cos(np.deg2rad(theta_max))
        self.axis = axis / np.linalg.norm(axis)
        self.idx = idx

class control_axis_angle_cone:
    def __init__(self, axis, theta_max, idx):
        self.cos_theta_max = np.cos(np.deg2rad(theta_max))
        self.axis = axis / np.linalg.norm(axis)
        self.idx = idx

class state_max_norm_cone:
    def __init__(self, max_val, idx):
        self.max_val = max_val
        self.idx = idx

class control_max_norm_cone:
    def __init__(self, max_val, idx):
        self.max_val = max_val
        self.idx = idx

class quaternion_cone:
    def __init__(self, quat_start_idx, theta_max, axis_num):
        self.quat_start_idx = quat_start_idx
        self.cos_theta_max = np.cos(np.deg2rad(theta_max))
        self.axis_num = axis_num
        self.rhs = np.sqrt((1.0 - self.cos_theta_max) * 0.5)

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