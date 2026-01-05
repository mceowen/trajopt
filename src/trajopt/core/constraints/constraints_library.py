import numpy as np
import cvxpy as cp
import jax
import jax.numpy as jnp
import trajopt.core.scp.convexify as convexify
from trajopt.utils.config_loader import resolve_function

# NOTE TO SELF (CARLOS):
# let z = M_in @ x, then:
# g(z) <= max_val  <==>  M_out^-1 @ g(M_in @ x) <= M_out^-1 @ max_val

# GIVEN THAT M IS POSITIVE DIAGONAL!
# WHEN WE NONDIM WE ARE DOING BOTH SCALING OF INPUTS AND CONDITIONING OF OUTPUTS!
# only need linear scaling for now because we're using deviation variables in scp


# dynamics:
# let z = M_in_x @ x, and nu = M_in_u @ u
# z_dot = f(z, nu)
# M_in_x @ x_dot = f(M_in_x @ x, M_in_u @ u)
# dx/dt = M_in_x^-1 @ f(M_in_x @ x, M_in_u @ u) # WRT TO PHYSICAL TIME!
#
# WRT TO normalized time:
# let t = nt * tau, then:
# dx/dtau = nt * M_in_x^-1 @ f(M_in_x @ x, M_in_u @ u)

# local caobra example has better scaling 


# TODO: need to add affine approx of convex constraints for ctcs

# ===============================================================
# CONVEX CONSTRAINTS
# ===============================================================

class equality_bc:
    def __init__(self, name, set, x, x_idx, boundary, eps=None, params=None):

        # parameters
        self.name = name
        self.set = set
        self.x_dim = x
        self.x_idx = x_idx
        self.boundary = boundary
        self.idx = 0 if boundary == 'init' else -1 if boundary == 'final' else None
        if eps is not None:
            self.eps = eps
        else:
            self.eps = np.zeros(len(x_idx))
        self.dimension = len(x_idx)

        self.x = None

    # written for nondim input
    def fcn(self, x):
        return x[self.x_idx] - self.x

    def nondim_constraint(self, nondim):
        if self.set == "state":
            self.x = nondim.M["state"]["d2nd"][np.ix_(self.x_idx, self.x_idx)] @ self.x_dim
        elif self.set == "control":
            self.x = nondim.M["ctrl"]["d2nd"][np.ix_(self.x_idx, self.x_idx)] @ self.x_dim

class inequality_bc:
    def __init__(self, name, set, x_min, x_min_idx, x_max, x_max_idx, boundary, eps=np.array([]), params=None):

        # parameters
        self.name = name
        self.set = set
        self.x_min_dim = x_min
        self.x_min_idx = x_min_idx
        self.x_max_dim = x_max
        self.x_max_idx = x_max_idx
        self.idx = 0 if boundary == 'init' else -1 if boundary == 'final' else None
        self.eps = eps
        self.dimension = len(x_min_idx) + len(x_max_idx)

        self.x_min = None
        self.x_max = None

    def nondim_constraint(self, nondim):
        if self.set == "state":
            self.x_min = nondim.M["state"]["d2nd"][np.ix_(self.x_min_idx, self.x_min_idx)] @ self.x_min_dim
            self.x_max = nondim.M["state"]["d2nd"][np.ix_(self.x_max_idx, self.x_max_idx)] @ self.x_max_dim
        elif self.set == "control":
            self.x_min = nondim.M["ctrl"]["d2nd"][np.ix_(self.x_min_idx, self.x_min_idx)] @ self.x_min_dim
            self.x_max = nondim.M["ctrl"]["d2nd"][np.ix_(self.x_max_idx, self.x_max_idx)] @ self.x_max_dim

# ---------------------------------------------------------------
# path inequality constraints
# ---------------------------------------------------------------
class box:
    def __init__(self, name, set, x_min, x_min_idx, x_max, x_max_idx, params=None):
        self.name = name
        self.set = set
        self.x_min_dim = x_min
        self.x_min_idx = x_min_idx
        self.x_max_dim = x_max
        self.x_max_idx = x_max_idx
        self.dimension = len(x_min_idx) + len(x_max_idx)

        self.x_min = None
        self.x_max = None


        if self.set == "state":
            n_elem = params['model']['dimensions']['n']
        elif self.set == "control":
            n_elem = params['model']['dimensions']['m']

        M_min = -np.eye(n_elem)[self.x_min_idx, :]
        M_max = np.eye(n_elem)[self.x_max_idx, :]

        self.M_select = np.vstack([M_min, M_max])

    def nondim_constraint(self, nondim):
        if self.set == "state":
            self.x_min = nondim.M["state"]["d2nd"][np.ix_(self.x_min_idx, self.x_min_idx)] @ self.x_min_dim
            self.x_max = nondim.M["state"]["d2nd"][np.ix_(self.x_max_idx, self.x_max_idx)] @ self.x_max_dim

        if self.set == "control":
            self.x_min = nondim.M["ctrl"]["d2nd"][np.ix_(self.x_min_idx, self.x_min_idx)] @ self.x_min_dim
            self.x_max = nondim.M["ctrl"]["d2nd"][np.ix_(self.x_max_idx, self.x_max_idx)] @ self.x_max_dim

# ---------------------------------------------------------------
# rate constraints
# ---------------------------------------------------------------
class control_rate_limit:
    def __init__(self, name, udot_max, udot_max_idx, params=None):
        self.name = name
        self.udot_max_dim = udot_max
        self.udot_max_idx = udot_max_idx
        self.dimension = len(udot_max_idx)

        n_elem = params['model']['dimensions']['m']
        M_min = -np.eye(n_elem)[self.udot_max_idx, :]
        M_max = np.eye(n_elem)[self.udot_max_idx, :]

        self.udot_max = None

        self.M_select = np.vstack([M_min, M_max])

    def nondim_constraint(self, nondim):
        self.udot_max = nondim.nt * nondim.M["ctrl"]["d2nd"][np.ix_(self.udot_max_idx, self.udot_max_idx)] @ self.udot_max_dim

# ---------------------------------------------------------------
# Second-order cone cosntraints
# ---------------------------------------------------------------

class axis_angle_cone:
    def __init__(self, name, set, axis, theta_max, x_idx, params=None):
        self.name = name
        self.set = set
        self.axis = axis / np.linalg.norm(axis)
        self.cos_theta_max = np.cos(np.deg2rad(theta_max))
        self.x_idx = x_idx
        self.dimension = 1

    def nondim_constraint(self, nondim):
        # the deg2rad is already nondimming
        pass

class max_norm_cone:
    def __init__(self, name, set, max_val, x_idx, params=None):
        self.name = name
        self.set = set
        self.max_val_dim = max_val
        self.x_idx = x_idx
        self.dimension = 1

        self.max_val = None

    def nondim_constraint(self, nondim):

        if self.set == "state":
            nondim_key = nondim.z_types[self.x_idx[0]]
            self.max_val = self.max_val_dim * nondim.scales[nondim_key]
        
        elif self.set == "control":
            nondim_key = nondim.u_types[self.x_idx[0]]
            self.max_val =  self.max_val_dim * nondim.scales[nondim_key]

class quaternion_cone:
    def __init__(self, name, theta_max, axis_num, quat_start_idx, params=None):
        self.name = name
        self.quat_start_idx = quat_start_idx
        self.cos_theta_max = np.cos(np.deg2rad(theta_max))
        self.axis_num = axis_num
        self.rhs = np.sqrt((1.0 - self.cos_theta_max) * 0.5)
        self.dimension = 1

    def nondim_constraint(self, nondim):
        pass

# ===============================================================
# NONCONVEX CONSTRAINTS
# ===============================================================

# TODO: change to (func - max_val) / scale
class nonconvex_inequality:
    def __init__(self, name, group, fcn, units, dimension, ct, eps=None, max_val=None, mission_params=None, params=None):
        self.name = name
        self.group = group
        self.mission_params = mission_params
        self.params = params
        self.units = units
        self.eps = eps
        self.dimension = dimension
        self.ct = ct

        self.fcn_dim = resolve_function(fcn)
        
        # this will be a function f(t, z, nu)
        self.fcn = None
        
        self.fcn_jit = None
        self.dfcn_dz_jit = None
        self.dfcn_du_jit = None

        self.has_max_val = max_val is not None
        
        if self.has_max_val is False:
            self.max_val_dim = None
        else:
            self.max_val_dim = jnp.atleast_1d(jnp.asarray(max_val))
    
    def g_aff(self, t, z, nu):
        return self.fcn_jit(z, nu), self.dfcn_dz_jit(z, nu), self.dfcn_du_jit(z, nu)

    def nondim_constraint(self, nondim):

        if self.max_val_dim is not None:
            M_out_d2nd = np.diag(1 / self.max_val_dim)
        else:
            M_out_d2nd, M_out_nd2d = nondim.build_nondim_matrix(self.units)

        if self.has_max_val:
            self.max_val = M_out_d2nd @ self.max_val_dim

        # fcn_dim is already bound with params/fcns by resolve_functions
        nd_fcn_lhs = nondim.nondim_function(self.fcn_dim, nondim.M["state"]["nd2d"], nondim.M["ctrl"]["nd2d"], M_out_d2nd)

        if self.has_max_val:
            self.fcn = lambda t, z, nu: nd_fcn_lhs(t, z, nu) - self.max_val
        else:
            self.fcn = lambda t, z, nu: nd_fcn_lhs(t, z, nu)

class dynamics:
    def __init__(self, name, fcn, mission_params=None, params=None):
        self.name = name
        self.mission_params = mission_params
        self.params = params
        
        # dynamically loaded from user module, gets closed
        # over with params and fcns dicts if necessary
        self.fcn_dim = resolve_function(fcn)

        # this will be a function f(t, z, nu) and operates on nondim variables
        self.fcn = None

        self.fcn_jit = None
        self.dfcn_dz_jit = None
        self.dfcn_du_jit = None

    def lin_dyn(self, t, z, nu):
        return self.fcn_jit(z, nu), self.dfcn_dz_jit(z, nu), self.dfcn_du_jit(z, nu)

    def nondim_constraint(self, nondim):
        M_out_d2nd = nondim.M["state"]["d2nd"] * nondim.nt

        # fcn_dim is already bound with params/fcns by resolve_functions
        self.fcn = nondim.nondim_function(self.fcn_dim, nondim.M["state"]["nd2d"], nondim.M["ctrl"]["nd2d"], M_out_d2nd)