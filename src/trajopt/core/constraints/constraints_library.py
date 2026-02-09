import numpy as np
import cvxpy as cp
import jax
import jax.numpy as jnp
import trajopt.library.methods.convexify as convexify
from trajopt.utils.config_loader import resolve_function

# ===============================================================
# CONVEX CONSTRAINTS
# ===============================================================

class equality_bc:
    def __init__(self, cnstr_config, config=None):
        # required properties
        self.name = cnstr_config["name"]
        self.group = cnstr_config.get("group", None)
        self.x_guess = cnstr_config.get("x_guess", None)

        # type-specific properties
        self.set = cnstr_config["set"]
        self.x = np.atleast_1d(cnstr_config["x"])

        self.x_idx = cnstr_config["x_idx"]
        self.boundary = cnstr_config["boundary"]
        
        if self.boundary == 'init':
            self.idx = 0 
        elif self.boundary == 'final':
            self.idx = -1 
        else:
            self.idx = None

        self.eps = cnstr_config.get("eps", np.zeros(len(self.x_idx)))
        self.dimension = len(self.x_idx)
        
        self.implement_type = 'equality_bc'

    def fcn(self, x):
        return x[self.x_idx] - self.x

    def nondim_constraint(self, nondim):
        if self.set == "state":
            self.x = nondim.M["state"]["d2nd"][np.ix_(self.x_idx, self.x_idx)] @ self.x
        elif self.set == "control":
            self.x = nondim.M["ctrl"]["d2nd"][np.ix_(self.x_idx, self.x_idx)] @ self.x

class inequality_bc:
    def __init__(self, cnstr_config, config=None):
        self.name = cnstr_config["name"]
        self.group = cnstr_config.get("group", None)
        self.set = cnstr_config["set"]
        self.x_min_dim = np.atleast_1d(cnstr_config["x_min"])
        self.x_min_idx = cnstr_config["x_min_idx"]
        self.x_max_dim = np.atleast_1d(cnstr_config["x_max"])
        self.x_max_idx = cnstr_config["x_max_idx"]
        self.boundary = cnstr_config["boundary"]
        self.idx = 0 if  self.boundary== 'init' else -1 if self.boundary == 'final' else None
        self.eps = cnstr_config["eps"]
        self.dimension = len(self.x_min_idx) + len(self.x_max_idx)
        self.x_min = None
        self.x_max = None
        self.implement_type = 'inequality_bc'

    def nondim_constraint(self, nondim):
        if self.set == "state":
            self.x_min = nondim.M["state"]["d2nd"][np.ix_(self.x_min_idx, self.x_min_idx)] @ self.x_min
            self.x_max = nondim.M["state"]["d2nd"][np.ix_(self.x_max_idx, self.x_max_idx)] @ self.x_max
        elif self.set == "control":
            self.x_min = nondim.M["ctrl"]["d2nd"][np.ix_(self.x_min_idx, self.x_min_idx)] @ self.x_min
            self.x_max = nondim.M["ctrl"]["d2nd"][np.ix_(self.x_max_idx, self.x_max_idx)] @ self.x_max

class box:
    def __init__(self, cnstr_config, config=None):
        self.name = cnstr_config["name"]
        self.group = cnstr_config.get("group", None)
        self.set = cnstr_config["set"]
        self.x_min = cnstr_config["x_min"]
        self.x_max = cnstr_config["x_max"]
        self.x_min_idx = cnstr_config["x_min_idx"]
        self.x_max_idx = cnstr_config["x_max_idx"]
        self.dimension = len(self.x_min_idx) + len(self.x_max_idx)
        self.implement_type = 'box'

        if self.set == "state":
            n_elem = config['model']['dimensions']['n']
        elif self.set == "control":
            n_elem = config['model']['dimensions']['m']

        M_min = -np.eye(n_elem)[self.x_min_idx, :]
        M_max = np.eye(n_elem)[self.x_max_idx, :]
        self.M_select = np.vstack([M_min, M_max])

    def nondim_constraint(self, nondim):
        
        if self.set == "state":
            self.x_max = nondim.M["state"]["d2nd"][np.ix_(self.x_max_idx, self.x_max_idx)] @ self.x_max
            self.x_min = nondim.M["state"]["d2nd"][np.ix_(self.x_min_idx, self.x_min_idx)] @ self.x_min
        elif self.set == "control":
            self.x_min = nondim.M["ctrl"]["d2nd"][np.ix_(self.x_min_idx, self.x_min_idx)] @ self.x_min
            self.x_max = nondim.M["ctrl"]["d2nd"][np.ix_(self.x_max_idx, self.x_max_idx)] @ self.x_max

    def compute_constraint_values(self, t, z, nu, params):
        if self.set == "state":
            values = z @ self.M_select.T
        elif self.set == "control":
            values = nu @ self.M_select.T

        stacked_limits = np.hstack([self.x_min, self.x_max])
        limits = np.tile(stacked_limits, (values.shape[0], 1))

        output = {
            "values": values,
            "limits": limits,
            "M_select": self.M_select,
            "x_min": self.x_min,
            "x_max": self.x_max
        }

        return output

# ---------------------------------------------------------------
# rate constraints
# ---------------------------------------------------------------
class control_rate_limit:
    def __init__(self, cnstr_config, config=None):
        self.name = cnstr_config["name"]
        self.group = cnstr_config.get("group", None)
        self.udot_max = np.atleast_1d(cnstr_config["udot_max"])
        self.udot_max_idx = cnstr_config["udot_max_idx"]
        self.dimension = len(cnstr_config["udot_max_idx"])
        self.implement_type = 'control_rate_limit'

        n_elem = config['model']['dimensions']['m']
        M_min = -np.eye(n_elem)[self.udot_max_idx, :]
        M_max = np.eye(n_elem)[self.udot_max_idx, :]
        self.M_select = np.vstack([M_min, M_max])

    def nondim_constraint(self, nondim):
        self.udot_max = nondim.nt * nondim.M["ctrl"]["d2nd"][np.ix_(self.udot_max_idx, self.udot_max_idx)] @ self.udot_max

# ---------------------------------------------------------------
# Second-order cone constraints
# ---------------------------------------------------------------

class axis_angle_cone:
    def __init__(self, cnstr_config, config=None):

        self.name = cnstr_config["name"]
        self.group = cnstr_config.get("group", None)
        self.set = cnstr_config["set"]
        self.axis = cnstr_config["axis"] / np.linalg.norm(cnstr_config["axis"])
        self.theta_max = cnstr_config["theta_max"]
        self.cos_theta_max = np.cos(np.deg2rad(self.theta_max))
        self.x_idx = cnstr_config["x_idx"]
        self.dimension = 1
        self.implement_type = 'axis_angle_cone'

    def nondim_constraint(self, nondim):
        pass

    def compute_constraint_values(self, t, z, nu, params):
        if self.set == "state":
            theta = (z[:, self.x_idx] @ self.axis.T / np.linalg.norm(z[:, self.x_idx], axis=1)).reshape(-1, 1)
        elif self.set == "control":
            theta = (nu[:, self.x_idx] @ self.axis.T / np.linalg.norm(nu[:, self.x_idx], axis=1)).reshape(-1, 1)
        return {"values": np.rad2deg(theta), "limits": self.theta_max}

class max_norm_cone:
    def __init__(self, cnstr_config, config=None):
        self.name = cnstr_config["name"]
        self.group = cnstr_config.get("group", None)
        self.set = cnstr_config["set"]
        self.max_val = cnstr_config["max_val"]
        self.x_idx = cnstr_config["x_idx"]
        self.dimension = 1
        self.implement_type = 'max_norm_cone'

    def nondim_constraint(self, nondim):
        if self.set == "state":
            nondim_key = nondim.z_types[self.x_idx[0]]
            self.max_val = self.max_val * nondim.scales[nondim_key]
        elif self.set == "control":
            nondim_key = nondim.u_types[self.x_idx[0]]
            self.max_val = self.max_val * nondim.scales[nondim_key]

    def compute_constraint_values(self, t, z, nu, params):
        if self.set == "state":
            values = np.linalg.norm(z[:, self.x_idx], axis=1).reshape(-1, 1)
        elif self.set == "control":
            values = np.linalg.norm(nu[:, self.x_idx], axis=1).reshape(-1, 1)
        return {"values": values, "limits": self.max_val}

class quaternion_cone:
    def __init__(self, cnstr_config, config=None):
        self.name = cnstr_config["name"]
        self.group = None

        self.quat_start_idx = cnstr_config["quat_start_idx"]
        self.cos_theta_max = np.cos(np.deg2rad(cnstr_config["theta_max"]))
        self.axis_num = cnstr_config["axis_num"]
        self.rhs = np.sqrt((1.0 - self.cos_theta_max) * 0.5)
        
        self.dimension = 1
        self.implement_type = 'quaternion_cone'

    def nondim_constraint(self, nondim):
        pass

# ===============================================================
# NONCONVEX CONSTRAINTS
# ===============================================================

class nonconvex_inequality:
    def __init__(self, cnstr_config, config=None):

        # required config
        self.name            = cnstr_config["name"]
        self.group           = cnstr_config["group"]
        self.units           = cnstr_config["units"]

        self.fcn_string      = cnstr_config["fcn"]
        self.eps             = cnstr_config["eps"]
        self.dimension       = cnstr_config["dimension"]

        # optional configs
        self.ct              = cnstr_config.get("ct", 0)
        self.hard            = cnstr_config.get("hard", 0)
        self.backend         = cnstr_config.get("backend", "jax")
        self.max_val         = cnstr_config.get("max_val", None)
        self.min_val         = cnstr_config.get("min_val", None)
        self.upper_and_lower = (self.max_val is not None) and (self.min_val is not None)

        if self.upper_and_lower == True:
            # if both upper and lower bounds are provided, stack constraint int:
            # g = [-g(x).T + lower  g(x).T - upper].T <= 0
            self.dimension = 2 * self.dimension
            self.eps = jnp.concatenate((self.eps, self.eps))
            self.units = [*self.units, *self.units]

        self.config     = config
        self.implement_type = 'nonconvex_inequality'

        # symbolic function in dimensional units provided by user
        # (jax or sympy)
        self.fcn_dim = resolve_function(self.fcn_string)
        self.fcn_nd = None

        # this is the symbolic nondimmed version of fcn_fim, it will 
        # be provided once the nondim_constraint() function is called
        self.fcn = None

        # the compiled version (jitted for jax / numpy for sympy)
        # will be provided by problem.constraints.convexify_constraints() 
        self.fcn_compiled = None
        self.dfcn_dz_compiled = None
        self.dfcn_du_compiled = None

    def nondim_constraint(self, nondim):
        if self.backend == "jax":
            if self.max_val is not None:
                self.max_val = jnp.asarray(jnp.atleast_1d(self.max_val))
                M_out_d2nd = np.diag(1 / np.abs(self.max_val))
                self.max_val = M_out_d2nd @ self.max_val
            else:
                M_out_d2nd, _ = nondim.build_nondim_matrix(self.units)

            if self.min_val is not None:
                self.min_val = jnp.asarray(jnp.atleast_1d(self.min_val))
                self.min_val = M_out_d2nd @ self.min_val

            M_state_nd2d = nondim.M["state"]["nd2d"]
            M_ctrl_nd2d  = nondim.M["ctrl"]["nd2d"]

            self.fcn_nd = nondim.nondim_function(self.fcn_dim, M_state_nd2d, M_ctrl_nd2d, M_out_d2nd)

    def convexify_constraint(self):
    
        if self.backend == "jax":
            if self.upper_and_lower == True:
                fcn_lb   = lambda t, z, nu, params: -self.fcn_nd(t, z, nu, params) + self.min_val
                fcn_ub   = lambda t, z, nu, params: self.fcn_nd(t, z, nu, params) - self.max_val
                
                self.fcn = lambda t, z, nu, params: jnp.concatenate([fcn_lb(t, z, nu, params), fcn_ub(t, z, nu, params)])
            
            elif (self.max_val is not None):
                self.fcn = lambda t, z, nu, params: self.fcn_nd(t, z, nu, params) - self.max_val
            
            elif (self.min_val is not None):
                self.fcn = lambda t, z, nu, params: -self.fcn_nd(t, z, nu, params) + self.min_val
            else:
                self.fcn = self.fcn_nd

            self.fcn_compiled, self.dfcn_dz_compiled, self.dfcn_du_compiled = convexify.linearize_jax(self.fcn)
        
        elif self.backend == "sympy":
            pass

    def g_aff(self, t, z, nu, params):
        return (
            self.fcn_compiled(t, z, nu, params),
            self.dfcn_dz_compiled(t, z, nu, params),
            self.dfcn_du_compiled(t, z, nu, params)
        )

    def compute_constraint_values(self, t, z, nu, params):
        if self.backend == "jax":
            t_jax = jnp.asarray(t)
            z_jax = jnp.asarray(z)
            nu_jax = jnp.asarray(nu)
            f_batched = jax.vmap(self.fcn, in_axes=(0,0,0, None))
            values = np.asarray(f_batched(t_jax, z_jax, nu_jax, params))
        elif self.backend == "sympy":
            pass
        return {"values": values, 'limits': None}

class dynamics:
    def __init__(self, cnstr_config, config=None):
        self.name       = cnstr_config["name"]
        self.fcn_string = cnstr_config["fcn"]
        self.group      = "dynamics"
        self.implement_type = 'dynamics'

        self.config = config

        self.fcn_dim = resolve_function(self.fcn_string)

        self.backend = cnstr_config.get("backend", "jax")
        
        self.fcn = None
        self.fcn_compiled = None
        self.dfcn_dz_compiled = None
        self.dfcn_du_compiled = None

    def lin_dyn(self, t, z, nu, params):
        return (
            self.fcn_compiled(t, z, nu, params),
            self.dfcn_dz_compiled(t, z, nu, params),
            self.dfcn_du_compiled(t, z, nu, params)
        )

    def nondim_constraint(self, nondim):
        M_out_d2nd   = nondim.M["state"]["d2nd"] * nondim.nd_time
        M_state_nd2d = nondim.M["state"]["nd2d"]
        M_ctrl_nd2d  = nondim.M["ctrl"]["nd2d"]

        # fcn_dim is already bound with params/fcns by resolve_functions
        self.fcn = nondim.nondim_function(self.fcn_dim, M_state_nd2d, M_ctrl_nd2d, M_out_d2nd)

    def convexify_constraint(self):
    
        if self.backend == "jax":
            self.fcn_compiled, self.dfcn_dz_compiled, self.dfcn_du_compiled = convexify.linearize_jax(self.fcn)
        
        elif self.backend == "sympy":
            pass