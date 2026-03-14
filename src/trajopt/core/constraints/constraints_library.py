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
    def __init__(self, cnstr_config, index_map):
        # required properties
        self.name    = cnstr_config["name"]
        self.group   = cnstr_config.get("group", None)
        self.value_guess = cnstr_config.get("value_guess", None)
        self.ct = cnstr_config.get("ct", 0)

        # type-specific properties
        self.set = cnstr_config["set"]
        self.value = np.atleast_1d(cnstr_config["value"])

        self.idx = cnstr_config["idx"]
        self.boundary = cnstr_config["boundary"]
        
        if self.boundary == 'init':
            self.boundary_idx = 0 
        elif self.boundary == 'final':
            self.boundary_idx = -1 
        else:
            self.boundary_idx = None

        self.eps = cnstr_config.get("eps", np.zeros(len(self.idx)))
        self.dimension = len(self.idx)
        
        self.type = 'equality_bc'

    def fcn(self, x):
        return x[self.idx] - self.value

    def nondim_constraint(self, nondim):
        if self.set == "state":
            self.value = nondim.M.state.d2nd[np.ix_(self.idx, self.idx)] @ self.value
            self.eps = nondim.M.state.d2nd[np.ix_(self.idx, self.idx)] @ self.eps
        
        elif self.set == "control":
            self.value = nondim.M.ctrl.d2nd[np.ix_(self.idx, self.idx)] @ self.value
            self.eps   = nondim.M.ctrl.d2nd[np.ix_(self.idx, self.idx)] @ self.eps

class inequality_bc:
    def __init__(self, cnstr_config, index_map):
        self.name  = cnstr_config["name"]
        self.group = cnstr_config.get("group", None)
        self.set   = cnstr_config["set"]
        self.ct = cnstr_config.get("ct", 0)
        
        self.min_value_dim = np.atleast_1d(cnstr_config["min_value"])
        self.min_value_idx = cnstr_config["min_value_idx"]
        self.max_value_dim = np.atleast_1d(cnstr_config["max_value"])
        self.max_value_idx = cnstr_config["max_value_idx"]
        
        self.boundary = cnstr_config["boundary"]
        self.idx = 0 if  self.boundary== 'init' else -1 if self.boundary == 'final' else None
        self.eps = cnstr_config["eps"]
        self.dimension = len(self.min_value_idx) + len(self.max_value_idx)
        self.min_value = None
        self.max_value = None
        self.type = 'inequality_bc'

    def nondim_constraint(self, nondim):
        if self.set == "state":
            self.min_value = nondim.M["state"]["d2nd"][np.ix_(self.min_value_idx, self.min_value_idx)] @ self.min_value
            self.max_value = nondim.M["state"]["d2nd"][np.ix_(self.max_value_idx, self.max_value_idx)] @ self.max_value
        elif self.set == "control":
            self.min_value = nondim.M["ctrl"]["d2nd"][np.ix_(self.min_value_idx, self.min_value_idx)] @ self.min_value
            self.max_value = nondim.M["ctrl"]["d2nd"][np.ix_(self.max_value_idx, self.max_value_idx)] @ self.max_value

class box:
    def __init__(self, cnstr_config, index_map):
        
        self.name  = cnstr_config["name"]
        self.group = cnstr_config.get("group", None)
        self.set   = cnstr_config["set"]
        self.ct = cnstr_config.get("ct", 0)
        
        self.min_value     = cnstr_config["min_value"]
        self.max_value     = cnstr_config["max_value"]
        self.min_value_idx = cnstr_config["min_value_idx"]
        self.max_value_idx = cnstr_config["max_value_idx"]
        self.dimension = len(self.min_value_idx) + len(self.max_value_idx)
        self.type = 'box'

        if self.set == "state":
            n_elem = index_map.n.state
        elif self.set == "control":
            n_elem = index_map.n.control

        M_min = -np.eye(n_elem)[self.min_value_idx, :]
        M_max = np.eye(n_elem)[self.max_value_idx, :]
        self.M_select = np.vstack([M_min, M_max])

    def nondim_constraint(self, nondim):
        
        if self.set == "state":
            self.max_value = nondim.M["state"]["d2nd"][np.ix_(self.max_value_idx, self.max_value_idx)] @ self.max_value
            self.min_value = nondim.M["state"]["d2nd"][np.ix_(self.min_value_idx, self.min_value_idx)] @ self.min_value
        elif self.set == "control":
            self.min_value = nondim.M["ctrl"]["d2nd"][np.ix_(self.min_value_idx, self.min_value_idx)] @ self.min_value
            self.max_value = nondim.M["ctrl"]["d2nd"][np.ix_(self.max_value_idx, self.max_value_idx)] @ self.max_value

    def compute_constraint_values(self, t, z, nu, params):
        if self.set == "state":
            values = z @ self.M_select.T
        elif self.set == "control":
            values = nu @ self.M_select.T

        stacked_limits = np.hstack([self.min_value, self.max_value])
        limits = np.tile(stacked_limits, (values.shape[0], 1))

        output = {
            "values": values,
            "limits": limits,
            "M_select": self.M_select,
            "min_value": self.min_value,
            "max_value": self.max_value
        }

        return output

# ---------------------------------------------------------------
# rate constraints
# ---------------------------------------------------------------
class control_rate_limit:
    def __init__(self, cnstr_config, index_map):
        self.name = cnstr_config["name"]
        self.group = cnstr_config.get("group", None)
        self.value = np.atleast_1d(cnstr_config["value"])
        self.idx = cnstr_config["idx"]
        self.dimension = len(cnstr_config["idx"])
        self.type = 'control_rate_limit'
        self.ct = cnstr_config.get("ct", 0)

        n_elem = index_map.n.control
        M_min = -np.eye(n_elem)[self.idx, :]
        M_max = np.eye(n_elem)[self.idx, :]
        self.M_select = np.vstack([M_min, M_max])

    def nondim_constraint(self, nondim):
        self.value = nondim.time_scale * nondim.M.ctrl.d2nd[np.ix_(self.idx, self.idx)] @ self.value

# ---------------------------------------------------------------
# Second-order cone constraints
# ---------------------------------------------------------------

class axis_angle_cone:
    def __init__(self, cnstr_config, index_map):

        self.name  = cnstr_config["name"]
        self.group = cnstr_config.get("group", None)
        self.set = cnstr_config["set"]
        self.axis = cnstr_config["axis"] / np.linalg.norm(cnstr_config["axis"])
        self.theta_max = cnstr_config["theta_max"]
        self.cos_theta_max = np.cos(np.deg2rad(self.theta_max))
        self.idx = cnstr_config["idx"]
        self.dimension = 1
        self.ct = cnstr_config.get("ct", 0)
        self.type = 'axis_angle_cone'

    def nondim_constraint(self, nondim):
        pass

    def compute_constraint_values(self, t, z, nu, params):
        if self.set == "state":
            theta = (z[:, self.idx] @ self.axis.T / np.linalg.norm(z[:, self.idx], axis=1)).reshape(-1, 1)
        elif self.set == "control":
            theta = (nu[:, self.idx] @ self.axis.T / np.linalg.norm(nu[:, self.idx], axis=1)).reshape(-1, 1)

        limits = np.tile(self.theta_max, (t.shape[0], 1))
        
        return {"values": np.rad2deg(np.arccos(theta)), "limits": limits}

class max_norm_cone:
    def __init__(self, cnstr_config, index_map):
        self.name = cnstr_config["name"]
        self.group = cnstr_config.get("group", None)
        self.set = cnstr_config["set"]
        self.max_value = cnstr_config["max_value"]
        self.idx = cnstr_config["idx"]
        self.dimension = 1
        self.ct = cnstr_config.get("ct", 0)
        self.type = 'max_norm_cone'

    def nondim_constraint(self, nondim):
        if self.set == "state":
            self.max_value = self.max_value * nondim.M.state.d2nd[self.idx[0], self.idx[0]]
        elif self.set == "control":
            self.max_value = self.max_value * nondim.M.state.d2nd[self.idx[0], self.idx[0]]

    def compute_constraint_values(self, t, z, nu, params):
        if self.set == "state":
            values = np.linalg.norm(z[:, self.idx], axis=1).reshape(-1, 1)
        elif self.set == "control":
            values = np.linalg.norm(nu[:, self.idx], axis=1).reshape(-1, 1)

        limits = np.tile(self.max_value, (t.shape[0], 1))
        
        
        return {"values": values, "limits": limits}

class quaternion_cone:
    def __init__(self, cnstr_config, index_map):
        self.name = cnstr_config["name"]
        self.group = None
        self.ct = cnstr_config.get("ct", 0)

        self.quat_start_idx = cnstr_config["quat_start_idx"]
        self.cos_theta_max = np.cos(np.deg2rad(cnstr_config["theta_max"]))
        self.axis_num = cnstr_config["axis_num"]
        self.rhs = np.sqrt((1.0 - self.cos_theta_max) * 0.5)
        
        self.dimension = 1
        self.type = 'quaternion_cone'

    def nondim_constraint(self, nondim):
        pass

# ===============================================================
# NONCONVEX CONSTRAINTS
# ===============================================================

class nonconvex_inequality:
    def __init__(self, cnstr_config, index_map):

        # required config
        self.name           = cnstr_config["name"]
        self.group          = cnstr_config.get("group", None)
        self.scale          = cnstr_config.get("scale", None)
        self.nodes           = cnstr_config.get("nodes", np.arange(0, index_map.N.N))

        self.fcn_string      = cnstr_config["fcn"]
        self.eps             = cnstr_config["eps"]
        self.dimension       = cnstr_config["dimension"]
        self.ct = cnstr_config.get("ct", 0)

        # optional configs
        self.ct              = cnstr_config.get("ct", 0)
        self.hard            = cnstr_config.get("hard", 0)
        self.backend         = cnstr_config.get("backend", "jax")
        self.max_value       = cnstr_config.get("max_value", None)
        self.min_value       = cnstr_config.get("min_value", None)
        self.upper_and_lower = (self.max_value is not None) and (self.min_value is not None)

        if self.upper_and_lower == True:
            # if both upper and lower bounds are provided, stack constraint int:
            # g = [-g(x).T + lower  g(x).T - upper].T <= 0
            self.dimension = 2 * self.dimension

        self.index_map     = index_map
        self.type = 'nonconvex_inequality'

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
            
            # get max and min values regardless
            if self.max_value is not None:
                self.max_value = jnp.asarray(jnp.atleast_1d(self.max_value))
            
            if self.min_value is not None:
                self.min_value = jnp.asarray(jnp.atleast_1d(self.min_value))

            # get provided scales if provided
            if self.scale is not None:
                # TODO (Carlos): revisit, temporary scaling if max value is equal to zero
                self.M_out_d2nd = jnp.diag(1 / jnp.abs(jnp.asarray(self.scale)))
            
            # if scales aren't provided, try to scale using max/ min values if provided
            elif self.max_value is not None:
                self.M_out_d2nd = jnp.diag(1 / jnp.abs(self.max_value))
            
            elif self.min_value is not None:
                self.M_out_d2nd = jnp.diag(1 / jnp.abs(self.min_value))
            
            else:
                raise ValueError("At least one of 'scales', 'max_value', or 'min_value' must be provided for nondimensionalization.")

            self.M_out_nd2d = jnp.diag(1 / jnp.diag(self.M_out_d2nd))

            if self.max_value is not None:
                self.max_value = self.M_out_d2nd @ self.max_value
            
            if self.min_value is not None:
                self.min_value = self.M_out_d2nd @ self.min_value

            M_state_nd2d_jax = jnp.asarray(nondim.M.state.nd2d)
            M_ctrl_nd2d_jax  = jnp.asarray(nondim.M.ctrl.nd2d)

            # debug print
            # print(f"name: {self.name}, M_out_d2nd: {self.M_out_d2nd}, max_value: {self.max_value}, min_value: {self.min_value}")

            self.fcn_nd = nondim.nondim_function(self.fcn_dim, M_state_nd2d_jax, M_ctrl_nd2d_jax, self.M_out_d2nd)

            self.eps = self.M_out_d2nd @ self.eps

            if self.upper_and_lower == True:
                self.eps = jnp.concatenate((self.eps, self.eps))

    def convexify_constraint(self):
        if self.backend == "jax":
            if self.upper_and_lower == True:
                fcn_lb   = lambda t, z, nu, params: -self.fcn_nd(t, z, nu, params) + self.min_value
                fcn_ub   = lambda t, z, nu, params: self.fcn_nd(t, z, nu, params)  - self.max_value
                
                self.fcn = lambda t, z, nu, params: jnp.concatenate([fcn_lb(t, z, nu, params), fcn_ub(t, z, nu, params)])
                
            elif (self.max_value is not None) and (self.min_value is None):
                self.fcn = lambda t, z, nu, params: self.fcn_nd(t, z, nu, params) - self.max_value

            elif (self.min_value is not None) and (self.max_value is None):
                self.fcn = lambda t, z, nu, params: -self.fcn_nd(t, z, nu, params) + self.min_value
                
            else:
                self.fcn = self.fcn_nd

            vals_function = lambda t, z, nu, params: self.M_out_nd2d @ (self.fcn_nd(t, z, nu, params))

            self.fcn_compiled, self.dfcn_dz_compiled, self.dfcn_du_compiled = convexify.linearize_jax(self.fcn)

            self.f_batched = jax.jit(jax.vmap(vals_function, in_axes=(0,0,0, None)))
        
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
            values = np.asarray(self.f_batched(t_jax, z_jax, nu_jax, params))

            n_t = t_jax.shape[0]

            if self.min_value is not None:
                self.limits = np.tile(np.asarray(self.M_out_nd2d @ self.min_value), (n_t, self.dimension))
            
            if self.max_value is not None and self.min_value is not None:
                max_val_limits = np.tile(np.asarray(self.M_out_nd2d @ self.max_value), (n_t, self.dimension))
                self.limits    = np.hstack([self.limits, max_val_limits])
            
            elif self.max_value is not None:
                max_val_limits = np.tile(np.asarray(self.M_out_nd2d @ self.max_value), (n_t, self.dimension))
                self.limits = max_val_limits

            elif self.min_value is not None:
                min_val_limits = np.tile(np.asarray(self.M_out_nd2d @ self.min_value), (n_t, self.dimension))
                self.limits = min_val_limits

            else: 
                self.limits = None

        elif self.backend == "sympy":
            pass
        return {"values": values, 'limits': self.limits}
    
class nonconvex_equality:
    def __init__(self, cnstr_config, index_map):

        # required config
        self.name           = cnstr_config["name"]
        self.group          = cnstr_config.get("group", None)
        self.scale          = cnstr_config.get("scale", None)
        self.nodes           = cnstr_config.get("nodes", np.arange(0, index_map.N.N))

        self.fcn_string      = cnstr_config["fcn"]
        self.eps             = cnstr_config["eps"]
        self.dimension       = cnstr_config["dimension"]
        self.ct = cnstr_config.get("ct", 0)

        # optional configs
        self.ct              = cnstr_config.get("ct", 0)
        self.hard            = cnstr_config.get("hard", 0)
        self.backend         = cnstr_config.get("backend", "jax")
        self.max_value       = cnstr_config.get("value", None)

        self.index_map     = index_map
        self.type = 'nonconvex_equality'

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
            
            # get max and min values regardless
            if self.value is not None:
                self.value = jnp.asarray(jnp.atleast_1d(self.value))

            # get provided scales if provided
            if self.scale is not None:
                # TODO (Carlos): revisit, temporary scaling if max value is equal to zero
                self.M_out_d2nd = jnp.diag(1 / jnp.abs(jnp.asarray(self.scale)))
            
            # if scales aren't provided, try to scale using max/ min values if provided
            elif self.value is not None:
                self.M_out_d2nd = jnp.diag(1 / jnp.abs(self.value))
            
            else:
                raise ValueError("At least one of 'scales', 'max_value', or 'min_value' must be provided for nondimensionalization.")

            self.M_out_nd2d = jnp.diag(1 / jnp.diag(self.M_out_d2nd))

            if self.value is not None:
                self.value = self.M_out_d2nd @ self.value

            M_state_nd2d_jax = jnp.asarray(nondim.M.state.nd2d)
            M_ctrl_nd2d_jax  = jnp.asarray(nondim.M.ctrl.nd2d)

            # debug print
            # print(f"name: {self.name}, M_out_d2nd: {self.M_out_d2nd}, max_value: {self.max_value}, min_value: {self.min_value}")

            self.fcn_nd = nondim.nondim_function(self.fcn_dim, M_state_nd2d_jax, M_ctrl_nd2d_jax, self.M_out_d2nd)

            self.eps = self.M_out_d2nd @ self.eps

    def convexify_constraint(self):
        if self.backend == "jax":
                
            if (self.max_value is not None):
                self.fcn = lambda t, z, nu, params: self.fcn_nd(t, z, nu, params) - self.max_value
            else:
                self.fcn = self.fcn_nd

            vals_function = lambda t, z, nu, params: self.M_out_nd2d @ (self.fcn_nd(t, z, nu, params))

            self.fcn_compiled, self.dfcn_dz_compiled, self.dfcn_du_compiled = convexify.linearize_jax(self.fcn)

            self.f_batched = jax.jit(jax.vmap(vals_function, in_axes=(0,0,0, None)))
        
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
            values = np.asarray(self.f_batched(t_jax, z_jax, nu_jax, params))

            n_t = t_jax.shape[0]

            if self.value is not None:
                self.limits = np.tile(np.asarray(self.M_out_nd2d @ self.value), (n_t, self.dimension))

            else: 
                self.limits = None

        elif self.backend == "sympy":
            pass
        return {"values": values, 'limits': self.limits}

class dynamics:
    def __init__(self, cnstr_config, index_map):
        self.name       = cnstr_config["name"]
        self.fcn_string = cnstr_config["fcn"]
        self.group      = "dynamics"
        self.type       = 'dynamics'

        self.index_map = index_map

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

        if self.backend == "jax":
            M_out_d2nd_jax   = jnp.asarray(nondim.M.state.d2nd * nondim.time_scale)
            M_state_nd2d_jax = jnp.asarray(nondim.M.state.nd2d)
            M_ctrl_nd2d_jax  = jnp.asarray(nondim.M.ctrl.nd2d)
        
        elif self.backend == "sympy":
            pass

        # fcn_dim is already bound with params/fcns by resolve_functions
        self.fcn = nondim.nondim_function(self.fcn_dim, M_state_nd2d_jax, M_ctrl_nd2d_jax, M_out_d2nd_jax)

    def convexify_constraint(self):
        if self.backend == "jax":
            self.fcn_compiled, self.dfcn_dz_compiled, self.dfcn_du_compiled = convexify.linearize_jax(self.fcn)
        
        elif self.backend == "sympy":
            pass