import numpy as np
import cvxpy as cp
import jax
import jax.numpy as jnp
import trajopt.library.methods.convexify as convexify
from trajopt.utils.config_loader import resolve_function

class min_time:
    def __init__(self, cost_config, config=None):
        self.type = __name__
        self.name = cost_config["name"]
        self.group = cost_config.get("group", None)
    
    def nondim_cost(self, nondim):
        pass

class terminal_state:
    def __init__(self, cost_config, config=None):
        self.type = __name__
        self.name = cost_config["name"]
        self.group = cost_config.get("group", None)
        self.idx = cost_config["idx"]
    def nondim_cost(self, nondim):
        pass

class min_norm_terminal:
    def __init__(self, cost_config, config=None):
        self.type = __name__
        self.name = cost_config["name"]
        self.group = cost_config.get("group", None)
        self.idx = cost_config["idx"]
    
    def nondim_cost(self, nondim):
        pass

class nonconvex:
    def __init__(self, cnstr_config, config=None):
        # required config
        self.type = __name__
        self.name            = cnstr_config["name"]
        self.group           = cnstr_config["group"]
        self.units           = cnstr_config["units"]

        self.fcn_string      = cnstr_config["fcn"]
        self.eps             = cnstr_config["eps"]

        # optional configs
        self.ct              = cnstr_config.get("ct", 0)
        self.backend         = cnstr_config.get("backend", "jax")

        self.config     = config

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

    def nondim_cost(self, nondim):
        if self.backend == "jax":
            M_out_d2nd, _ = nondim.build_nondim_matrix(self.units)

            M_state_nd2d = nondim.M["state"]["nd2d"]
            M_ctrl_nd2d  = nondim.M["ctrl"]["nd2d"]

            self.fcn_nd = nondim.nondim_function(self.fcn_dim, M_state_nd2d, M_ctrl_nd2d, M_out_d2nd)

    def convexify_cost(self):
        if self.backend == "jax":
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