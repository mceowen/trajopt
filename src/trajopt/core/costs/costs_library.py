import numpy as np
import cvxpy as cp
import jax
import jax.numpy as jnp
import trajopt.library.methods.convexify as convexify
from trajopt.utils.config_loader import resolve_function

class min_time:
    def __init__(self, name, params=None, fcns=None):
        self.name = name
        self.group = None
    def nondim_cost(self, nondim):
        pass

class terminal_state:
    def __init__(self, name, x_idx, params=None, fcns=None):
        self.name = name
        self.group = None
        self.x_idx = x_idx
    def nondim_cost(self, nondim):
        pass

class min_norm_terminal:
    def __init__(self, name, x_idx, params=None, fcns=None):
        self.name = name
        self.group = None
        self.x_idx = x_idx
    
    def nondim_cost(self, nondim):
        pass

class nonconvex:
    def __init__(self, name, group, fcn, max_val, ct=None):
        self.name = name
        self.group = group

        self.implement_type = 'nonconvex_terminal'
        self.fcn_dim = resolve_function(fcn)
        
        # this will be a function f(t, z, nu)
        self.fcn = None
        
        self.fcn_jit = None
        self.dfcn_dz_jit = None
        self.dfcn_du_jit = None
        
        self.max_val = max_val
    
    def g_aff(self, t, z, nu):
        return self.fcn_jit(z, nu), self.dfcn_dz_jit(z, nu), self.dfcn_du_jit(z, nu)

    def nondim_cost(self, nondim):

        M_out_d2nd = np.diag(1 / self.max_val)

        # fcn_dim is already bound with params/fcns by resolve_functions
        nd_fcn = nondim.nondim_function(self.fcn_dim, nondim.M["state"]["nd2d"], nondim.M["ctrl"]["nd2d"], M_out_d2nd)

        self.fcn = lambda t, z, nu: nd_fcn(t, z, nu)
    
    # def make_epigraph_constraint()