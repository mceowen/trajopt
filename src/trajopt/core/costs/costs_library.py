import numpy as np
import cvxpy as cp
import jax
import jax.numpy as jnp
import trajopt.core.scp.convexify as convexify

class min_time:
    def __init__(self, name, params=None, fcns=None):
        self.name = name
        self.group = None

class terminal_state:
    def __init__(self, name, x_idx, params=None, fcns=None):
        self.name = name
        self.group = None
        self.x_idx = x_idx

class nonconvex_terminal:
    def __init__(self, name, x_idx, fcn, params=None, fcns=None):
        self.name = name
        self.group = None
        self.x_idx = x_idx
        self._fcn = fcn

class nonconvex_running:
    def __init__(self, name, x_idx, fcn, params=None, fcns=None):
        self.name = name
        self.group = None
        self.x_idx = x_idx
        self._fcn = fcn