import numpy as np
import cvxpy as cp
import jax
import jax.numpy as jnp
import trajopt.core.methods.convexify as convexify


class terminal_state:
    def __init__(self, name, idx):
        self.name = name
        self.idx = idx

class running_state:
    def __init__(self, name, fcn):
        self.name = name
        self._fcn = fcn

    def fcn(self, t, z, nu):
        pass

class min_time:
    def __init__(self, name):
        self.name = name
        self.group = None