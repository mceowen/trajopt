import numpy as np
import cvxpy as cp
import jax
import jax.numpy as jnp
import trajopt.core.modules.method.convexify as convexify


class terminal_state:
    def __init__(self, idx):
        pass

class running_state:
    def __init__(self, fcn):
        pass

    def fcn(self, t, z, nu):
        pass

class time:
    def __init__(self):
        pass