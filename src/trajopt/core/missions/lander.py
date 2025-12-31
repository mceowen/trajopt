import numpy as np
import cvxpy as cp
import jax 
import jax.numpy as jnp


class Mission:
    def __init__(self, mission_config):
        self.mission_config = mission_config
        self.planet = self.mission_config['planet']
        self.vehicle = self.mission_config['vehicle']