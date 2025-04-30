
import numpy as np

import Mission

# The model is the high level representation of the system.
class Model:

    control: np.ndarray
    limit: np.ndarray

    mission: Mission

    def __init__(self, config: object):
        self.control = config['control']
        self.limit = config['limit']
        self.time_horizon = 0 # tf


    def cost_function(self):
        self.mission.model_aerodynamics(1, 2, 3)

    def dynamics(self):
        pass

    def state_constraints(self):
        pass

    def control_constraints(self):
        pass

    def control_rate(self):
        pass

    def path_constraints(self):
        pass

    def time_constraints(self):
        pass

    def boundary_conditions(self):
        pass