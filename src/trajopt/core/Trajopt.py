import numpy as np

import trajopt; import importlib; importlib.reload(trajopt)
import trajopt.utils.nondim as nondim

from trajopt.core.Problem   import Problem
from trajopt.core.Method    import Method
from trajopt.core.modules.method.indices import Indices  


class Trajopt:
    def __init__(self, config, subprob=None):

        # example
        self.name       = config.get("example_name", "unnamed_trajopt_obj_obj")

        # construct mission / model / method objects from configs
        self.problem      = Problem(self, config)
        self.method     = Method(self, config)
        
        # initialize nondim
        nondim.set_nondim_params(self)

        # finish mission, model, method (this order currently matters)
        self.model.update_model_params()
        self.mission.update_mission_params()
        self.indices = Indices(self)
        self.method.update_method_params()
 
        # get initial guess
        self.method.get_initial_guess()

        # use precompiled cvxpy subtrajopt_obj if provided
        if subprob is not None:
            self.method.subprob = subprob