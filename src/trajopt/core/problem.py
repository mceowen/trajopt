import numpy as np

import trajopt; import importlib; importlib.reload(trajopt)
import trajopt.utils.nondim as nondim

from trajopt.core.mission   import Mission
from trajopt.core.model     import Model
from trajopt.core.method    import Method
from trajopt.core.modules.method.indices import Indices  


class Problem:
    def __init__(self, config, subprob=None):

        # example
        self.name = config.get("example_name", "unnamed_problem")

        # construct mission / model / method objects from configs
        self.mission = Mission(self, config)
        self.model   = Model(self, config)
        self.method  = Method(self, config)
        
        # initialize nondim
        nondim.set_nondim_params(self)

        # finish mission, model, method (this order currently matters)
        self.mission.update_mission_params()
        self.model.update_model_params()
        self.indices = Indices(self)
        self.method.update_method_params()
 
        # get initial guess
        self.method.get_initial_guess()

        

        # use precompiled cvxpy subproblem if provided
        if subprob is not None:
            self.method.subprob = subprob