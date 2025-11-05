import trajopt; import importlib; importlib.reload(trajopt)
import trajopt.utils.nondim                 as nondim

from trajopt.core.mission   import Mission
from trajopt.core.model     import Model
from trajopt.core.method    import Method
import numpy as np

class Problem:
    def __init__(self, config, subprob=None):

        # construct mission / model / method objects from configs
        self.mission = Mission(self, config)
        self.model   = Model(self, config)
        self.method  = Method(self, config)

        # complete the mission / model / method object setup with interdependent definitions
        self._initialize_problem()

        # use precompiled cvxpy subproblem if provided
        if subprob is not None:
            self.method.subprob = subprob

    def _initialize_problem(self):

        mission = self.mission
        model   = self.model
        method  = self.method

        # initialize nondim (mission.n_nfz, and mission.nfz_idx need to be set first)
        mission.initialize_nfz()
        nondim.set_nondim_params(self)

        # finish mission, model, method (this order currently matters)
        mission.update_mission_params()
        model.update_model_params()
        method.update_method_params()
        method.get_initial_guess()