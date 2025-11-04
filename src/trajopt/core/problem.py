import trajopt; import importlib; importlib.reload(trajopt)
import trajopt.utils.config_loader          as cfg
import trajopt.utils.nondim                 as nondim

from trajopt.core.mission import Mission
from trajopt.core.model import Model
from trajopt.core.method import Method
import numpy as np

class Problem:
    def __init__(self, example_name):

        config = cfg.load_configs(example_name)

        self.mission = Mission(self, config)
        self.model   = Model(self, config)
        self.method  = Method(self, config)

        self._initialize_problem()
        self.method.get_initial_guess()

        self.case_flag = 1

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

        # Save variable names
        self.save_var_names    = ["ts_opt", "zs_opt", "us_opt", "params", "O"]