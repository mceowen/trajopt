import numpy as np
import importlib

import trajopt.utils.set_defaults as defaults
import trajopt.utils.tools as tools
import trajopt.algorithm.initial_guess as guess
import trajopt.algorithm.convergence as convergence
import trajopt.algorithm.convexification as convexify
import trajopt.utils.nondim as nondim

class Mission:
    def __init__(self, problem, config):

        self.problem = problem

        # ===============================================================
        # load config parameters
        # ===============================================================
        mission_config         = config["mission"]
        self.mission_name      = mission_config["mission_name"]
        self.ge                = mission_config["ge"]
        self.mass              = mission_config["mass"]
        self.bools             = mission_config["bools"]
        self.nfz_option_list   = mission_config["nfz_option_list"]
        self.zi                = mission_config["zi"]
        self.zi_idx            = mission_config["zi_idx"]
        self.zf                = mission_config["zf"]
        self.zf_idx            = mission_config["zf_idx"]
        # TODO: update these to not be hardcoded in problem.intialize_problem() and loaded from yaml 
        # self.ui              = mission_config["ui"]
        # self.uf              = mission_config["uf"]
        self.z_min             = mission_config["z_min"]
        self.z_min_idx         = mission_config["z_min_idx"]
        self.z_max             = mission_config["z_max"]
        self.z_max_idx         = mission_config["z_max_idx"]
        self.udot_max          = mission_config["udot_max"]
        self.udot_max_idx      = mission_config["udot_max_idx"]
        self.nfz_idx           = mission_config["nfz_idx"]
        self.zi_min            = mission_config["zi_min"]
        self.zi_max            = mission_config["zi_max"]
        self.zi_min_idx        = mission_config["zi_min_idx"]
        self.zi_max_idx        = mission_config["zi_max_idx"]
        self.zf_min            = mission_config["zf_min"]
        self.zf_max            = mission_config["zf_max"]
        self.zf_min_idx        = mission_config["zf_min_idx"]
        self.zf_max_idx        = mission_config["zf_max_idx"]
        self.u_min             = mission_config["u_min"]
        self.u_max             = mission_config["u_max"]
        self.u_min_idx         = mission_config["u_min_idx"]
        self.u_max_idx         = mission_config["u_max_idx"]
        self.path_lim          = mission_config["path_lim"]
        self.path_idx          = mission_config["path_idx"]
        self.aux_idx           = mission_config["aux_idx"]
        self.custom_input_dict = mission_config["custom_input_dict"]

        # ===============================================================
        # point to module and corresponding methods based on configs
        # ===============================================================

        mission_module = importlib.import_module(f"trajopt.mission_modules.{self.mission_name}")

        # set cost function
        self._cost = mission_module.cost
        self._analytical_cost = mission_module.analytical_cost

        # set linearized cost function
        if config["method"]["bools"]["auto_jac"]:
            self._lin_cost = convexify.generate_jacobians(self._cost)
        else:
            self._lin_cost = self._analytical_cost

        # set custom inputs
        self._custom_inputs = mission_module.custom_inputs
        self._custom_variables = mission_module.custom_variables
        self._custom_constraints = mission_module.custom_constraints
        self._custom_cost = mission_module.custom_cost

    # ===============================================================
    # member functions point to selected fcns from selected module
    # ===============================================================

    def cost(self, ts, zs, us):
        return self._cost(ts, zs, us, self.problem)

    def lin_cost(self, ts, zs, us):
        return self._lin_cost(ts, zs, us, self.problem)

    def custom_inputs(self, problem, local_vars):
        return self._custom_inputs(problem, local_vars)

    def custom_variables(self, problem, local_vars):
        return self._custom_variables(problem, local_vars)

    def custom_constraints(self, problem, local_vars):
        return self._custom_constraints(problem, local_vars)

    def custom_cost(self, problem, local_vars):
        return self._custom_cost(problem, local_vars)
