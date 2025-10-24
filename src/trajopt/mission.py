import numpy as np
import importlib

import trajopt.utils.set_defaults as defaults
import trajopt.utils.tools as tools
import trajopt.algorithm.initial_guess as guess
import trajopt.algorithm.convergence as convergence
import trajopt.algorithm.convexification as convexify
import trajopt.utils.nondim as nondim


# TODO: configs and params will likely change
class Mission:
    def __init__(self, problem):

        self.problem = problem
        self.params = problem["params"]

        # point to selected mission module
        mission_name = self.params["mission"]["mission_name"]
        mission_module = importlib.import_module(f"trajopt.missions.{mission_name}")

        # set cost function
        self._cost = mission_module.cost
        self._analytical_cost = mission_module.analytical_cost

        # set linearized cost function
        if self.params["method"]["bools"]["auto_jac"]:
            self._lin_cost = convexify.generate_jacobians(self._cost)
        else:
            self._lin_cost = self._analytical_cost

        # set custom inputs
        self._custom_inputs = mission_module.custom_inputs
        self._custom_subprob_variables = mission_module.custom_subprob_variables
        self._custom_subprob_constraints = mission_module.custom_subprob_constraints
        self._custom_subprob_cost = mission_module.custom_subprob_cost

    # ===============================================================
    # member functions point to selected fcns from selected module
    # ===============================================================

    def cost(self, ts, zs, us):
        return self._cost(ts, zs, us, self)

    def lin_cost(self, ts, zs, us):
        return self._lin_cost(ts, zs, us, self)

    def custom_inputs(self, problem, local_vars):
        return self._custom_inputs(problem, local_vars)

    def custom_subprob_variables(self, problem, local_vars):
        return self._custom_subprob_variables(problem, local_vars)

    def custom_subprob_constraints(self, problem, local_vars):
        return self._custom_subprob_constraints(problem, local_vars)

    def custom_subprob_cost(self, problem, local_vars):
        return self._custom_subprob_cost(problem, local_vars)
