import numpy as np
import importlib
from typing import Dict, Any, Optional, List

import numpy as np
import importlib

import trajopt.algorithm.convexification as convexify

class Model:

    def __init__(self, problem, config):

        self.problem = problem

        # ===============================================================
        # load config parameters
        # ===============================================================

        model_config = config["model"]
        self.model_name       = model_config["model_name"]
        self.n                = model_config["n"]
        self.m                = model_config["m"]
        self.z_types          = model_config["z_types"]
        self.u_types          = model_config["u_types"]
        self.anchor_types     = model_config["anchor_types"]
        self.anchor_scales    = model_config["anchor_scales"]
        self.base_unit_labels = model_config["base_unit_labels"]
        self.bools            = model_config["bools"]

        # =================================================================
        # point to module containing corresponding methods based on configs
        # =================================================================

        # point to selected model module
        model_module = importlib.import_module(f"trajopt.model_modules.{self.model_name}")

        # set dynamics
        self._dynamics = model_module.system_dynamics

        # set ltv dynamics
        if config["method"]["bools"]["auto_jac"]:
            self._lin_dyn = convexify.generate_jacobians(self.dynamics)
        else:
            self._lin_dyn = model_module.analytical_linsys

        # set nonlinear inequality constraints
        self._nonlinear_inequality_constraints = (model_module.nonlinear_inequality_constraints)

        # set linearized constraints
        if config["method"]["bools"]["auto_jac_cnst"]:
            self._lin_constr = convexify.generate_jacobians(self.nonlinear_inequality_constraints)
        else:
            self._lin_constr = model_module.analytical_inequality_constraints

        self._get_initial_guess_control = model_module.get_initial_guess_control

    # ===============================================================
    # member functions point to selected fcns from selected module
    # ===============================================================

    def dynamics(self, ts, zs, us, t_vec=None):
        return self._dynamics(ts, zs, us, self.problem, t_vec)

    def lin_dyn(self, ts, zs, us):
        return self._lin_dyn(ts, zs, us, self.problem)

    def nonlinear_inequality_constraints(self, ts, zs, us):
        return self._nonlinear_inequality_constraints(ts, zs, us, self.problem)

    def lin_constr(self, ts, zs, us):
        return self._lin_constr(ts, zs, us, self.problem)
    
    def get_initial_guess_control(self):
        return self._get_initial_guess_control(self.problem)
    
    def update_model_params(self):
        problem = self.problem
        method = problem.method
        mission = problem.mission
        
        # ctcs state vector update 
        # (this couples mission, model and method, maybe it can be done in a cleaner way)

        if method.bools.get("ctcs", False):
            self.nz = self.n + mission.n_ineq
        else:
            self.nz = self.n
        mission.n_dyn = self.nz
