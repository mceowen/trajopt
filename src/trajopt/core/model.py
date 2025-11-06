import numpy as np
import importlib
from typing import Dict, Any, Optional, List

import numpy as np
import importlib

import trajopt.core.modules.methods.convexify as convexify

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
        self.model_module = importlib.import_module(f"trajopt.core.modules.models.{self.model_name}")

    # ===============================================================
    # member functions point to selected fcns from selected module
    # ===============================================================

    def dynamics(self, ts, zs, us, t_vec=None):
        return self._dynamics(ts, zs, us, self.problem, t_vec)

    def lin_dyn(self, ts, zs, us):

        method = self.problem.method

        if method.bools["jax_dyn"] == 1:
            return self._lin_dyn(ts, zs, us)

        else:
            return self._lin_dyn(ts, zs, us, self.problem)

    def nonlinear_inequality_constraints(self, ts, zs, us):
        return self._nonlinear_inequality_constraints(ts, zs, us, self.problem)

    def lin_constr(self, ts, zs, us):
        return self._lin_constr(ts, zs, us, self.problem)
    
    def update_model_params(self):
        problem = self.problem
        method = problem.method
        mission = problem.mission
        import importlib

        # ------------------------------------------------------------
        # Try to import a custom override module if it exists
        # ------------------------------------------------------------
        custom_model_module = None
        try:
            example_name        = getattr(self.problem, "name", None)
            custom_model_module = importlib.import_module(f"trajopt.examples.{example_name}.custom")
        except ModuleNotFoundError:
            pass  # no custom module — use defaults

        # ------------------------------------------------------------
        # system dynamics
        # ------------------------------------------------------------
        if method.bools["jax_dyn"]:
            if custom_model_module and hasattr(custom_model_module, "system_dynamics_jax"):
                self._dynamics = custom_model_module.system_dynamics_jax
            else:
                self._dynamics = self.model_module.system_dynamics_jax
        else:
            if custom_model_module and hasattr(custom_model_module, "system_dynamics"):
                self._dynamics = custom_model_module.system_dynamics
            else:
                self._dynamics = self.model_module.system_dynamics

        # ------------------------------------------------------------
        # linearized dynamics
        # ------------------------------------------------------------
        if method.bools["jax_dyn"]:
            if custom_model_module and hasattr(custom_model_module, "analytical_linsys_jax"):
                self._lin_dyn = custom_model_module.analytical_linsys_jax
            else:
                self._lin_dyn = convexify.generate_lin_sys_jax(self._dynamics, problem)
        else:
            if custom_model_module and hasattr(custom_model_module, "analytical_linsys"):
                self._lin_dyn = custom_model_module.analytical_linsys
            else:
                self._lin_dyn = self.model_module.analytical_linsys

        # ------------------------------------------------------------
        # nonlinear inequality constraints
        # ------------------------------------------------------------
        if custom_model_module and hasattr(custom_model_module, "nonlinear_inequality_constraints"):
            self._nonlinear_inequality_constraints = custom_model_module.nonlinear_inequality_constraints
        else:
            self._nonlinear_inequality_constraints = self.model_module.nonlinear_inequality_constraints

        # ------------------------------------------------------------
        # linearized constraints
        # ------------------------------------------------------------
        if method.bools["auto_jac_cnst"]:
            self._lin_constr = convexify.generate_jacobians(self.nonlinear_inequality_constraints)
        else:
            if custom_model_module and hasattr(custom_model_module, "analytical_inequality_constraints"):
                self._lin_constr = custom_model_module.analytical_inequality_constraints
            else:
                self._lin_constr = self.model_module.analytical_inequality_constraints

        # ------------------------------------------------------------
        # CTCS / state vector update
        # ------------------------------------------------------------
        if method.bools.get("ctcs", False):
            self.nz = self.n + mission.n_ineq
        else:
            self.nz = self.n
        mission.n_dyn = self.nz
