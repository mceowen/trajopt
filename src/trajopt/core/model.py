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

        model_config          = config["model"]
        self.model_name       = model_config["model_name"]
        self.n                = model_config["n"]
        self.m                = model_config["m"]
        self.z_types          = model_config["z_types"]
        self.u_types          = model_config["u_types"]
        self.anchor_types     = model_config["anchor_types"]
        self.anchor_scales    = model_config["anchor_scales"]
        self.base_unit_labels = model_config["base_unit_labels"]
        self.bools            = model_config["bools"]

        self.custom_modules   = model_config.get("custom_modules", None)

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
        """
        Attach model functions for dynamics, linearizations, and constraints.

        Priority order for each callable:
        1. YAML-defined module override path (config["model"]["custom_modules"] or ["modules"])
        2. Function defined in examples/<example_name>/custom.py
        3. Default from trajopt.core.modules.models.<model_name>
        """

        problem = self.problem
        method = problem.method
        mission = problem.mission
        base_mod = self.model_module  # default model module

        # ------------------------------------------------------------
        # Try to import a custom module under trajopt.examples.<example_name>.custom
        # ------------------------------------------------------------
        custom_model_module = None
        try:
            # use problem.name if available, else fall back to model_name
            example_name = getattr(problem, "name", self.model_name)
            if example_name:
                custom_model_module = importlib.import_module(
                    f"trajopt.examples.{example_name}.custom"
                )
        except ModuleNotFoundError:
            pass  # no custom module found

        # ------------------------------------------------------------
        # Load YAML custom module mappings (if present)
        # ------------------------------------------------------------
        custom_map = self.custom_modules
        yaml_funcs = None

        if custom_map is not None:
            yaml_funcs = {}
            for key, path in custom_map.items():
                try:
                    mod_path, attr = path.rsplit(".", 1)
                    mod = importlib.import_module(mod_path)
                    yaml_funcs[key] = getattr(mod, attr)
                except Exception as e:
                    print(f"⚠️ Could not import custom module for '{key}' from {path}: {e}")

        # literal None if not defined
        self.custom_modules = yaml_funcs if yaml_funcs else None

        # ------------------------------------------------------------
        # Unified resolver: YAML > custom.py > base model
        # ------------------------------------------------------------
        def _resolve_function(name: str):
            # 1. YAML override
            if self.custom_modules and name in self.custom_modules:
                return self.custom_modules[name]

            # 2. custom.py override
            if custom_model_module and hasattr(custom_model_module, name):
                return getattr(custom_model_module, name)

            # 3. fallback to base model
            return getattr(base_mod, name)

        # ------------------------------------------------------------
        # System Dynamics
        # ------------------------------------------------------------
        if method.bools.get("jax_dyn", 0):
            self._dynamics = _resolve_function("system_dynamics_jax")
        else:
            self._dynamics = _resolve_function("system_dynamics")

        # ------------------------------------------------------------
        # Linearized Dynamics
        # ------------------------------------------------------------
        if method.bools.get("jax_dyn", 0):
            self._lin_dyn = convexify.generate_lin_sys_jax(self._dynamics, problem)
        else:
            self._lin_dyn = _resolve_function("analytical_linsys")

        # ------------------------------------------------------------
        # Nonlinear Inequality Constraints
        # ------------------------------------------------------------
        self._nonlinear_inequality_constraints = _resolve_function(
            "nonlinear_inequality_constraints"
        )

        # ------------------------------------------------------------
        # Linearized Inequality Constraints
        # ------------------------------------------------------------
        if method.bools.get("auto_jac_cnst", 0):
            self._lin_constr = convexify.generate_jacobians(
                self.nonlinear_inequality_constraints
            )
        else:
            self._lin_constr = _resolve_function("analytical_inequality_constraints")

        # ------------------------------------------------------------
        # CTCS / state bookkeeping
        # ------------------------------------------------------------
        if method.bools.get("ctcs", False):
            self.nz = self.n + mission.n_ineq
        else:
            self.nz = self.n
        mission.n_dyn = self.nz


