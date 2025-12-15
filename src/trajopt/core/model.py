import numpy as np
import importlib
from typing import Dict, Any, Optional, List

import numpy as np
import importlib

import trajopt.core.modules.method.convexify as convexify

class Model:

    def __init__(self, problem, config):

        self.problem = problem

        # ===============================================================
        # load config parameters
        # ===============================================================

        self.mission_config   = config["mission"]
        model_config          = config["model"]
        self.name             = model_config["module_path"].split(".")[-1]
        self.n                = model_config["n"]
        self.m                = model_config["m"]
        self.z_types          = model_config["z_types"]
        self.u_types          = model_config["u_types"]
        self.anchor_types     = model_config["anchor_types"]
        self.anchor_scales    = model_config["anchor_scales"]
        self.base_unit_labels = model_config["base_unit_labels"]
        self.flags            = model_config['flags']

        self.obs              = model_config["obs"]

        self.custom_modules               = model_config.get("custom_modules", None)
        self.constraint_config_list       = model_config.get("constraints", None)

        # =================================================================
        # point to module containing corresponding methods based on configs
        # =================================================================

        # point to selected model module
        self.model_module = importlib.import_module(model_config["module_path"])

    # ===============================================================
    # member functions point to selected fcns from selected module
    # ===============================================================

    # updated nondim will look like this, i commented it out to keep compatibility with old code
    # def dynamics(self, t, z, nu):

    #     method = self.problem.method

    #     t_dim = method.nondim["nt"] * t
    #     z_dim = method.nondim["M"]["state"]["nd2d"] @ z
    #     nu_dim = method.nondim["M"]["ctrl"]["nd2d"] @ nu

    #     x_dot_dim = self._dynamics(t_dim, z_dim, nu_dim, self.problem)

    #     return method.nondim["M"]["dyn"]["d2nd"] @ x_dot_dim

    def dynamics(self, t, z, nu):

        return self._dynamics(t, z, nu, self.problem)
    
    def update_model_params(self):
        """
        Attach model functions for dynamics, linearizations, and constraints.
        Uses custom_modules from YAML config if specified, otherwise uses base model.
        """

        problem = self.problem
        method = problem.method
        mission = problem.mission
        base_mod = self.model_module

        # Load YAML custom module mappings if present
        if self.custom_modules:
            yaml_funcs = {}
            for key, path in self.custom_modules.items():
                try:
                    mod_path, attr = path.rsplit(".", 1)
                    mod = importlib.import_module(mod_path)
                    yaml_funcs[key] = getattr(mod, attr)
                except Exception as e:
                    print(f"⚠️ Could not import custom module for '{key}' from {path}: {e}")
            self.custom_modules = yaml_funcs
        else:
            self.custom_modules = {}

        # Resolver: custom_modules > base model
        def _resolve_function(name: str):
            if name in self.custom_modules:
                return self.custom_modules[name]
            return getattr(base_mod, name)

        # ------------------------------------------------------------
        # System Dynamics
        # ------------------------------------------------------------
        if method.flags.get("jax_dyn", 0):
            self._dynamics = _resolve_function("dynamics_jax")
        else:
            self._dynamics = _resolve_function("dynamics")

        # ------------------------------------------------------------
        # Linearized Dynamics
        # ------------------------------------------------------------
        if method.flags.get("jax_dyn", 0):

            f, dfcn_dz, dfcn_du = convexify.linearize_jax(self._dynamics, problem)

            def lin_dyn(t, z, nu):
                return f(z, nu), dfcn_dz(z, nu), dfcn_du(z, nu)

            if method.flags["ctcs"] != "none":
                f_ctcs, dfcn_dz_ctcs, dfcn_du_ctcs = convexify.linearize_jax_ctcs(self._dynamics, problem)

                def lin_dyn_ctcs(t, z, nu):
                    return f_ctcs(z, nu), dfcn_dz_ctcs(z, nu), dfcn_du_ctcs(z, nu)
                
                self.lin_dyn_ctcs = lin_dyn_ctcs

        else:
            _lin_dyn = _resolve_function("analytical_linsys")

            def lin_dyn(t, z, nu):
                return _lin_dyn(t, z, nu, problem)

        self.lin_dyn = lin_dyn