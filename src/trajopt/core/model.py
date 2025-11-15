import numpy as np
import importlib
from typing import Dict, Any, Optional, List

import numpy as np
import importlib

import trajopt.core.modules.method.convexify as convexify
import trajopt.utils.tools as tools

class Model:

    def __init__(self, problem, config):

        self.problem = problem

        # ===============================================================
        # load config parameters
        # ===============================================================

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

        self.constraints = []
        for constraint_config in self.constraint_config_list:
            self.constraints.append(Constraint(yaml_config=constraint_config))

    # ===============================================================
    # member functions point to selected fcns from selected module
    # ===============================================================

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

        # ------------------------------------------------------------
        # get constraints from configs and convexify them
        # ------------------------------------------------------------

        convexify.convexify_constraints(problem)

        # TODO (CARLOS): all nonconvex constraints assumed to be inequality and linearized for now
        # will update to more granular classifcations soon
        self.nonconvex_inequality_constraints = [c for c in self.constraints if not c.convex]
        self.convex_constraints               = [c for c in self.constraints if     c.convex]

        # ------------------------------------------------------------
        # CTCS / state bookkeeping
        # ------------------------------------------------------------
        if method.flags["ctcs"] != "none":
            self.nz = self.n + mission.n_ineq
        else:
            self.nz = self.n
        mission.n_dyn = self.nz

class Constraint:
    def __init__(self, fcn=None, yaml_config=None, **kwargs):
        
        # constraint configs mainly pull from config files, but this class can also be used
        # to define constraints from other places like the obstacle constraints
        if yaml_config is not None:
            self.convex     = yaml_config.get('convex', 0)
            self.always     = yaml_config.get('always', 0) # "always", "now"
            self.auto_diff  = yaml_config.get('auto_diff', 1)
            self.type       = yaml_config.get('type', 'inequality')
            self.dimension  = yaml_config.get('dimension', 0)
            self.jax        = yaml_config.get('jax', 0)
            self.sympy      = yaml_config.get('sympy', 0)
            self.name       = yaml_config.get('fcn', None).split('.')[-1]
            self.units      = yaml_config.get('units', None)

            # import functions from config strings
            # Handle YAML parsing where None might be parsed as string "None"
            analytical_affine_approx_str = yaml_config.get('analytical_affine_approximation', None)
            if analytical_affine_approx_str == "None" or analytical_affine_approx_str is None:
                analytical_affine_approx_str = None
            
            self.fcn                            = tools._import_from_string(yaml_config.get('fcn', None))
            self.analytical_affine_approximation = tools._import_from_string(analytical_affine_approx_str)
        
        else:
            self.convex     = kwargs.get('convex', 1)
            self.always     = kwargs.get('when', 'always_discrete') # "always", "now"
            self.auto_diff  = kwargs.get('auto_diff', 0)
            self.type       = kwargs.get('type', 'inequality')
            self.dimension  = kwargs.get('dimension', 0)
            self.jax        = kwargs.get('jax', 0)
            self.sympy      = kwargs.get('sympy', 0)
            self.name       = fcn.__name__
            self.units      = kwargs.get('units', None)
            
            self.fcn       = fcn
            self.analytical_affine_approximation = kwargs.get('analytical_affine_approximation', None)
    