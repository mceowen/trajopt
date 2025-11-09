import numpy as np
import importlib
from typing import Dict, Any, Optional, List

import numpy as np
import importlib

import trajopt.core.modules.methods.convexify as convexify
import trajopt.utils.tools as tools

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
        self.flags            = model_config['flags']

        self.obs              = model_config["obs"]

        self.custom_modules               = model_config.get("custom_modules", None)
        self.constraint_config_list       = model_config.get("constraints", None)

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
        if method.flags.get("jax_dyn", 0):
            self._dynamics = _resolve_function("dynamics_jax")
        else:
            self._dynamics = _resolve_function("dynamics")

        # ------------------------------------------------------------
        # Linearized Dynamics
        # ------------------------------------------------------------
        if method.flags.get("jax_dyn", 0):
            f, dfcn_dz, dfcn_du = convexify.linearize_jax(self._dynamics, problem)

            def lin_dyn(ts, zs, us):
                return f(zs, us), dfcn_dz(zs, us), dfcn_du(zs, us)

        else:
            _lin_dyn = _resolve_function("analytical_linsys")

            def lin_dyn(ts, zs, us):
                return _lin_dyn(ts, zs, us, problem)

        self.lin_dyn = lin_dyn

        # ------------------------------------------------------------
        # get constraints from configs and convexify them
        # ------------------------------------------------------------

        self.constraints = []
        for constraint_config in self.constraint_config_list:
            self.constraints.append(Constraint(yaml_config=constraint_config))

        convexify.convexify_constraints(problem)

        self.nonconvex_nodal_constraints = [c for c in self.constraints if not c.convex and not c.always]
        self.nonconvex_ct_constraints    = [c for c in self.constraints if not c.convex and     c.always]
        self.convex_constraints          = [c for c in self.constraints if     c.convex]

        # ------------------------------------------------------------
        # CTCS / state bookkeeping
        # ------------------------------------------------------------
        if method.flags.get("ctcs", False):
            self.nz = self.n + mission.n_ineq
        else:
            self.nz = self.n
        mission.n_dyn = self.nz

class Constraint:
    def __init__(self, func=None, yaml_config=None, **kwargs):
        
        # constraint configs mainly pull from config files, but this class can also be used
        # to define constraints from other places like the obstacle constraints
        if yaml_config is not None:
            self.convex     = yaml_config.get('convex', 0)
            self.always     = yaml_config.get('always', 0) # "always", "now"
            self.auto_diff  = yaml_config.get('auto_diff', 1)
            self.type       = yaml_config.get('type', 'inequality')
            self.dimension  = yaml_config.get('dimension', 0)
            self.name       = yaml_config.get('analytical_affine_approximation', None).split('.')[-1]

            # import functions from config strings
            self.func                 = tools._import_from_string(yaml_config.get('function', None))
            self.analytical_affine_approximation = tools._import_from_string(yaml_config.get('analytical_affine_approximation', None))
        
        else:
            self.convex     = kwargs.get('convex', 1)
            self.always     = kwargs.get('when', 'always_discrete') # "always", "now"
            self.auto_diff  = kwargs.get('auto_diff', 0)
            self.type       = kwargs.get('type', 'inequality')
            self.dimension  = kwargs.get('dimension', 0)
            self.name       = func.__name__
            
            self.func       = func
            self.analytical_affine_approximation = kwargs.get('analytical_affine_approximation', None)
    