import numpy as np
import importlib
from typing import Dict, Any, Optional, List

import numpy as np
import importlib

import trajopt.core.modules.method.convexify as convexify
from trajopt.core.Constraints import Constraints
from trajopt.core.Costs import Costs

# ████████████████████████████████████████████████████████████████████████████

# TODO: STILL UNDER CONSTRUCTION, RUNS AND "CONVERGES" for:
# examples/lander_6dof/standalone_prototype.ipynb

# ████████████████████████████████████████████████████████████████████████████

class Problem:

    def __init__(self, config):

        # ████████████████████████████████████████████████████████████████████████████
        # █                                                                          █
        # █                    M I S S I O N    C O N F I G S                        █
        # █                                                                          █
        # ████████████████████████████████████████████████████████████████████████████

        self.mission_config = config['problem']['mission']
        self.mission_name = config['problem']['mission']['name']
        self.mission_module = importlib.import_module(f"trajopt.core.modules.mission.{self.mission_name}")

        self.mission = self.mission_module.Mission(self.mission_config)

        # ████████████████████████████████████████████████████████████████████████████
        # █                                                                          █
        # █                    M O D E L    C O N F I G S                            █
        # █                                                                          █
        # ████████████████████████████████████████████████████████████████████████████

        self.model_config = config['problem']['model']
        self.model_name = config['problem']['model']['name']
        self.model_module = importlib.import_module(f"trajopt.core.modules.model.{self.model_name}")

        self.model = self.model_module.Model(self.model_config, self.mission)

        self.n = self.model.n
        self.m = self.model.m

        # ████████████████████████████████████████████████████████████████████████████
        # █                                                                          █
        # █                         O C P   D E F I N I T I O N                      █
        # █                                                                          █
        # ████████████████████████████████████████████████████████████████████████████

        # ------------------------------------------------------------
        # System Dynamics
        # ------------------------------------------------------------
        self.dynamics = self.model.dynamics_jax

        # ------------------------------------------------------------
        # Linearized Dynamics
        # ------------------------------------------------------------

        f, dfcn_dz, dfcn_du = convexify.linearize_jax(self.dynamics)
        lin_dyn = lambda t, z, nu: (f(z, nu), dfcn_dz(z, nu), dfcn_du(z, nu))

        self.lin_dyn = lin_dyn

        # ------------------------------------------------------------
        # Constraints
        # ------------------------------------------------------------

        # TODO (carlos): update this documentation and add more features
        # to the costs class so less for loops are needed outside the class

        # ------------------------------------------------------------
        # constraints class contains a list of constraint objects with 
        # a mapping of constraint type to a list of constraint ids for
        # fast lookup
        #
        # example usage of constraints class:
        # nonconvex_inequality_constraints = self.constraints.get('nodal', 'nonconvex_inequality')
        # ctcs_constraints = self.constraints.get('ct', 'all')
        #
        # each constraint object is an instantiation from the constraints_library module
        #
        # example structure of constaint_ids
        # (type: dict[str, dict[str, list[int]]]) 
        # 
        #
        # constraints = [axis_angle_cone, axis_angle_cone,  quaternion_cone, quaternion_cone, max_norm_cone]
        # constraint_ids = {
        #   "ct": {"control_axis_angle_cone": [0, 1], "state_max_norm_cone": [4]},
        #   "nodal": {"quaternion_cone": [2, 3]} 
        # }
        #
        # ------------------------------------------------------------

        constraint_config_list = config["problem"]["constraints"]
        self.constraints = Constraints(constraint_config_list)

        # constraint book keeping
        self.n_ineq = sum(constraint.dimension for constraint in self.constraints.get('nodal', 'nonconvex_inequality'))

        # TODO: should the algorithm need to distinguish between path, nfz, and custom, can we collapse into n_ineq?
        # TODO: ADD this to constraints class lol, ideally, shouldn't need any loops
        self.n_path = sum(constraint.dimension for constraint in self.constraints.get('nodal', 'nonconvex_inequality') if constraint.group == "path")
        self.n_nfz = sum(constraint.dimension for constraint in self.constraints.get('nodal', 'nonconvex_inequality') if constraint.group == "nfz")
        self.n_custom = sum(constraint.dimension for constraint in self.constraints.get('nodal', 'nonconvex_inequality') if constraint.group == "custom")

        if self.constraints.has('ct'):
            self.n_ctcs = sum(constraint.dimension for constraint in self.constraints.get('ct', 'all'))
        else:
            self.n_ctcs = 0

        self.nz = self.n + self.n_ctcs

        # TODO: same here
        self.n_term = sum(constraint.dimension for constraint in self.constraints.get('nodal', 'equality_bc') if constraint.boundary == "final" and constraint.set == "state")
        self.n_term_ineq = sum(constraint.dimension for constraint in self.constraints.get('nodal', 'inequality_bc') if constraint.boundary == "final" and constraint.set == "state")
        self.n_term_ctcs = self.n_ctcs
        self.n_term_total = self.n_term + self.n_term_ineq + self.n_ctcs

        # ------------------------------------------------------------
        # Augmented CTCS dynamics
        # ------------------------------------------------------------

        if self.constraints.has('ct'):
            f_ctcs, dfcn_dz_ctcs, dfcn_du_ctcs = convexify.linearize_jax_ctcs(self.dynamics, self.constraints, self.n)

            lin_dyn_ctcs = lambda t, z, nu: (f_ctcs(z, nu), dfcn_dz_ctcs(z, nu), dfcn_du_ctcs(z, nu))
            
            self.lin_dyn = lin_dyn_ctcs

        # ------------------------------------------------------------
        # Cost
        # ------------------------------------------------------------

        # TODO (carlos): update this documentation and add more features
        # to the costs class for more granular constraint lookups

        # similar structure to the constraints class without the outer
        # "ct" / "nodal" distinction, usage is the same

        cost_config_list = config["problem"]["costs"]
        self.costs = Costs(cost_config_list)

        # resolve functions
        for constraint in self.constraints.constraints_list:
            if getattr(constraint, 'fcn_name', None) is not None:
                obj_name, func_name = constraint.fcn_name   .split(".")

                if obj_name == "model":
                    obj = self.model
                
                elif obj_name == "mission":
                    obj = self.mission

                if constraint.fcn_params == {}:
                    constraint.fcn = getattr(obj, func_name)
                else:
                    constraint.fcn = lambda t, z, nu: getattr(obj, func_name)(t, z, nu, constraint.fcn_params)

                constraint.fcn_jit, constraint.dfcn_dz_jit, constraint.dfcn_du_jit = convexify.linearize_jax(constraint.fcn)

        # # ------------------------------------------------------------
        # # Custom Input/Variable/Constraint/Cost
        # # ------------------------------------------------------------
        # self._set_custom_params = _resolve_function("set_custom_params")
        # self._custom_constraints = _resolve_function("custom_constraints")
        # self._custom_cost = _resolve_function("custom_cost")

        # self.set_custom_params()


# TODO (carlos): add this back

# """
# Attach mission functions for costs and constraints.
# Uses custom_modules from YAML config if specified, otherwise uses base mission.
# """
# model = self.trajopt_obj.model
# method = self.trajopt_obj.method
# trajopt_obj = self.trajopt_obj

# # Load YAML custom module mappings if present
# if self.custom_modules:
#     yaml_funcs = {}
#     for key, path in self.custom_modules.items():
#         try:
#             mod_path, attr = path.rsplit(".", 1)
#             mod = importlib.import_module(mod_path)
#             yaml_funcs[key] = getattr(mod, attr)
#         except Exception as e:
#             print(f"⚠️ Could not import custom mission module for '{key}' from {path}: {e}")
#     self.custom_modules = yaml_funcs
# else:
#     self.custom_modules = {}
# # Resolver: custom_modules > base mission
# def _resolve_function(name: str):
#     if name in self.custom_modules:
#         return self.custom_modules[name]
#     return getattr(self.mission_module, name)
