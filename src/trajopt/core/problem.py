import numpy as np
import importlib
from typing import Dict, Any, Optional, List

import numpy as np
import importlib

import trajopt.core.modules.utils.tools as tools
import trajopt.core.modules.method.initial_guess as guess
import trajopt.core.modules.method.convergence as convergence
import trajopt.core.modules.method.convexify as convexify
import trajopt.core.modules.model.constraints_library as constraints_module
from trajopt.core.Constraints import Constraints
import trajopt.core.modules.model.costs_library as costs_module

from pprint import pprint

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

        # ------------------------------------------------------------
        # example structure of constaint_ids
        # (type: dict[str, dict[str, list[int]]]) 
        # 
        #
        # constraints = [control_axis_angle_cone, control_axis_angle_cone,  quaternion_cone, quaternion_cone, state_max_norm_cone]
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

        # TODO: temp?
        self.n_path = sum(constraint.dimension for constraint in self.constraints.get('nodal', 'nonconvex_inequality') if constraint.group == "path")
        self.n_nfz = sum(constraint.dimension for constraint in self.constraints.get('nodal', 'nonconvex_inequality') if constraint.group == "nfz")
        self.n_custom = sum(constraint.dimension for constraint in self.constraints.get('nodal', 'nonconvex_inequality') if constraint.group == "custom")

        if self.constraints.has('ct'):
            self.n_ctcs = sum(constraint.dimension for constraint in self.constraints.get('ct', 'all'))
        else:
            self.n_ctcs = 0

        self.nz = self.n + self.n_ctcs

        self.n_term = sum(constraint.dimension for constraint in self.constraints.get('nodal', 'equality_bc') if constraint.boundary == "final")
        self.n_term_ineq = sum(constraint.dimension for constraint in self.constraints.get('nodal', 'inequality_bc') if constraint.boundary == "final" and constraint.set == "state")
        self.n_term_ctcs = self.n_ctcs
        
        print(f"\nconstraints loaded successfully!")
        print("constraint_ids: \n")
        pprint(self.constraints.constraint_ids)

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

        # ------------------------------------------------------------
        # example structure of cost_ids
        # (type: dict[str, list[int]]) 
        # 
        #
        # costs = [cost_type_1, cost_type_2]
        # cost_ids = {
        #   "cost_type_1": [0],
        #   "cost_type_2": [1]
        # }
        #
        # ------------------------------------------------------------

        self.costs = []
        self.cost_ids = {}

        print(f"\nloading costs:")

        # build constraint_ids mapping
        for i, cost_config in enumerate(config["problem"]["costs"]):
            cost_type = cost_config["type"]
            print(f"  {i}: {cost_type}")
            cost_params = {k:v for k, v in cost_config.items() if k != "type"}
            costClass = getattr(costs_module, cost_type)
            self.costs.append(costClass(**cost_params))

            if cost_type not in self.cost_ids:
                self.cost_ids[cost_type] = []
            
            self.cost_ids[cost_type].append(i)

        print("\ncosts loaded successfully!")
        print("cost_ids: \n")

        pprint(self.cost_ids)

        # # ------------------------------------------------------------
        # # Custom Input/Variable/Constraint/Cost
        # # ------------------------------------------------------------
        # self._set_custom_params = _resolve_function("set_custom_params")
        # self._custom_constraints = _resolve_function("custom_constraints")
        # self._custom_cost = _resolve_function("custom_cost")

        # self.set_custom_params()









# TODO: add this back

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
