import numpy as np
import importlib
from typing import Dict, Any, Optional, List

import numpy as np
import importlib

import trajopt.utils.tools as tools
import trajopt.core.modules.method.initial_guess as guess
import trajopt.core.modules.method.convergence as convergence
import trajopt.core.modules.method.convexify as convexify
import trajopt.core.modules.model.constraints as constraints_module
import trajopt.core.modules.model.costs as costs_module

class Problem:

    def __init__(self, config):

        # ████████████████████████████████████████████████████████████████████████████
        # █                                                                          █
        # █                    M O D E L    C O N F I G S                            █
        # █                                                                          █
        # ████████████████████████████████████████████████████████████████████████████

        self.model_name = config['model']
        self.model_module = importlib.import_module(f"trajopt.core.modules.model.{self.model_name}")
        self.model_config = config['model_config']

        self.model = self.model_module.Model(self.model_config)

        # ████████████████████████████████████████████████████████████████████████████
        # █                                                                          █
        # █                    M I S S I O N    C O N F I G S                        █
        # █                                                                          █
        # ████████████████████████████████████████████████████████████████████████████

        
        self.mission_name = config['mission']
        self.mission_module = importlib.import_module(f"trajopt.core.modules.mission.{self.mission_name}")
        self.mission_config = config['mission_config']

        self.mission = self.mission_module.Mission(self.mission_config)


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

        if "ct" in self.constraint_ids:
            f_ctcs, dfcn_dz_ctcs, dfcn_du_ctcs = convexify.linearize_jax_ctcs(self.dynamics)

            lin_dyn_ctcs = lambda t, z, nu: (f_ctcs(z, nu), dfcn_dz_ctcs(z, nu), dfcn_du_ctcs(z, nu))
            
            self.lin_dyn_ctcs = lin_dyn_ctcs

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

        self.constraints = []
        self.constraint_ids = {}

        # build constraint_ids mapping
        for i, constraint_config in enumerate(config["constraints"]):
            constraint_type = constraint_config["type"]
            constraint_params = {k:v for k, v in constraint_config.items() if k != "type"}
            constraintClass = getattr(constraints_module, constraint_type)
            self.constraints.append(constraintClass(**constraint_params))

            # add constraint to constraint_id map for indexing into list
            ct_type = "ct" if constraint_config["ct"] else "nodal"
                
            if ct_type not in self.constraint_ids:
                self.constraint_ids[ct_type] = {}
                self.constraint_ids[ct_type]['all'] = []

            if constraint_type not in self.constraint_ids[ct_type]:
                self.constraint_ids[ct_type][constraint_type] = []
            
            self.constraint_ids[ct_type][constraint_type].append(i)
            self.constraint_ids[ct_type]['all'].append(i)

        # constraint book keeping
        if "nonconvex_inequality" in self.constraint_ids["nodal"]:
            nodal_ncvx_ineq_ids = self.constraint_ids["nodal"]["nonconvex_inequality"]
            self.n_ineq = sum(constraint.dimension for constraint in self.constraints[nodal_ncvx_ineq_ids])
        else:
            self.n_ineq = 0

        if "ct" in self.constraint_ids:
            ct_ids = self.constraint_ids["ct"]["all"]
            self.n_ctcs = sum(constraint.dimension for constraint in self.constraints[ct_ids])
        else:
            self.n_ctcs = 0

        self.nz = self.n + self.n_ctcs

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

        # build constraint_ids mapping
        for i, cost_config in enumerate(config["costs"]):
            cost_type = cost_config["type"]
            costClass = getattr(costs_module, cost_type)
            self.costs.append(costClass(**cost_config))

            if cost_type not in self.cost_ids:
                self.cost_ids[cost_type] = []
            
            self.cost_ids[cost_type].append(i)

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
