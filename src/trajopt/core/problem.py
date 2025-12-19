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

    def __init__(self, trajopt_obj, config):

        self.trajopt_obj = trajopt_obj


        # ████████████████████████████████████████████████████████████████████████████
        # █                                                                          █
        # █                    M O D E L    P A R A M E T E R S                      █
        # █                                                                          █
        # ████████████████████████████████████████████████████████████████████████████

        # load config parameters

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

        # Load constraints

        # ------------------------------------------------------------
        # example structure of constaint_ids
        # (type: dict[str, dict[str, list[int]]]) 
        # 
        # constraints = []
        # constraint_ids = {
        #   "ct": {"gimbal_cone": [0, 1], "constraint_type2": [4]}
        #   "nodal": {"constraint_type3": [2, 3]} 
        # }
        # ------------------------------------------------------------

        self.constraints = []
        self.constraint_ids = {}

        # build constraint_ids mapping
        for i, constraint_config in enumerate(model_config["constraints"]):
            constraint_type = constraint_config["type"]
            constraintClass = getattr(constraints_module, constraint_type)
            self.constraints.append(constraintClass(**constraint_config))

            # add constraint to constraint_id map for indexing into list
            ct_type = "ct" if constraint_config["ct"] else "nodal"
                
            if ct_type not in self.constraint_ids:
                self.constraint_ids[ct_type] = {}

            if constraint_type not in self.constraint_ids[ct_type]:
                self.constraint_ids[ct_type][constraint_type] = []
            
            self.constraint_types[ct_type][constraint_type].append(i)

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
            self.n_ctcs

        # point to module containing corresponding methods based on configs

        # point to selected model module
        self.model_module = importlib.import_module(model_config["module_path"])


        # ████████████████████████████████████████████████████████████████████████████
        # █                                                                          █
        # █                    M I S S I O N    P A R A M E T E R S                  █
        # █                                                                          █
        # ████████████████████████████████████████████████████████████████████████████

        
        mission_config         = config["mission"]
        self.name              = mission_config["module_path"].split(".")[-1]
        self.flags             = mission_config['flags']
        
        # standard constraint parameters
        self.zi_guess          = mission_config["zi_guess"]
        self.zf_guess          = mission_config["zf_guess"]

        self.planet            = mission_config["planet"]
        self.vehicle           = mission_config["vehicle"]

        self.custom_modules    = mission_config.get("custom_modules", None)

        # #TODO: Load costs similarly to constraints above (in model)

        self.mission_module = importlib.import_module(mission_config["module_path"])

        # set cost/constraint nondim setter function (needed for nondim initialization)
        self._get_cost_cnstr_nondim = self.mission_module.get_cost_cnstr_nondim




    # ████████████████████████████████████████████████████████████████████████████
    # █                                                                          █
    # █                    M O D E L    F U N C T I O N S                        █
    # █                                                                          █
    # ████████████████████████████████████████████████████████████████████████████

    def dynamics(self, t, z, nu):

        return self._dynamics(t, z, nu, self.trajopt_obj)
    
    def update_model_params(self):
        """
        Attach model functions for dynamics, linearizations, and constraints.
        Uses custom_modules from YAML config if specified, otherwise uses base model.
        """

        trajopt_obj = self.trajopt_obj
        method = trajopt_obj.method
        mission = trajopt_obj.mission
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

            f, dfcn_dz, dfcn_du = convexify.linearize_jax(self._dynamics, trajopt_obj)

            def lin_dyn(t, z, nu):
                return f(z, nu), dfcn_dz(z, nu), dfcn_du(z, nu)

            if method.flags["ctcs"] != "none":
                f_ctcs, dfcn_dz_ctcs, dfcn_du_ctcs = convexify.linearize_jax_ctcs(self._dynamics, trajopt_obj)

                def lin_dyn_ctcs(t, z, nu):
                    return f_ctcs(z, nu), dfcn_dz_ctcs(z, nu), dfcn_du_ctcs(z, nu)
                
                self.lin_dyn_ctcs = lin_dyn_ctcs

        else:
            _lin_dyn = _resolve_function("analytical_linsys")

            def lin_dyn(t, z, nu):
                return _lin_dyn(t, z, nu, trajopt_obj)

        self.lin_dyn = lin_dyn


    # ████████████████████████████████████████████████████████████████████████████
    # █                                                                          █
    # █                    M I S S I O N    F U N C T I O N S                    █
    # █                                                                          █
    # ████████████████████████████████████████████████████████████████████████████


    def nonlinear_aero(self, t, z, nu):
        return self._nonlinear_aero(t, z, nu, self.trajopt_obj)

    def custom_constraints(self, subtrajopt_obj):
        return self._custom_constraints(subtrajopt_obj)

    def custom_cost(self, subtrajopt_obj):
        return self._custom_cost(subtrajopt_obj)
    
    def get_cost_cnstr_nondim(self):
        return self._get_cost_cnstr_nondim(self.trajopt_obj)
        
    def set_custom_params(self):
        return self._set_custom_params(self.trajopt_obj)

    # ===============================================================
    # UPDATE PARAMETERS
    # ===============================================================
    def update_mission_params(self):
        """
        Attach mission functions for costs and constraints.
        Uses custom_modules from YAML config if specified, otherwise uses base mission.
        """
        model = self.trajopt_obj.model
        method = self.trajopt_obj.method
        trajopt_obj = self.trajopt_obj

        # Load YAML custom module mappings if present
        if self.custom_modules:
            yaml_funcs = {}
            for key, path in self.custom_modules.items():
                try:
                    mod_path, attr = path.rsplit(".", 1)
                    mod = importlib.import_module(mod_path)
                    yaml_funcs[key] = getattr(mod, attr)
                except Exception as e:
                    print(f"⚠️ Could not import custom mission module for '{key}' from {path}: {e}")
            self.custom_modules = yaml_funcs
        else:
            self.custom_modules = {}

        # Resolver: custom_modules > base mission
        def _resolve_function(name: str):
            if name in self.custom_modules:
                return self.custom_modules[name]
            return getattr(self.mission_module, name)

        # ------------------------------------------------------------
        # Cost & Linearized Cost
        # ------------------------------------------------------------

        # TODO (carlos): move to constructor and mirror the constraints
        # list implementation
        self.costs = []
        for cost_config in self.cost_config_list:
            self.costs.append(Cost(yaml_config=cost_config))

        convexify.convexify_costs(trajopt_obj)

        # ------------------------------------------------------------
        # Aerodynamics
        # ------------------------------------------------------------
        if self.flags["aero_type"] != "none":
            if method.flags["jax_dyn"]:
                self._nonlinear_aero = _resolve_function("nonlinear_aero_jax")
            else:
                self._nonlinear_aero = _resolve_function("nonlinear_aero")
        else:
            self._nonlinear_aero = None

        # ------------------------------------------------------------
        # Custom Input/Variable/Constraint/Cost
        # ------------------------------------------------------------
        self._set_custom_params = _resolve_function("set_custom_params")
        self._custom_constraints = _resolve_function("custom_constraints")
        self._custom_cost = _resolve_function("custom_cost")

        # ------------------------------------------------------------
        # Continue with the rest of nondim and constraint setup
        # ------------------------------------------------------------
        # Load CTCS terminal constraints if applicable
        self.n_term_ctcs  = (model.n_ctcs if method.flags['ctcs']=="term" else 0)

        # nondimensionalization
        self.M_zi = method.nondim["M"]["state"]["d2nd"][np.ix_(self.zi_idx, self.zi_idx)]
        self.M_zf = method.nondim["M"]["state"]["d2nd"][np.ix_(self.zf_idx, self.zf_idx)]
        self.zi = self.M_zi @ self.zi
        self.zf = self.M_zf @ self.zf

        if self.zi_guess.size > 0:
            self.zi_guess = method.nondim["M"]["state"]["d2nd"] @ self.zi_guess

        if self.zf_guess.size > 0:
            self.zf_guess = method.nondim["M"]["state"]["d2nd"] @ self.zf_guess

        M_z_min = method.nondim["M"]["state"]["d2nd"][np.ix_(self.z_min_idx, self.z_min_idx)]
        M_z_max = method.nondim["M"]["state"]["d2nd"][np.ix_(self.z_max_idx, self.z_max_idx)]
        self.z_min = M_z_min @ self.z_min
        self.z_max = M_z_max @ self.z_max

        if self.flags["init_ctrl"] == 1:
            self.ui = method.nondim["M"]["ctrl"]["d2nd"] @ self.ui
            self.uf = method.nondim["M"]["ctrl"]["d2nd"] @ self.uf

        M_u_min = method.nondim["M"]["ctrl"]["d2nd"][np.ix_(self.u_min_idx, self.u_min_idx)]
        M_u_max = method.nondim["M"]["ctrl"]["d2nd"][np.ix_(self.u_max_idx, self.u_max_idx)]
        self.u_min = M_u_min @ self.u_min
        self.u_max = M_u_max @ self.u_max

        M_udot_max = method.nondim["M"]["ctrl"]["d2nd"][np.ix_(self.udot_max_idx, self.udot_max_idx)]
        self.udot_max = M_udot_max @ self.udot_max * method.nondim["nt"]

        self.set_custom_params()

class Cost:
    def __init__(self, func=None, yaml_config=None, **kwargs):
        
        # constraint configs mainly pull from config files, but this class can also be used
        # to define constraints from other places like the obstacle constraints
        if yaml_config is not None:
            self.convex     = yaml_config.get('convex', 0)
            self.auto_diff  = yaml_config.get('auto_diff', 1)
            self.name       = yaml_config.get('analytical_affine_approximation', None).split('.')[-1]
            self.category   = yaml_config.get('category', "running")

            # import functions from config strings
            self.func = tools._import_from_string(yaml_config.get('function', None))
            self.analytical_affine_approximation = tools._import_from_string(yaml_config.get('analytical_affine_approximation', None))
        
        else:
            self.convex     = kwargs.get('convex', 1)
            self.auto_diff  = kwargs.get('auto_diff', 0)
            self.name       = func.__name__
            self.category   = kwargs.get('category', "running")
            
            self.func       = func
            self.analytical_affine_approximation = kwargs.get('analytical_affine_approximation', None)