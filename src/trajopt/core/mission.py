import numpy as np
import importlib

import trajopt.utils.tools as tools
import trajopt.core.modules.methods.initial_guess as guess
import trajopt.core.modules.methods.convergence as convergence
import trajopt.core.modules.methods.convexify as convexify
import trajopt.utils.nondim as nondim

class Mission:
    def __init__(self, problem, config):

        self.problem = problem

        # ===============================================================
        # load dimensional config parameters
        # ===============================================================
        
        mission_config         = config["mission"]
        self.mission_name      = mission_config["mission_name"]
        self.flags             = mission_config['flags']
        
        # standard constraint parameters
        self.zi                = mission_config["zi"]
        self.zi_idx            = mission_config["zi_idx"]
        self.zf                = mission_config["zf"]
        self.zf_idx            = mission_config["zf_idx"]
        self.ui                = mission_config["ui"]
        self.uf                = mission_config["uf"]
        self.ui_idx            = mission_config["ui_idx"]
        self.uf_idx            = mission_config["uf_idx"]
        self.z_min             = mission_config["z_min"]
        self.z_min_idx         = mission_config["z_min_idx"]
        self.z_max             = mission_config["z_max"]
        self.z_max_idx         = mission_config["z_max_idx"]
        self.udot_max          = mission_config["udot_max"]
        self.udot_max_idx      = mission_config["udot_max_idx"]
        self.zi_min            = mission_config["zi_min"]
        self.zi_max            = mission_config["zi_max"]
        self.zi_min_idx        = mission_config["zi_min_idx"]
        self.zi_max_idx        = mission_config["zi_max_idx"]
        self.zf_min            = mission_config["zf_min"]
        self.zf_max            = mission_config["zf_max"]
        self.zf_min_idx        = mission_config["zf_min_idx"]
        self.zf_max_idx        = mission_config["zf_max_idx"]
        self.u_min             = mission_config["u_min"]
        self.u_max             = mission_config["u_max"]
        self.u_min_idx         = mission_config["u_min_idx"]
        self.u_max_idx         = mission_config["u_max_idx"]
        
        # dictionaries
        self.path_limits       = mission_config["path_limits"]
        self.aux_limits        = mission_config["aux_limits"]
        self.custom_input_dict = mission_config["custom_input_dict"]

        self.planet            = mission_config["planet"]
        self.vehicle           = mission_config["vehicle"]
        self.obs               = mission_config["obs"]

        self.cost_config_list = mission_config.get("costs", None)

        self.custom_modules   = mission_config.get("custom_modules", None)

        # ------------------------------------------------------------
        # Constraint bookkeeping
        # ------------------------------------------------------------
        self.n_init       = len(self.zi_idx)
        self.n_term       = len(self.zf_idx)
        self.n_init_ctrl  = len(self.ui_idx)
        self.n_term_ctrl  = len(self.uf_idx)
        self.n_init_ineq  = len(self.zi_min_idx) + len(self.zi_max_idx)
        self.n_term_ineq  = len(self.zf_min_idx) + len(self.zf_max_idx)
        self.n_ctrl       = len(self.u_min_idx)  + len(self.u_max_idx)
        self.n_state      = len(self.z_min_idx)  + len(self.z_max_idx)
        self.n_udot       = len(self.udot_max_idx)
        self.n_path       = sum(1 if np.isscalar(v) else len(v) for v in self.path_limits.values())
        self.n_nfz        = len(self.obs["xc"])
        self.n_aux        = sum(1 if np.isscalar(v) else len(v) for v in self.aux_limits.values())
        self.n_ineq       = self.n_path + self.n_nfz + self.n_aux

        self.path_idx     = np.arange(self.n_path)
        self.nfz_idx      = np.arange(self.n_path, self.n_path + self.n_nfz)
        self.aux_idx      = np.arange(self.n_path + self.n_nfz, self.n_ineq)

        # ===============================================================
        # point to module and corresponding methods based on configs
        # ===============================================================

        self.mission_module = importlib.import_module(f"trajopt.core.modules.missions.{self.mission_name}")

        # set cost/constraint nondim setter function (needed for nondim initialization)
        self._get_cost_cnstr_nondim = self.mission_module.get_cost_cnstr_nondim

    # ===============================================================
    # member functions point to selected fcns from selected module
    # ===============================================================
    
    def nonlinear_aero(self, ts, zs, us):
        return self._nonlinear_aero(ts, zs, us, self.problem)

    def custom_constraints(self, subproblem):
        return self._custom_constraints(subproblem)

    def custom_cost(self, subproblem):
        return self._custom_cost(subproblem)
    
    def get_cost_cnstr_nondim(self):
        return self._get_cost_cnstr_nondim(self.problem)
        
    def set_custom_params(self):
        return self._set_custom_params(self.problem)

    # ===============================================================
    # UPDATE PARAMETERS
    # ===============================================================
    def update_mission_params(self):
        method = self.problem.method
        problem = self.problem

        # ------------------------------------------------------------
        # Try to import example-level custom overrides
        # ------------------------------------------------------------
        custom_mission_module = None
        try:
            example_name = getattr(problem, "name", self.mission_name)
            if example_name:
                custom_mission_module = importlib.import_module(
                    f"trajopt.examples.{example_name}.custom"
                )
        except ModuleNotFoundError:
            pass

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
                    print(f"⚠️ Could not import custom mission module for '{key}' from {path}: {e}")

        # literal None if not defined
        self.custom_modules = yaml_funcs if yaml_funcs else None

        # unified resolver: YAML > custom.py > base module
        def _resolve_function(name: str):
            if self.custom_modules and name in self.custom_modules:
                return self.custom_modules[name]
            if custom_mission_module and hasattr(custom_mission_module, name):
                return getattr(custom_mission_module, name)
            return getattr(self.mission_module, name)

        # ------------------------------------------------------------
        # Cost & Linearized Cost
        # ------------------------------------------------------------

        self.costs = []
        for cost_config in self.cost_config_list:
            self.costs.append(Cost(yaml_config=cost_config))

        convexify.convexify_costs(problem)

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
        method = self.problem.method
        # nondimensionalization
        self.M_zi = method.nondim["M"]["state"]["d2nd"][np.ix_(self.zi_idx, self.zi_idx)]
        self.M_zf = method.nondim["M"]["state"]["d2nd"][np.ix_(self.zf_idx, self.zf_idx)]
        self.zi = self.M_zi @ self.zi
        self.zf = self.M_zf @ self.zf

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

        # TODO (CARLOS): make nondim correct for both distances and angles
        self.obs["xc"] = self.obs["xc"] / method.nondim["nd"]
        self.obs["yc"] = self.obs["yc"] / method.nondim["nd"]
        self.obs["rc"] = self.obs["rc"] / method.nondim["nd"]

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