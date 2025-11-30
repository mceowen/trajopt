import numpy as np
import importlib

import trajopt.utils.tools as tools
import trajopt.core.modules.method.initial_guess as guess
import trajopt.core.modules.method.convergence as convergence
import trajopt.core.modules.method.convexify as convexify
import trajopt.utils.nondim as nondim
import jax.numpy as jnp

class Mission:
    def __init__(self, problem, config):

        self.problem = problem

        # ===============================================================
        # load dimensional config parameters
        # ===============================================================
        
        mission_config         = config["mission"]
        self.name              = mission_config["module_path"].split(".")[-1]
        self.flags             = mission_config['flags']
        
        # standard constraint parameters
        self.zi                = mission_config["zi"]
        self.zi_idx            = mission_config["zi_idx"]
        self.zi_guess          = mission_config["zi_guess"]
        self.zf                = mission_config["zf"]
        self.zf_idx            = mission_config["zf_idx"]
        self.zf_guess          = mission_config["zf_guess"]
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
        self.custom_limits        = mission_config["custom_limits"]
        self.custom_input_dict = mission_config["custom_input_dict"]
        self.convex_limits     = mission_config["convex_limits"]

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
        self.n_custom     = sum(1 if np.isscalar(v) else len(v) for v in self.custom_limits.values())
        self.n_ineq       = self.n_path + self.n_nfz + self.n_custom

        self.path_idx     = np.arange(self.n_path)
        self.nfz_idx      = np.arange(self.n_path, self.n_path + self.n_nfz)
        self.custom_idx      = np.arange(self.n_path + self.n_nfz, self.n_ineq)

        # ===============================================================
        # point to module and corresponding methods based on configs
        # ===============================================================
        self.mission_module = importlib.import_module(mission_config["module_path"])

        # set cost/constraint nondim setter function (needed for nondim initialization)
        self._get_cost_cnstr_nondim = self.mission_module.get_cost_cnstr_nondim

    # ===============================================================
    # member functions point to selected fcns from selected module
    # ===============================================================
    
    def nonlinear_aero(self, t, z, nu):
        return self._nonlinear_aero(t, z, nu, self.problem)

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
        """
        Attach mission functions for costs and constraints.
        Uses custom_modules from YAML config if specified, otherwise uses base mission.
        """
        model = self.problem.model
        method = self.problem.method
        problem = self.problem

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
        # Load CTCS terminal constraints if applicable
        self.n_term_ctcs  = (model.n_ctcs if method.flags['ctcs']=="term" else 0)

        # nondimensionalization
        self.M_zi = method.nondim["M"]["state"]["d2nd"][np.ix_(self.zi_idx, self.zi_idx)]
        self.M_zf = method.nondim["M"]["state"]["d2nd"][np.ix_(self.zf_idx, self.zf_idx)]
        self.zi = self.M_zi @ self.zi
        self.zf = self.M_zf @ self.zf

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