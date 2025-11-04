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
        self.bools             = mission_config["bools"]
        self.nfz_option_list   = mission_config["nfz_option_list"]
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
        self.nfz_idx           = mission_config["nfz_idx"]
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
        self.path_lim          = mission_config["path_lim"]
        self.path_idx          = mission_config["path_idx"]
        self.aux_idx           = mission_config["aux_idx"]
        self.custom_input_dict = mission_config["custom_input_dict"]

        self.planet            = mission_config["planet"]
        self.vehicle           = mission_config["vehicle"]

        self.obs = {}

        # ===============================================================
        # point to module and corresponding methods based on configs
        # ===============================================================

        self.mission_module = importlib.import_module(f"trajopt.core.modules.missions.{self.mission_name}")

        # set cost/constraint nondim setter function (needed for nondim initialization)
        self._get_cost_cnstr_nondim = self.mission_module.get_cost_cnstr_nondim

    # ===============================================================
    # member functions point to selected fcns from selected module
    # ===============================================================
    
    def cost(self, ts, zs, us):
        return self._cost(ts, zs, us, self.problem)

    def lin_cost(self, ts, zs, us):
        return self._lin_cost(ts, zs, us, self.problem)
    
    def nonlinear_aero(self, ts, zs, us):
        return self._nonlinear_aero(ts, zs, us, self.problem)

    def custom_inputs(self, problem, local_vars):
        return self._custom_inputs(problem, local_vars)

    def custom_variables(self, problem, local_vars):
        return self._custom_variables(problem, local_vars)

    def custom_constraints(self, problem, local_vars):
        return self._custom_constraints(problem, local_vars)

    def custom_cost(self, problem, local_vars):
        return self._custom_cost(problem, local_vars)
    
    def get_cost_cnstr_nondim(self):
        return self._get_cost_cnstr_nondim(self.problem)
        
    def set_custom_params(self):
        return self._set_custom_params(self.problem)

    def update_mission_params(self):
        method = self.problem.method

        # ===============================================================
        # function setup
        # ===============================================================

        # set cost function
        self._cost = self.mission_module.cost
        self._analytical_cost = self.mission_module.analytical_cost

        # set linearized cost function
        if method.bools["auto_jac"]:
            self._lin_cost = convexify.generate_jacobians(self._cost)
        else:
            self._lin_cost = self._analytical_cost

        if self.bools["aero_type"] != 'none':
            if method.bools["jax_dyn"]:
                self._nonlinear_aero = self.mission_module.nonlinear_aero_jax
            else:
                self._nonlinear_aero = self.mission_module.nonlinear_aero
        else:
            self._nonlinear_aero = None

        # set custom inputs
        self._set_custom_params = self.mission_module.set_custom_params
        self._custom_inputs = self.mission_module.custom_inputs
        self._custom_variables = self.mission_module.custom_variables
        self._custom_constraints = self.mission_module.custom_constraints
        self._custom_cost = self.mission_module.custom_cost

        # ===============================================================
        # nondim parameters
        # ===============================================================
        # TODO (carlos): this only contains things necessary for quadrotor example
        # need to add setup for all non-custom params soon

        self.M_zi = method.nondim["M"]["state"]["d2nd"][np.ix_(self.zi_idx, self.zi_idx)]
        self.M_zf = method.nondim["M"]["state"]["d2nd"][np.ix_(self.zf_idx, self.zf_idx)]

        self.zi = self.M_zi @ self.zi
        self.zf = self.M_zf @ self.zf  

        M_z_min = method.nondim["M"]["state"]["d2nd"][np.ix_(self.z_min_idx, self.z_min_idx)]
        M_z_max = method.nondim["M"]["state"]["d2nd"][np.ix_(self.z_max_idx, self.z_max_idx)]
        
        self.z_min = M_z_min @ self.z_min
        self.z_max = M_z_max @ self.z_max

        if self.bools["init_ctrl"] == 1:
            self.ui = method.nondim["M"]["ctrl"]["d2nd"] @ self.ui
            self.uf = method.nondim["M"]["ctrl"]["d2nd"] @ self.uf

        M_u_min = method.nondim["M"]["ctrl"]["d2nd"][np.ix_(self.u_min_idx, self.u_min_idx)]
        M_u_max = method.nondim["M"]["ctrl"]["d2nd"][np.ix_(self.u_max_idx, self.u_max_idx)]
        
        self.u_min = M_u_min @ self.u_min
        self.u_max = M_u_max @ self.u_max
        
        M_udot_max = method.nondim["M"]["ctrl"]["d2nd"][np.ix_(self.udot_max_idx, self.udot_max_idx)]
        self.udot_max = M_udot_max @ self.udot_max * method.nondim["nt"]

        self.obs["posc"] = self.obs["posc"] / method.nondim["nd"]
        self.obs["rc"]   = self.obs["rc"]   / method.nondim["nd"]

        # ===============================================================
        # constraint bookkeeping
        # ===============================================================
        self.n_init       = len(self.zi_idx)
        self.n_init_ineq  = len(self.zi_min_idx) + len(self.zi_max_idx)
        self.n_term       = len(self.zf_idx)
        self.n_term_ineq  = len(self.zf_min_idx) + len(self.zf_max_idx)
        self.n_ctrl       = len(self.u_min_idx)  + len(self.u_max_idx)
        self.n_state      = len(self.z_min_idx)  + len(self.z_max_idx)
        self.n_udot       = len(self.udot_max_idx)
        self.n_path       = len(self.path_idx)
        self.n_nfz        = len(self.nfz_idx)
        self.n_aux        = len(self.aux_idx)
        self.n_init_ctrl  = len(self.ui_idx)
        self.n_term_ctrl  = len(self.uf_idx)
        self.n_ineq       = self.n_path + self.n_nfz + self.n_aux

        self.set_custom_params()

    # TODO: maybe this can be cleaner
    def initialize_nfz(self):
        # extracts nfz_idx which is neces
        nfz_option     = self.bools["flag_nfz"]
        xc             = self.nfz_option_list[nfz_option]["xc"]
        yc             = self.nfz_option_list[nfz_option]["yc"]

        self.nfz_idx = np.arange(0, xc.size)
        self.n_nfz   = len(self.nfz_idx)

        # initializes obs dictionary with dimensional values
        # will be nondimmed in update_mission_params()
        self.obs["rc"] = self.nfz_option_list[nfz_option]["rc"]
        self.obs["posc"] = np.array([xc, yc])