import trajopt.algorithm.initial_guess      as guess
import trajopt.utils.config_loader          as cfg
import trajopt.utils.nondim                 as nondim
import trajopt.algorithm.convergence        as convergence
import trajopt.utils.set_defaults           as defaults

from trajopt.mission import Mission
from trajopt.model import Model
from trajopt.method import Method
import numpy as np

class Problem:
    def __init__(self, example_name):

        config = cfg.load_configs(example_name)

        self.mission = Mission(self, config)
        self.model   = Model(self, config)
        self.method  = Method(self, config)

        self._initialize_problem()
        self.method.get_initial_guess()

        self.case_flag = 1

    def _initialize_problem(self):

        mission = self.mission
        model   = self.model
        method  = self.method

        nondim.set_nondim_params(self)

        #======================
        # Path /NFZ constraints
        #======================

        nd = method.nondim["nd"]
        nt = method.nondim["nt"]
        nf = method.nondim["nf"]

        # no fly zones, specified by position and radius [rad]
        nfz_option = mission.bools["flag_nfz"]
        xc      = mission.nfz_option_list[nfz_option]["xc"] / nd
        yc      = mission.nfz_option_list[nfz_option]["yc"] / nd
        rc      = mission.nfz_option_list[nfz_option]["rc"] / nd

        mission.nfz_idx       = np.arange(0, xc.size)
        mission.n_nfz         = len(mission.nfz_idx)

        obs = {}
        obs["posc"] = np.array([xc, yc]) # xc and yc may be vectors
        obs["rc"]     = rc
        mission.obs = obs

        nondim.set_cost_cnst_nondim_params(self)

        #====================
        # Boundary Conditions
        #====================
        # initial conditions

        # equality initial conditions
        mission.zi            = method.nondim["M"]["state"]["d2nd"] @ mission.zi

        # inequality initial conditions
        # none

        # equality terminal conditions
        mission.zf            = method.nondim["M"]["state"]["d2nd"] @ mission.zf  

        # control boundary conditions
        mission.ui            = -mission.ge * mission.mass / method.nondim["nf"]
        mission.uf            = -mission.ge * mission.mass / method.nondim["nf"]

        #==============================
        # Control and state constraints
        #==============================
        # no state constraints
        mission.z_min         = mission.z_min / nd
        mission.z_min_idx     = mission.z_min_idx
        mission.z_max         = mission.z_max / nd
        mission.z_max_idx     = mission.z_max_idx

        mission.u_norm_min    = mission.custom_input_dict["u_norm_min"] / nf
        mission.u_norm_max    = mission.custom_input_dict["u_norm_max"] / nf
        mission.udot_max      = mission.udot_max / (nf / nt)# [N/s]

        defaults.set_params_constraint_default(self)

        ### Time of flight constraints ###
        Ts_min               = method.T_min / method.nondim["nt"]  # 50
        Ts_max               = method.T_max / method.nondim["nt"]
        method.ddts_max      = method.dT_max / ((method.N - 1) * method.nondim["nt"])  # 0.025
        method.dts_min       = Ts_min / (method.N - 1)
        method.dts_max       = Ts_max / (method.N - 1)

        #============================================
        # Optimization parameters and hyperparameters
        #============================================
        # PTR penalty weights
            # Wtr: weight for trust region cost                        
            # w_term: weight for terminal constraint buffer cost
            # w_path: weight for path constraint buffer cost
            # w_nfz: weight for path constraint buffer cost

        # === Baseline cost + trust region weights ===
        method.weights["w_cost"]         = 0.
        method.weights["eps_nonzero1"]   = 2e-1
        method.weights["eps_nonzero2"]   = 1e-10

        M_state  = method.nondim["M"]["state"]["nd2d"]
        avg_state_nd_sq  = np.mean(np.diag(M_state)**2)

        # === Trust region weights ===
        method.weights.setdefault("alpha_z", 0.5)
        method.weights.setdefault("alpha_u", np.inf)

        method.weights["wtr_z"]          = avg_state_nd_sq  * 1 / (2 * method.weights["alpha_z"])
        method.weights["wtr_u"]          = 0 if np.isinf(method.weights["alpha_u"]) else 1 / (2 * method.weights["alpha_u"])

        # === Autotune modes (flag_autotune ∈ {0,2,3,al-scvx}) ===
        if str(method.bools["flag_autotune"]) in {"0", "2", "3", "al-scvx"}:

            method.weights.setdefault("beta", 1)
            method.weights.setdefault("gamma", 1e-1)

            # --- Buffer weights ---
            if str(method.bools["flag_autotune"]) in {"0", "al-scvx"}:
                if "wbuff" not in method.weights:
                    wbuff = 1e2
                    if str(method.bools["flag_autotune"]) == "0":

                        w_nfz_dim  = wbuff / method.weights["w_fac_N"]
                        w_dyn_dim  = 1e5 * wbuff / method.weights["w_fac_Nm1"]
                        w_term_dim = 1e2 * wbuff

                        # scaled nondim weights to approximately preserve relative scaling between cost terms
                        M_nfz  = method.nondim["M"]["nfz"]["nd2d"]
                        M_dyn  = method.nondim["M"]["dyn"]["nd2d"]
                        M_term = method.nondim["M"]["term"]["nd2d"]

                        avg_nfz_nd_sq  = np.mean(np.diag(M_nfz)**2)
                        avg_dyn_nd_sq  = np.mean(np.diag(M_dyn)**2)
                        avg_term_nd_sq = np.mean(np.diag(M_term)**2)

                        w_nfz   = avg_nfz_nd_sq  * w_nfz_dim
                        w_dyn   = avg_dyn_nd_sq  * w_dyn_dim
                        w_term  = avg_term_nd_sq * w_term_dim
                else:
                    wbuff = method.weights["wbuff"]
                    w_nfz = wbuff / method.weights["w_fac_N"]
                    w_dyn = wbuff / method.weights["w_fac_Nm1"]
                    w_term = wbuff
            else:
                wbuff = 1
                w_nfz = wbuff / method.weights["w_fac_N"]
                w_dyn = wbuff / method.weights["w_fac_Nm1"]
                w_term = wbuff

            method.weights["W_nfz"] += w_nfz

            if method.bools["free_final_time"] or method.bools["ctcs"]:
                buff_dyn = str(method.bools.get("buff_dyn", ""))
                if buff_dyn in {"l1", "l2"}:
                    method.weights["W_dyn"] += w_dyn
                elif buff_dyn in {"quad-1", "quad-2", "quad-3"}:
                    method.weights["W_plus"] += w_dyn
                    method.weights["W_minus"] += w_dyn
                else:
                    method.weights["W_term"] += w_term

        # === Autotune mode: {1,3,al-scvx} ===
        if str(method.bools["flag_autotune"]) in {"1", "3", "al-scvx"}:

            method.weights.setdefault("beta", 1)
            method.weights.setdefault("gamma", 1e-1)

            method.weights["dual_nfz"] += method.weights["eps_nonzero1"]

            if method.bools["free_final_time"]:
                buff_dyn = str(method.bools.get("buff_dyn", ""))
                if buff_dyn == "term":
                    method.weights["dual_term"] += method.weights["eps_nonzero1"]
                else:
                    method.weights["dual_dyn"] += method.weights["eps_nonzero1"]

                    if str(method.bools.get("buff_dyn_dual", "")) == "l1":
                        method.weights["dual_plus"] += method.weights["eps_nonzero1"]
                        method.weights["dual_minus"] += method.weights["eps_nonzero1"]

        ### ctcs convergence adjustments ###
        ctcs_mult_state         = 1e0
        ctcs_mult_cnst          = 1e0 
        eps_ctcs                = 1e-4

        method.conv["ctcs_mult_state"]                  = ctcs_mult_state
        method.conv["ctcs_mult_cnst"]                   = ctcs_mult_cnst

        method.conv["eps_ctcs"]                         = eps_ctcs
        method.weights["w_ctcs"]                        = method.nondim["nd"]**2

        ### State convergence ###
        eps_d_state             = 1e-1  # [m]
        eps_v_state             = 1e-1   # [m/s]
        method.conv["eps_state"]                        = np.concatenate((eps_d_state * np.ones(model.n // 2), 
                                                                        eps_v_state * np.ones(model.n // 2)))

        method.conv.setdefault("state", {})["eps_d"]    = eps_d_state
        method.conv["state"]["eps_v"]                   = eps_v_state

        ### Cost convergence ###
        eps_F_cost              = 1e0 # N

        # Assign to cost eps and store data
        method.conv["eps_cost"] = eps_F_cost
        method.conv.setdefault("cost", {})["eps_v"]     = eps_F_cost

        ### NFZ convergence values ###
        eps_nfz_dim             = 1e-1 # [m]
        rc_dim = mission.obs["rc"] * method.nondim["nd"]

        eps_nfz_cnst            = 2 * rc_dim * eps_nfz_dim - eps_nfz_dim**2
        method.conv["eps_nfz"]                          = eps_nfz_cnst * np.ones(mission.n_nfz)
        method.conv.setdefault("cnst", {})["eps_nfz"]   = eps_nfz_cnst

        ### Terminal constraint values ###
        eps_d_term              = 1e-1 # [m]
        eps_v_term              = 1e-1 # [m/s]

        # Create eps_vector for full terminal state equality, min, max constraints
        eps_term                = np.array([eps_d_term, eps_d_term, eps_d_term, eps_v_term, eps_v_term, eps_v_term])
        eps_term_min            = eps_term.copy()
        eps_term_max            = eps_term.copy()

        # Extract only those terminal constraints used
        method.conv["eps_term"]                         = np.concatenate((eps_term[mission.zf_idx], 
                                                                        eps_term_min[mission.zf_min_idx], 
                                                                        eps_term_max[mission.zf_max_idx]))

        # Store data
        method.conv.setdefault("term", {})["eps_d"]     = eps_d_term

        #### Configure multiple shooting dynamics defect convergence values ###
        method.conv["eps_defect"]                       = np.array([1e-2])

        ### Dynamics convergence ###
        eps_d_dyn               = 1e-1  # [m]
        eps_v_dyn               = 1e-1   # [m/s]
        method.conv["eps_dyn"]                          = np.concatenate((eps_d_dyn * np.ones(model.n // 2), 
                                                                        eps_v_dyn * np.ones(model.n // 2)))

        # Store data
        method.conv.setdefault("dyn", {})["eps_d"]      = eps_d_dyn
        method.conv["dyn"]["eps_v"]                     = eps_v_dyn

        ### Configure generic convergence criterion and max iterations ###
        convergence.set_convergence_tolerance(self)

        # Iterations
        method.conv["iter_max"]  = 20

        # Save variable names
        self.save_var_names    = ["ts_opt", "zs_opt", "us_opt", "params", "O"]