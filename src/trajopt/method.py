import numpy as np
import trajopt.algorithm.initial_guess      as guess
import trajopt.algorithm.convergence        as convergence
import trajopt.algorithm.discretization     as discretize

class Method:

    def __init__(self, problem, config):

        self.problem = problem

        # ===============================================================
        # load config params
        # ===============================================================

        method_config    = config["method"]
        self.method_name = method_config["method_name"]
        self.N           = method_config["N"]
        self.bools       = method_config["bools"]
        self.solver_opts = method_config["solver_opts"]
        self.T_init      = method_config["T_init"]
        self.T_min       = method_config["T_min"]
        self.T_max       = method_config["T_max"]
        self.dT_max      = method_config["dT_max"]

        self.conv        = method_config["conv"]
        self.weights     = method_config["weights"]

        self.nondim      = {}
        self.conv_data   = {}

        # ===============================================================
        # point to module and corresponding methods based on configs
        # ===============================================================

        # TODO:
        # will probably point to discretization, convergence, subprob functions etc

    def get_initial_guess(self):
        problem = self.problem
        mission = problem.mission
        model   = problem.model

        nl_guess_us_range, line_guess_us_init = model.get_initial_guess_control()

        if self.bools["free_final_time"] and (self.bools.get("buff_dyn")=="term"):
            us_range = nl_guess_us_range
            guess.nonlinear_initial_guess(us_range, problem)
        else:
            guess.straight_line_initial_guess(problem)
            self.us_init = line_guess_us_init

        if self.bools["ctcs"]:
            guess.ctcs_initial_guess(problem)

        self.cost_init = mission.cost(self.ts_init, self.zs_init, self.us_init)

    def update_method_params(self):
        problem = self.problem
        mission = problem.mission
        model = problem.model

        buff_dyn = str(self.bools.get("buff_dyn", "term"))

        # --- CTCS-specific adjustment ---
        if self.bools.get("ctcs") and buff_dyn == "term":
            buff_dyn = "l1"

        # --- Dynamics buffering ---
        if buff_dyn in {"term", "l1", "l2"}:
            self.n_plus = 0
            self.n_minus = 0
            self.Npm = 0
        elif buff_dyn == "quad-1":
            self.n_plus = 1
            self.n_minus = 1
            self.Npm = 1
        elif buff_dyn == "quad-2":
            self.n_plus = 1
            self.n_minus = 1
            self.Npm = self.N - 1
        elif buff_dyn == "quad-3":
            self.n_plus = model.nz
            self.n_minus = model.nz
            self.Npm = 1
        else:
            raise ValueError("Invalid buff_dyn flag.")

        ### Time of flight constraints ###
        Ts_min        = self.T_min / self.nondim["nt"]
        Ts_max        = self.T_max / self.nondim["nt"]
        self.ddts_max = self.dT_max / ((self.N - 1) * self.nondim["nt"])
        self.dts_min  = Ts_min / (self.N - 1)
        self.dts_max  = Ts_max / (self.N - 1)

        # --- Terminal nondimensionalization matrix ---
        M_state_vec = np.diag(self.nondim["M"]["state"]["d2nd"])
        zf_idx      = mission.zf_idx
        zf_min_idx  = mission.zf_min_idx
        zf_max_idx  = mission.zf_max_idx
        M_term_diag = np.concatenate([M_state_vec[zf_idx], M_state_vec[zf_min_idx], M_state_vec[zf_max_idx]])
        self.nondim["M"]["term"]["d2nd"] = np.diag(M_term_diag)

        # --- LTV indexing ---
        discretize.set_ltv_indices(problem)

        self.set_weights()

        ### NFZ convergence values ###
        rc_dim = mission.obs["rc"] * self.nondim["nd"]

        eps_nfz_cnst = 2 * rc_dim * self.conv["eps_nfz"] - self.conv["eps_nfz"]**2
        self.conv["eps_nfz"] = eps_nfz_cnst * np.ones(mission.n_nfz)

        # Extract only those terminal constraints used
        self.conv["eps_term"] = np.concatenate((self.conv["eps_term"][mission.zf_idx], self.conv["eps_term_min"][mission.zf_min_idx], self.conv["eps_term_max"][mission.zf_max_idx]))

        ### Configure generic convergence criterion and max iterations ###
        convergence.set_convergence_tolerance(problem)

        # --- Initialize virtual buffers ---
        self.conv_data["vb_path"] = np.zeros((self.N,   mission.n_path))
        self.conv_data["vb_nfz"]  = np.zeros((self.N,   mission.n_nfz))
        self.conv_data["vb_aux"]  = np.zeros((self.N,   mission.n_aux))
        self.conv_data["vb_dyn"]  = np.zeros((self.N-1, model.nz))
        self.conv_data["vb_term"] = np.zeros(model.nz)

    def set_weights(self):
        problem = self.problem
        mission = problem.mission

        # --- Default weights ---
        self.weights["dual_path"]    = np.zeros((self.N, mission.n_path))
        self.weights["dual_nfz"]     = np.zeros((self.N, mission.n_nfz))
        self.weights["dual_aux"]     = np.zeros((self.N, mission.n_aux))
        self.weights["dual_term"]    = np.zeros(mission.n_term + mission.n_term_ineq)
        self.weights["dual_dyn"]     = np.zeros((self.N - 1, mission.n_dyn))
        self.weights["dual_plus"]    = np.zeros((self.N - 1, mission.n_dyn))
        self.weights["dual_minus"]   = np.zeros((self.N - 1, mission.n_dyn))

        self.weights["W_path"]       = np.zeros((self.N, mission.n_path))
        self.weights["W_nfz"]        = np.zeros((self.N, mission.n_nfz))
        self.weights["W_aux"]        = np.zeros((self.N, mission.n_aux))
        self.weights["W_term"]       = np.zeros(mission.n_term + mission.n_term_ineq)
        self.weights["W_dyn"]        = np.zeros((self.N - 1, mission.n_dyn))
        self.weights["W_plus"]       = np.zeros((self.Npm, self.n_plus))
        self.weights["W_minus"]      = np.zeros((self.Npm, self.n_minus))

        # PTR penalty weights
            # Wtr: weight for trust region cost                        
            # w_term: weight for terminal constraint buffer cost
            # w_path: weight for path constraint buffer cost
            # w_nfz: weight for path constraint buffer cost

        M_state  = self.nondim["M"]["state"]["nd2d"]
        avg_state_nd_sq = np.mean(np.diag(M_state)**2)

        self.weights["wtr_z"] = avg_state_nd_sq  * 1 / (2 * self.weights["alpha_z"])
        self.weights["wtr_u"] = 0 if np.isinf(self.weights["alpha_u"]) else 1 / (2 * self.weights["alpha_u"])

        self.weights["w_fac_N"]      = self.N
        self.weights["w_fac_Nm1"]    = self.N - 1

        # === Autotune modes (flag_autotune ∈ {0,2,3,al-scvx}) ===
        if str(self.bools["flag_autotune"]) in {"0", "2", "3", "al-scvx"}:

            self.weights.setdefault("beta", 1)
            self.weights.setdefault("gamma", 1e-1)

            # --- Buffer weights ---
            if str(self.bools["flag_autotune"]) in {"0", "al-scvx"}:
                if "wbuff" not in self.weights:
                    wbuff = 1e2
                    if str(self.bools["flag_autotune"]) == "0":

                        w_nfz_dim  = wbuff / self.weights["w_fac_N"]
                        w_dyn_dim  = 1e5 * wbuff / self.weights["w_fac_Nm1"]
                        w_term_dim = 1e2 * wbuff

                        # scaled nondim weights to approximately preserve relative scaling between cost terms
                        M_nfz  = self.nondim["M"]["nfz"]["nd2d"]
                        M_dyn  = self.nondim["M"]["dyn"]["nd2d"]
                        M_term = self.nondim["M"]["term"]["nd2d"]

                        avg_nfz_nd_sq  = np.mean(np.diag(M_nfz)**2)
                        avg_dyn_nd_sq  = np.mean(np.diag(M_dyn)**2)
                        avg_term_nd_sq = np.mean(np.diag(M_term)**2)

                        w_nfz   = avg_nfz_nd_sq  * w_nfz_dim
                        w_dyn   = avg_dyn_nd_sq  * w_dyn_dim
                        w_term  = avg_term_nd_sq * w_term_dim
                else:
                    wbuff = self.weights["wbuff"]
                    w_nfz = wbuff / self.weights["w_fac_N"]
                    w_dyn = wbuff / self.weights["w_fac_Nm1"]
                    w_term = wbuff
            else:
                wbuff = 1
                w_nfz = wbuff / self.weights["w_fac_N"]
                w_dyn = wbuff / self.weights["w_fac_Nm1"]
                w_term = wbuff

            self.weights["W_nfz"] += w_nfz

            if self.bools["free_final_time"] or self.bools["ctcs"]:
                buff_dyn = str(self.bools.get("buff_dyn", ""))
                if buff_dyn in {"l1", "l2"}:
                    self.weights["W_dyn"] += w_dyn
                elif buff_dyn in {"quad-1", "quad-2", "quad-3"}:
                    self.weights["W_plus"] += w_dyn
                    self.weights["W_minus"] += w_dyn
                else:
                    self.weights["W_term"] += w_term

        # === Autotune mode: {1,3,al-scvx} ===
        if str(self.bools["flag_autotune"]) in {"1", "3", "al-scvx"}:

            self.weights.setdefault("beta", 1)
            self.weights.setdefault("gamma", 1e-1)

            self.weights["dual_nfz"] += self.weights["eps_nonzero1"]

            if self.bools["free_final_time"]:
                buff_dyn = str(self.bools.get("buff_dyn", ""))
                if buff_dyn == "term":
                    self.weights["dual_term"] += self.weights["eps_nonzero1"]
                else:
                    self.weights["dual_dyn"] += self.weights["eps_nonzero1"]

                    if str(self.bools.get("buff_dyn_dual", "")) == "l1":
                        self.weights["dual_plus"] += self.weights["eps_nonzero1"]
                        self.weights["dual_minus"] += self.weights["eps_nonzero1"]

        ### ctcs convergence adjustments ###
        # TODO: will probably need to change this weight later (shouldn't be tied to nondim["nd"])
        self.weights["w_ctcs"] = self.nondim["nd"]**2