import numpy as np
from trajopt.core.modules.method    import initial_guess as guess
from trajopt.core.modules.method    import convergence
from trajopt.core.modules.method    import hyperparameters
from trajopt.core.modules.method    import discretize


class Method:

    def __init__(self, problem, config):

        self.problem = problem

        # ===============================================================
        # load config params
        # ===============================================================

        method_config    = config["method"]
        self.N           = method_config["N"]
        self.flags       = method_config['flags']
        self.solver_opts = method_config["solver_opts"]
        self.T_init      = method_config["T_init"]
        self.T_min       = method_config["T_min"]
        self.T_max       = method_config["T_max"]
        self.dT_max      = method_config["dT_max"]

        self.conv        = method_config["conv"]
        self.weights     = method_config["weights"]

        self.nl_guess_u_start = method_config["nl_guess_u_start"]
        self.nl_guess_u_stop  = method_config["nl_guess_u_stop"]

        self.line_guess_u_init = method_config["line_guess_u_init"]

        self.nondim      = {}
        self.conv_data   = {}

        # ===============================================================
        # point to module and corresponding methods based on configs
        # ===============================================================

        # TODO:
        # will probably point to discretize, convergence, subprob functions etc

    def get_initial_guess(self):
        problem = self.problem
        mission = problem.mission
        model   = problem.model

        self.nl_guess_u_start = self.nondim["M"]["ctrl"]["d2nd"] @ self.nl_guess_u_start
        self.nl_guess_u_stop  = self.nondim["M"]["ctrl"]["d2nd"] @ self.nl_guess_u_stop

        self.line_guess_u_init = self.line_guess_u_init @ self.nondim["M"]["ctrl"]["d2nd"]

        if self.flags["free_final_time"] and (self.flags.get("buff_dyn")=="term"):
            nu_range = np.vstack([self.nl_guess_u_start, self.nl_guess_u_stop])
            guess.nonlinear_initial_guess(nu_range, problem)
        else:
            guess.straight_line_initial_guess(problem)
            self.nu_init = self.line_guess_u_init

        if self.flags["ctcs"] != "none":
            guess.ctcs_initial_guess(problem)

        self.cost_init = discretize.compute_linearized_costs(self.t_init, self.z_init, self.nu_init, problem)[0].sum().item()

    def update_method_params(self):
        problem = self.problem
        mission = problem.mission
        model = problem.model

        # precompile discretize functions for jax
        if self.flags['jax_dyn'] == 1:
            discretize.jit_jax_discretize(problem)

        buff_dyn = str(self.flags.get("buff_dyn", "term"))

        # --- CTCS-specific adjustment ---
        if self.flags["ctcs"]!="none" and buff_dyn == "term":
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
        self.ddt_max = self.dT_max / ((self.N - 1) * self.nondim["nt"])
        self.dt_min  = Ts_min / (self.N - 1)
        self.dt_max  = Ts_max / (self.N - 1)

        # --- Terminal nondimensionalization matrix ---
        M_state_vec = np.diag(self.nondim["M"]["state"]["d2nd"])
        zf_idx      = mission.zf_idx
        zf_min_idx  = mission.zf_min_idx
        zf_max_idx  = mission.zf_max_idx
        M_term_diag = np.concatenate([M_state_vec[zf_idx], M_state_vec[zf_min_idx], M_state_vec[zf_max_idx]])
        self.nondim["M"]["term"]["d2nd"] = np.diag(M_term_diag)

        # --- LTV indexing ---
        discretize.set_ltv_indices(problem)

        # for reference
        #self.set_weights()
        hyperparameters.configure_penalty_weights(problem)

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
        self.conv_data["vb_custom"]  = np.zeros((self.N,   mission.n_custom))
        self.conv_data["vb_dyn"]  = np.zeros((self.N-1, model.nz))
        self.conv_data["vb_term"] = np.zeros(model.nz)