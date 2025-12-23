import numpy as np
from trajopt.core.modules.method    import initial_guess as guess
from trajopt.core.modules.method    import convergence
from trajopt.core.modules.method    import hyperparameters
from trajopt.core.modules.method    import discretize
from trajopt.core.modules.method    import integrators
from trajopt.core.modules.method    import nondim


class Method:

    def __init__(self, problem, config):

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

        self.conv_data   = {}

        nondim.set_nondim_params(problem, self)

        discretize.jit_jax_discretize(problem, self)
        integrators.jit_rk4_jax_dense(problem, self)

        buff_dyn = str(self.flags.get("buff_dyn", "term"))

        # --- Dynamics buffering ---
        if buff_dyn in {"term", "l1", "l2"}:
            self.n_plus_real = 0
            self.n_minus_real = 0
            self.Npm_real = 0
        elif buff_dyn == "quad-1":
            self.n_plus_real = 1
            self.n_minus_real = 1
            self.Npm_real = 1
        elif buff_dyn == "quad-2":
            self.n_plus_real = 1
            self.n_minus_real = 1
            self.Npm_real = self.N - 1
        elif buff_dyn == "quad-3":
            self.n_plus_real = problem.n
            self.n_minus_real = problem.n
            self.Npm_real = 1
        else:
            raise ValueError("Invalid buff_dyn flag.")

        ctcs = str(self.flags.get("ctcs", "none"))

        # --- ctcs buffering ---
        if ctcs in {"term", "l1", "l2", "none"}:
            self.n_plus_ctcs = 0
            self.n_minus_ctcs = 0
            self.Npm_ctcs = 0
        elif ctcs == "quad-1":
            self.n_plus_ctcs = 1
            self.n_minus_ctcs = 1
            self.Npm_ctcs = 1
        elif ctcs == "quad-2":
            self.n_plus_ctcs = 1
            self.n_minus_ctcs = 1
            self.Npm_ctcs = self.N - 1
        elif ctcs == "quad-3":
            self.n_plus_ctcs = problem.n_ctcs
            self.n_minus_ctcs = problem.n_ctcs
            self.Npm_ctcs = 1
        else:
            raise ValueError("Invalid ctcs flag.")

        ### Time of flight constraints ###
        Ts_min        = self.T_min / self.nondim["nt"]
        Ts_max        = self.T_max / self.nondim["nt"]
        self.ddt_max = self.dT_max / ((self.N - 1) * self.nondim["nt"])
        self.dt_min  = Ts_min / (self.N - 1)
        self.dt_max  = Ts_max / (self.N - 1)

        # --- LTV indexing ---
        discretize.set_ltv_indices(problem, self)

        hyperparameters.configure_penalty_weights(problem, self)

        # Extract only those terminal constraints used
        self.conv["eps_term"] = np.concatenate((self.conv["eps_term"][mission.zf_idx], self.conv["eps_term_min"][mission.zf_min_idx], self.conv["eps_term_max"][mission.zf_max_idx]))

        # --- Initialize virtual buffers ---
        self.conv_data["vb_path"] = np.zeros((self.N,   problem.n_path))
        self.conv_data["vb_nfz"]  = np.zeros((self.N,   problem.n_nfz))
        self.conv_data["vb_custom"]  = np.zeros((self.N,   problem.n_custom))
        self.conv_data["vb_dyn"]  = np.zeros((self.N-1, problem.nz))
        self.conv_data["vb_term"] = np.zeros(problem.nz)

        ### Configure generic convergence criterion and max iterations ###
        convergence.set_convergence_tolerance(problem, self)

    def get_initial_guess(self, problem):

        self.nl_guess_u_start = self.nondim["M"]["ctrl"]["d2nd"] @ self.nl_guess_u_start
        self.nl_guess_u_stop  = self.nondim["M"]["ctrl"]["d2nd"] @ self.nl_guess_u_stop

        self.line_guess_u_init = self.line_guess_u_init @ self.nondim["M"]["ctrl"]["d2nd"]

        if self.flags["dynamics_nonconvex"] and (self.flags.get("buff_dyn")=="term"):
            nu_range = np.vstack([self.nl_guess_u_start, self.nl_guess_u_stop])
            guess.nonlinear_initial_guess(nu_range, trajopt_obj)
        else:
            guess.straight_line_initial_guess(problem, self)
            self.nu_init = self.line_guess_u_init

        if problem.constraints.has('ct'):
            guess.ctcs_initial_guess(self)

        self.cost_init = discretize.compute_linearized_costs(self.t_init, self.z_init, self.nu_init, self, problem)[0].sum().item()

        print(f"Cost initial: {self.cost_init}")