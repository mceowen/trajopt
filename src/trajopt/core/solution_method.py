import numpy as np
from trajopt.library.methods    import initial_guess as guess
from trajopt.library.methods    import convergence
from trajopt.library.methods    import hyperparameters
from trajopt.library.methods    import discretize
from trajopt.library.methods    import integrators
from trajopt.core.indexing.index_map import IndexMap
from trajopt.core.scaling.nondim import Nondim

class SolutionMethod:

    def __init__(self, problem, config):

        # ===============================================================
        # load config params
        # ===============================================================

        self.problem = problem

        method_config    = config
        self.N           = method_config["N"]
        self.flags       = method_config['flags']
        self.guess       = method_config["guess"]
        self.conv        = method_config["conv"]
        self.weights     = method_config["weights"]
        self.solver_opts = method_config["solver_opts"]
        self.conv_data   = {}

        self.index_map = IndexMap(self)
        self.nondim = Nondim(problem)

        self.problem.constraints.nondim_constraints(self.nondim)
        self.problem.constraints.convexify_constraints()
        self.problem.constraints.augment_ctcs_dynamics(self.problem.n)
        self.problem.costs.nondim_costs(self.nondim)
        self.problem.costs.convexify_costs()

        # ---- Time grid initialization ----
        self.dt_init  = (self.guess["T_init"] / (self.N - 1)) * np.ones(self.N - 1)
        self.Ts_init  = self.guess["T_init"] / self.nondim.nt
        self.dt_init  = self.dt_init / self.nondim.nt

        discretize.compile_jax_discretization(problem, self)
        integrators.compile_dense_jax_propagator(problem, self, problem.params)

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
        Ts_min       = self.guess["T_min"] / self.nondim.nt
        Ts_max       = self.guess["T_max"] / self.nondim.nt
        self.ddt_max = self.guess["dT_max"] / ((self.N - 1) * self.nondim.nt)
        self.dt_min  = Ts_min / (self.N - 1)
        self.dt_max  = Ts_max / (self.N - 1)

        # --- LTV indexing ---
        discretize.set_ltv_indices(problem, self)
        hyperparameters.configure_penalty_weights(problem, self)

        # Extract only those terminal constraints used
        term_eq_constraints   = problem.constraints.get(ct=0, type='equality_bc', boundary="final", set="state")
        term_ineq_constraints = problem.constraints.get(ct=0, type='inequality_bc', boundary="final", set="state")
        term_ctcs_constraints = problem.constraints.get(ct=1)

        term_constraints = term_eq_constraints + term_ineq_constraints + term_ctcs_constraints

        self.conv["eps_term"] = np.concatenate([constraint.eps for constraint in term_constraints])

        # --- Initialize virtual buffers ---
        self.conv_data["vb_path"] = np.zeros((self.N,   problem.n_path))
        self.conv_data["vb_nfz"]  = np.zeros((self.N,   problem.n_nfz))
        self.conv_data["vb_custom"]  = np.zeros((self.N,   problem.n_custom))
        self.conv_data["vb_dyn"]  = np.zeros((self.N-1, problem.nz))
        self.conv_data["vb_term"] = np.zeros(problem.nz)

        ### Configure generic convergence criterion and max iterations ###
        convergence.set_convergence_tolerance(problem, self)

        self.get_initial_guess(problem)

    def get_initial_guess(self, problem):

        if (self.guess.get("type", "propagation") == "propagation") or self.flags.get("buff_dyn") == "term":
            self.t_init, self.z_init, self.nu_init = guess.nonlinear_initial_guess(problem, self)
        else:
            self.t_init, self.z_init, self.nu_init = guess.straight_line_initial_guess(problem, self)

        if problem.constraints.has(ct=1):
            self.z_init = guess.ctcs_initial_guess(problem, self)

        self.cost_init = discretize.compute_linearized_costs(self.t_init, self.z_init, self.nu_init, problem, self)[0].sum().item()

        print(f"Cost initial: {self.cost_init}")