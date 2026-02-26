import numpy as np
from trajopt.library.methods    import initial_guess as guess
from trajopt.library.methods    import convergence
from trajopt.library.methods    import hyperparameters
from trajopt.library.methods    import discretize
from trajopt.library.methods    import integrators
from trajopt.core.indexing.index_map import IndexMap
from trajopt.core.scaling.nondim import Nondim

from trajopt.utils.tools import AttrDict, recursive_attrdict

class SolutionMethod:

    def __init__(self, problem, config, index_map=None):

        # ===============================================================
        # load config params
        # ===============================================================
        method_config    = config

        self.problem    = problem
        self.index_map  = index_map if index_map is not None else problem.index_map

        self.flags       = recursive_attrdict(method_config['flags'])
        self.guess       = recursive_attrdict(method_config["guess"])
        self.conv        = recursive_attrdict(method_config["conv"])
        self.penalty     = recursive_attrdict(method_config["weights"])
        self.solver_opts = recursive_attrdict(method_config["solver_opts"])

        self.conv_data   = AttrDict()

        # update index_map
        self.index_map.update_index_map(problem=self.problem, method=self)


        # Use the same index_map as problem, but update with method config
        self.nondim = Nondim(problem)

        # nondimensionalize and convexfiy constraints
        self.problem.constraints.nondim_constraints(self.nondim)
        self.problem.constraints.convexify_constraints()

        # nondimensionalize and convexify costs
        self.problem.costs.nondim_costs(self.nondim)
        self.problem.costs.convexify_costs()

        # augment dynamics with ct constraints and costs
        self.problem.constraints.augment_ctcs_dynamics(self.index_map.n['state'])

        # ---- Time grid initialization ----
        self.dt_init  = (self.guess.T_init / (self.index_map.N.N - 1)) * np.ones(self.index_map.N.N - 1)
        self.Ts_init  = self.guess.T_init / self.nondim.nt
        self.dt_init  = self.dt_init / self.nondim.nt

        discretize.compile_jax_discretization(problem, self)
        # discretize.compile_jax_discretization_bwd(problem, self)
        integrators.compile_dense_jax_propagator(problem, self, problem.params)

        ### Time of flight constraints ###
        Ts_min       = self.guess.T_min / self.nondim.nt
        Ts_max       = self.guess.T_max / self.nondim.nt
        self.ddt_max = self.guess.dT_max / ((self.index_map.N.N - 1) * self.nondim.nt)
        self.dt_min  = Ts_min / (self.index_map.N.N - 1)
        self.dt_max  = Ts_max / (self.index_map.N.N - 1)

        # --- LTV indexing ---
        discretize.set_ltv_indices(problem, self)
        hyperparameters.configure_penalty_weights(problem, self)

        # Extract only those terminal constraints used
        term_eq_constraints   = problem.constraints.get(ct=0, type='equality_bc', boundary="final", set="state")
        term_ineq_constraints = problem.constraints.get(ct=0, type='inequality_bc', boundary="final", set="state")
        term_ctcs_constraints = problem.constraints.get(ct=1)

        term_constraints = term_eq_constraints + term_ineq_constraints + term_ctcs_constraints

        self.conv.eps_term = np.concatenate([constraint.eps for constraint in term_constraints])

        # --- Initialize virtual buffers ---
        self.conv_data.vb_path = np.zeros((self.index_map.N.N,      problem.index_map.n.path))
        self.conv_data.vb_nfz  = np.zeros((self.index_map.N.N,      problem.index_map.n.nfz))
        self.conv_data.vb_custom  = np.zeros((self.index_map.N.N,   problem.index_map.n.custom))
        self.conv_data.vb_dyn  = np.zeros((self.index_map.N.N-1,    problem.index_map.n.z))
        self.conv_data.vb_terminal = np.zeros(problem.index_map.n.z)

        ### Configure generic convergence criterion and max iterations ###
        convergence.set_convergence_tolerance(problem, self)

        self.get_initial_guess(problem)

        # TODO(Skye/Carlos): Potentially move the method=specific constraint modeling here 
        # (instead of the subproblem)

    def get_initial_guess(self, problem):

        if (getattr(self.guess, "type", "propagation") == "propagation") or self.flags.buff_dyn == "term":
            self.t_init, self.z_init, self.nu_init = guess.nonlinear_initial_guess(problem, self)
        else:
            self.t_init, self.z_init, self.nu_init = guess.straight_line_initial_guess(problem, self)

        if problem.constraints.has(ct=1):
            self.z_init = guess.ctcs_initial_guess(problem, self)

        self.cost_init = discretize.compute_nonconvex_costs(self.t_init, self.z_init, self.nu_init, problem, self)

        # print(f"Cost initial: {self.cost_init}")