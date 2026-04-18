import numpy as np
from trajopt.library.methods.scp    import initial_guess as guess
from trajopt.library.methods.scp    import convergence
from trajopt.library.methods.scp    import hyperparameters
from trajopt.library.methods.scp    import discretize
from trajopt.library.methods.scp    import integrators
from trajopt.library.methods.scp.subproblem import Subproblem
from trajopt.core.scaling.nondim import Nondim

from trajopt.utils.tools import AttrDict, recursive_attrdict

class SolutionMethod:

    def __init__(self, problem, config, index_map=None):

        # ===============================================================
        # load config params
        # ===============================================================
        self.problem    = problem
        self.index_map  = index_map if index_map is not None else problem.index_map

        self.flags       = recursive_attrdict(config.method.flags)
        self.initial_guess = AttrDict(recursive_attrdict(config.method.guess))
        self.conv        = recursive_attrdict(config.method.conv)
        self.penalty     = recursive_attrdict(config.method.weights)
        self.solver_opts = recursive_attrdict(config.method.solver_opts)

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

        # ---- Time grid initialization ----
        self.Ts_init  = self.initial_guess.T_init / self.nondim.time_scale
        t_init = np.linspace(0.0, self.Ts_init, self.index_map.N.time_grid).reshape(-1, 1)
        dt_init = np.diff(t_init, axis=0)
        self.initial_guess.t = t_init.reshape(-1)
        self.initial_guess.dt = dt_init
        self.initial_guess.x = None
        self.initial_guess.u = None
        self.initial_guess.z = None
        self.initial_guess.nu = None

        discretize.compile_jax_discretization(problem, self)
        # discretize.compile_jax_discretization_bwd(problem, self)
        integrators.compile_dense_jax_propagator(problem, self, problem.params)
        # TODO(SKYE): Verify compilation below
        integrators.compile_dense_jax_propagator(
            problem,
            self,
            problem.params,
            dynamics=problem.constraints.get(type="dynamics")[0].fcn_base,
            compiled_attr_name="propagate_rk4_physical_jit",
        )
        integrators.compile_tau_propagator(problem, self)

        ### Time of flight constraints ###
        self.Ts_min       = self.initial_guess.T_min / self.nondim.time_scale
        self.Ts_max       = self.initial_guess.T_max / self.nondim.time_scale
        self.ddt_max = self.initial_guess.dT_max / ((self.index_map.N.time_grid - 1) * self.nondim.time_scale)
        self.dt_min  = self.Ts_min / (self.index_map.N.time_grid - 1)
        self.dt_max  = self.Ts_max / (self.index_map.N.time_grid - 1)

        # --- LTV indexing ---
        discretize.set_ltv_indices(problem, self)
        hyperparameters.configure_penalty_weights(problem, self)

        ### Configure generic convergence criterion and max iterations ###
        convergence.set_convergence_tolerance(problem, self)

        guess.set_initial_guess(problem,self)

        # TODO(Skye/Carlos): Potentially move the method=specific constraint modeling here 
        # (instead of the subproblem)
        # OR maybe just initialize the subproblem itself here and only run solve later?
        # --- Initialize virtual buffers ---
        self.conv_data.vb_ineq     = np.zeros((self.index_map.N.time_grid,      self.index_map.n.nonconvex_inequality))
        self.conv_data.vb_dyn      = np.zeros((self.index_map.N.time_grid-1,    self.index_map.n.z))
        self.conv_data.vb_terminal = np.zeros(self.index_map.n.z)

        # --- Initialize reusable compiled Subproblem (DPP) ---
        self.subproblem = Subproblem(problem, self)

