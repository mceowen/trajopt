import time

import numpy as np
import cvxpy as cp

from trajopt.methods.dev.scvx.reporter import SolveReporter
from trajopt.methods.common.scp.subproblem import SCPSubproblem
from trajopt.utils.tools import AttrDict

class SCPMethod():
    """SCvx over a single flat trajectory -- one subproblem, no phase splitting."""

    def __init__(self, method_config, trajectory) -> None:

        self.method_config = method_config

        # the whole problem is one flat subproblem
        self.subproblem = SCPSubproblem(trajectory, self.method_config)

        # one-entry dict so analysis/plotting can loop over it like the segments methods
        self.scp_trajectory = AttrDict(scp_subproblems=AttrDict(main=self.subproblem))

        self.cp_cost        = self.subproblem.cp_cost
        self.cp_constraints = self.subproblem.cp_constraints
        self.cp_subproblem  = cp.Problem(cp.Minimize(self.cp_cost), self.cp_constraints)

        total_param_scalars = sum(p.size for p in self.cp_subproblem.parameters())
        self._converged = False

        quiet = bool(self.method_config.flags.get("quiet", False))
        self.reporter = SolveReporter(quiet=quiet)
        self.reporter.subproblem_stats(
            num_params=total_param_scalars,
            num_constraints=len(self.cp_constraints),
            is_dpp=self.cp_subproblem.is_dcp(dpp=True),
        )

    def update_cvxpy_parameters(self) -> None:
        self.subproblem.update_cvxpy_parameters()

    def update_current_iter_data(self) -> None:
        parse_time = self.cp_subproblem.compilation_time * 1000.0
        solve_time = self.cp_subproblem.solver_stats.solve_time * 1000.0

        self.subproblem.current_iter_data.parse_time = parse_time
        self.subproblem.current_iter_data.solve_time = solve_time
        self.subproblem.read_solution()

        self.subproblem.cp_subproblem_status = self.cp_subproblem.status
        self.subproblem.apply_step(alpha=1.0)

        self._converged = self.subproblem.current_iter_data.converged

        self.subproblem.update_constraint_penalties(alpha=1.0)
        self.subproblem.record_iter_data()

    def warmup_jax(self):
        """Run a dummy discretization pass to trigger all JAX JIT compilations."""
        self.reporter.message("Compiling JAX kernels (warmup)...")
        warmup_start = time.perf_counter()
        self.update_cvxpy_parameters()
        warmup_ms = (time.perf_counter() - warmup_start) * 1000.0
        self.reporter.message(f"done ({warmup_ms:.0f} ms)")

    def solve(self, verbose=None):
        if verbose is not None:
            self.reporter.quiet = not verbose

        self.warmup_jax()
        self.reporter.header()

        max_iter = int(self.method_config.flags.iter_max)

        total_discretization_ms = 0.0
        total_solve_ms = 0.0
        reason = None

        for i in range(max_iter + 1):
            self.update_cvxpy_parameters()
            self.cp_subproblem.solve(warm_start=False, **self.method_config.solver_opts)

            if self.cp_subproblem.status not in {"optimal", "optimal_inaccurate", "user_limit"}:
                reason = f"Terminated from non-optimal convex subproblem! Status: {self.cp_subproblem.status}"
                break

            self.update_current_iter_data()
            self.display_status()

            total_discretization_ms += self.subproblem.current_iter_data.discretization_time
            total_solve_ms += self.cp_subproblem.solver_stats.solve_time * 1000.0

            if self._converged:
                reason = "Terminated from convergence criteria!"
                break

        ran_iterations = self.subproblem.iter_data_list[-1].iter_num > 0
        if reason is None and ran_iterations and not self._converged:
            reason = "Terminated from hitting maximum iterations!"

        total_ms = total_discretization_ms + total_solve_ms
        self.reporter.footer(
            reason=reason, total_ms=total_ms,
            disc_ms=total_discretization_ms, solve_ms=total_solve_ms,
        )
        self.reporter.trajectory_summary(
            self.subproblem.current_iter_data.t_start, self.subproblem.current_iter_data.t_final,
        )

    def display_status(self) -> None:
        self.reporter.row(self.subproblem.current_iter_data)
