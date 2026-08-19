import time

import numpy as np
import cvxpy as cp

from trajopt.methods.dev.sqp.reporter import SolveReporter
from trajopt.methods.dev.sqp.scp_trajectory import SCPTrajectory
from trajopt.methods.common import trust_region

class SCPMethod():

    def __init__(self, method_config, trajectory) -> None:

        self.method_config = method_config

        # create scp trajectory
        self.scp_trajectory = SCPTrajectory(trajectory, self.method_config)

        # define the total cost and constraints from all segments for this method
        self.cp_cost = sum(seg.cp_cost for seg in self.scp_trajectory.scp_segments.values())
        self.cp_constraints = [c for s in self.scp_trajectory.scp_segments.values() for c in s.cp_constraints]
        self.cp_subproblem = cp.Problem(cp.Minimize(self.cp_cost), self.cp_constraints)

        total_param_scalars = sum(p.size for p in self.cp_subproblem.parameters())
        self._converged = False

        quiet = bool(self.method_config.flags.get("quiet", False))
        multi = len(self.scp_trajectory.scp_segments) > 1
        self.reporter = SolveReporter(multi=multi, quiet=quiet)
        self.reporter.subproblem_stats(
            num_segments=len(self.scp_trajectory.scp_segments),
            num_params=total_param_scalars,
            num_constraints=len(self.cp_constraints),
            is_dpp=self.cp_subproblem.is_dcp(dpp=True),
        )

    def update_cvxpy_parameters(self) -> None:
        for scp_segment in self.scp_trajectory.scp_segments.values():
            scp_segment.update_cvxpy_parameters()

    def update_current_iter_data(self) -> None:
        parse_time = self.cp_subproblem.compilation_time * 1000.0
        solve_time = self.cp_subproblem.solver_stats.solve_time * 1000.0

        for scp_segment in self.scp_trajectory.scp_segments.values():
            scp_segment.current_iter_data.parse_time = parse_time
            scp_segment.current_iter_data.solve_time = solve_time
            scp_segment.read_solution()

        if getattr(self.method_config.flags, 'line_search', True):
            alpha = trust_region.line_search(self)
        else:
            alpha = 1.0

        for scp_segment in self.scp_trajectory.scp_segments.values():
            scp_segment.cp_subproblem_status = self.cp_subproblem.status
            scp_segment.apply_step(alpha)

        self._converged = all(s.current_iter_data.converged for s in self.scp_trajectory.scp_segments.values())

        for scp_segment in self.scp_trajectory.scp_segments.values():
            scp_segment.update_W_dual(alpha)

        for scp_segment in self.scp_trajectory.scp_segments.values():
            scp_segment.record_iter_data()

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

            try:
                self.cp_subproblem.solve(warm_start=False, **self.method_config.solver_opts)
            except cp.error.SolverError as exc:
                self.reporter.message(f"  subproblem refused ({exc}), tightening trust region")
                for scp_segment in self.scp_trajectory.scp_segments.values():
                    scp_segment.lm_mu = min(max(scp_segment.lm_mu, 1e-6) * 10.0, 1e4)
                continue

            if self.cp_subproblem.status not in {"optimal", "optimal_inaccurate", "user_limit"}:
                reason = f"Terminated from non-optimal convex subproblem! Status: {self.cp_subproblem.status}"
                break

            if not trust_region.step_is_usable(self):
                self.reporter.message(f"  step rejected (status {self.cp_subproblem.status}), tightening trust region")
                for scp_segment in self.scp_trajectory.scp_segments.values():
                    scp_segment.lm_mu = min(scp_segment.lm_mu * 10.0, 1e4)
                continue

            self.update_current_iter_data()
            self.display_status()

            for seg in self.scp_trajectory.scp_segments.values():
                total_discretization_ms += seg.current_iter_data.discretization_time
            total_solve_ms += self.cp_subproblem.solver_stats.solve_time * 1000.0

            if self._converged:
                reason = "Terminated from convergence criteria!"
                break

        ran_iterations = any(s.iter_data_list[-1].iter_num > 0 for s in self.scp_trajectory.scp_segments.values())
        if reason is None and ran_iterations and not self._converged:
            reason = "Terminated from hitting maximum iterations!"

        total_ms = total_discretization_ms + total_solve_ms
        self.reporter.footer(
            reason=reason, total_ms=total_ms,
            disc_ms=total_discretization_ms, solve_ms=total_solve_ms,
        )
        self.reporter.trajectory_summary([
            (s.name, s.current_iter_data.t_start, s.current_iter_data.t_final)
            for s in self.scp_trajectory.scp_segments.values()
        ])

    def display_status(self) -> None:
        multi = len(self.scp_trajectory.scp_segments) > 1
        for scp_segment in self.scp_trajectory.scp_segments.values():
            self.reporter.row(
                scp_segment.current_iter_data,
                segment_name=scp_segment.name if multi else None,
            )