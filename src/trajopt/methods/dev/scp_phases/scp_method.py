import time

import cvxpy as cp

from trajopt.methods.common.scp.reporter_phases import SolveReporter
from trajopt.methods.common.scp.trajectory import SCPTrajectory
from trajopt.methods.common import trust_region

class SCPMethod():
    """SCP over a multi-phase trajectory.
    """

    def __init__(self, method_config, trajectory) -> None:

        self.method_config = method_config

        # create scp trajectory
        self.scp_trajectory = SCPTrajectory(trajectory, self.method_config)

        # define the total cost and constraints from all phases for this method
        self.cp_cost        = sum(seg.cp_cost for seg in self.scp_trajectory.scp_phases.values())
        self.cp_constraints = [c for s in self.scp_trajectory.scp_phases.values() for c in s.cp_constraints]
        self.cp_subproblem  = cp.Problem(cp.Minimize(self.cp_cost), self.cp_constraints)

        total_param_scalars = sum(p.size for p in self.cp_subproblem.parameters())
        self._converged = False

        quiet = bool(self.method_config.hyperparams.flags.get("quiet", False))
        multi = len(self.scp_trajectory.scp_phases) > 1
        self.reporter = SolveReporter(multi=multi, quiet=quiet)
        self.reporter.subproblem_stats(
            num_phases=len(self.scp_trajectory.scp_phases),
            num_params=total_param_scalars,
            num_constraints=len(self.cp_constraints),
            is_dpp=self.cp_subproblem.is_dcp(dpp=True),
        )

    def update_cvxpy_parameters(self) -> None:
        for scp_phase in self.scp_trajectory.scp_phases.values():
            scp_phase.update_cvxpy_parameters()

    def update_current_iter_data(self) -> None:
        parse_time = self.cp_subproblem.compilation_time * 1000.0
        solve_time = self.cp_subproblem.solver_stats.solve_time * 1000.0

        for scp_phase in self.scp_trajectory.scp_phases.values():
            scp_phase.current_iter_data.parse_time = parse_time
            scp_phase.current_iter_data.solve_time = solve_time
            scp_phase.read_solution()

        trust_region_params = next(iter(self.scp_trajectory.scp_phases.values())).hyperparams.trust_region
        if getattr(trust_region_params, 'line_search', True):
            alpha = trust_region.line_search(
                self.scp_trajectory.scp_phases.values(),
                alpha_min=float(getattr(trust_region_params, 'alpha_min_ls', 1e-7)),
            )
        else:
            alpha = 1.0

        for scp_phase in self.scp_trajectory.scp_phases.values():
            scp_phase.cp_subproblem_status = self.cp_subproblem.status
            scp_phase.apply_step(alpha)

        self._converged = all(s.current_iter_data.converged for s in self.scp_trajectory.scp_phases.values())

        for scp_phase in self.scp_trajectory.scp_phases.values():
            scp_phase.record_iter_data()

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

        max_iter = int(self.method_config.hyperparams.convergence.iter_max)

        total_discretization_ms = 0.0
        total_solve_ms = 0.0
        reason = None

        for i in range(max_iter + 1):
            self.update_cvxpy_parameters()
            try:
                self.cp_subproblem.solve(warm_start=False, **self.method_config.solver_opts)
            except cp.error.SolverError as exc:
                self.reporter.message(f"  {self.scp_trajectory.troubleshoot(exc)}")
                continue

            if self.cp_subproblem.status not in {"optimal", "optimal_inaccurate", "user_limit"}:
                reason = f"Terminated from non-optimal convex subproblem! Status: {self.cp_subproblem.status}"
                break

            usable, message = self.scp_trajectory.troubleshoot_step()
            if not usable:
                self.reporter.message(f"  {message}")
                continue

            self.update_current_iter_data()
            self.display_status()

            for seg in self.scp_trajectory.scp_phases.values():
                total_discretization_ms += seg.current_iter_data.discretization_time
            total_solve_ms += self.cp_subproblem.solver_stats.solve_time * 1000.0

            if self._converged:
                reason = "Terminated from convergence criteria!"
                break

        ran_iterations = any(s.iter_data_list[-1].iter_num > 0 for s in self.scp_trajectory.scp_phases.values())
        if reason is None and ran_iterations and not self._converged:
            reason = "Terminated from hitting maximum iterations!"

        total_ms = total_discretization_ms + total_solve_ms
        self.reporter.footer(
            reason=reason, total_ms=total_ms,
            disc_ms=total_discretization_ms, solve_ms=total_solve_ms,
        )
        self.reporter.trajectory_summary([
            (s.name, s.current_iter_data.t_start, s.current_iter_data.t_final)
            for s in self.scp_trajectory.scp_phases.values()
        ])

    def display_status(self) -> None:
        multi = len(self.scp_trajectory.scp_phases) > 1
        for scp_phase in self.scp_trajectory.scp_phases.values():
            self.reporter.row(
                scp_phase.current_iter_data,
                phase_name=scp_phase.name if multi else None,
            )
