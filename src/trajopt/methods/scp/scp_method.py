import time

import numpy as np
import cvxpy as cp

from trajopt.methods.scp.scp_trajectory import SCPTrajectory

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

        print("subproblem stats:")
        print("------------------------------------------------------------")
        print(f"total number of segments: {len(self.scp_trajectory.scp_segments)}")
        print(f"total number of cvxpy parameters: {total_param_scalars}")
        print(f"total number of cvxpy constraints: {len(self.cp_constraints)}")
        print(f"is DPP: {self.cp_subproblem.is_dcp(dpp=True)}")

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
            alpha = self.line_search()
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

    def line_search(self, c1=1e-4, beta=0.5, max_iter=20, alpha_min=0.000000001):
        segments = self.scp_trajectory.scp_segments

        phi_0, dphi = 0.0, 0.0
        for seg in segments.values():
            v, g = seg.merit_grad_at_zero()
            phi_0 += v
            dphi += g

        slope = min(dphi, -abs(dphi) * 1e-6)

        alpha = 1.0
        for _ in range(max_iter):
            phi = sum(seg.evaluate_merit_at_alpha(alpha) for seg in segments.values())
            if np.isfinite(phi) and phi <= phi_0 + c1 * alpha * slope:
                return alpha
            alpha *= beta
            if alpha < alpha_min:
                return alpha_min

        return alpha_min

    def warmup_jax(self):
        """Run a dummy discretization pass to trigger all JAX JIT compilations."""
        print("Compiling JAX kernels (warmup)...", end=" ", flush=True)
        warmup_start = time.perf_counter()
        self.update_cvxpy_parameters()
        warmup_ms = (time.perf_counter() - warmup_start) * 1000.0
        print(f"done ({warmup_ms:.0f} ms)")

    def solve(self):

        self.warmup_jax()

        print("-" * 172)
        print("  Iteration |  Discretization |   Solve   |    Parse   |  log(dx/eps) | log(vb_ineq/eps) | log(vb_term/eps) | log(vb_dyn/eps) | Solve status | alpha |  Time of    |   Cost    ")
        print("            |    time [ms]    | time [ms] |  time [ms] |     (state)  |    (ncvx_ineq)   |      (terminal)  |    (dynamics)   |              |       |  Flight [s] |           ")
        print("-" * 172)

        max_iter = int(self.method_config.flags.iter_max)

        total_discretization_ms = 0.0
        total_solve_ms = 0.0

        for i in range(max_iter + 1):
            self.update_cvxpy_parameters()
            self.cp_subproblem.solve(warm_start=False, **self.method_config.solver_opts)

            if self.cp_subproblem.status not in {"optimal", "optimal_inaccurate", "user_limit"}:
                print(f"Terminated from non-optimal convex subproblem! Status: {self.cp_subproblem.status}")
                break

            self.update_current_iter_data()
            self.display_status()

            for seg in self.scp_trajectory.scp_segments.values():
                total_discretization_ms += seg.current_iter_data.discretization_time
            total_solve_ms += self.cp_subproblem.solver_stats.solve_time * 1000.0

            if self._converged:
                print("Terminated from convergence criteria!")
                break

        ran_iterations = any(s.iter_data_list[-1].iter_num > 0 for s in self.scp_trajectory.scp_segments.values())
        if ran_iterations and not self._converged:
            print("Terminated from hitting maximum iterations!")

        total_ms = total_discretization_ms + total_solve_ms
        print(f"\nTotal SCP time: {total_ms:.1f} ms (discretize: {total_discretization_ms:.1f}, solve: {total_solve_ms:.1f})")

    def display_status(self) -> None:
        multi = len(self.scp_trajectory.scp_segments) > 1
        for scp_segment in self.scp_trajectory.scp_segments.values():
            current_iter_data = scp_segment.current_iter_data

            with np.errstate(divide="ignore"):
                log_dz_ratio      = float(np.log10(current_iter_data.chk.dz))
                log_vb_ineq_ratio = float(np.log10(current_iter_data.chk.nonconvex_inequality))
                log_vb_term_ratio = float(np.log10(current_iter_data.chk.final_state))
                log_vb_dyn_ratio  = float(np.log10(current_iter_data.chk.dynamics))

            prefix = f"[{scp_segment.name}] " if multi else ""
            print(
                prefix + "{:^12d}|{:^17.1f}|{:^11.1f}|{:^12.1f}|{:^+14.1f}|{:^+18.1f}|{:^+18.1f}|{:^+17.1f}|{:^14s}|{:^7.3f}|{:^13.2f}|{:^11.1f}".format(
                    int(current_iter_data.iter_num),
                    float(current_iter_data.discretization_time),
                    float(current_iter_data.solve_time),
                    float(current_iter_data.parse_time),
                    log_dz_ratio,
                    log_vb_ineq_ratio,
                    log_vb_term_ratio,
                    log_vb_dyn_ratio,
                    str(current_iter_data.status),
                    float(current_iter_data.get("alpha", 1.0)),
                    float(current_iter_data.T_opt),
                    float(current_iter_data.cost),
                )
            )