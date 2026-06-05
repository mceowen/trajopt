import time

import numpy as np
import cvxpy as cp

from trajopt.methods.scp.scp_segment import SCPSegment
import trajopt.methods.scp.scp_link_constraints.scp_link_constraint_types as scp_link_constraint_type_module
import trajopt.methods.scp.scp_link_costs.scp_link_cost_types as scp_link_cost_type_module
from trajopt.utils.tools import AttrDict


class SCPMethod():

    def __init__(self, method_config, segments, links) -> None:

        self.method_config = method_config

        self.scp_segments = AttrDict()
        for name, segment in segments.items():
            print("=" * 60)
            print(f"segment: {name}:")
            print("=" * 60)
            self.scp_segments[name] = SCPSegment(segment, method_config)

        self.scp_links = AttrDict({"constraints": AttrDict(), "costs": AttrDict()})

        for name, link_constraint in links.constraints.items():
            print("=" * 60)
            print(f"link constraint: {name}:")
            print("=" * 60)
            scp_class_name = f"scp_{link_constraint.type}"
            scpLinkConstraintClass = getattr(scp_link_constraint_type_module, scp_class_name)
            scp_link = scpLinkConstraintClass(link_constraint)
            scp_link.build(self.scp_segments)
            self.scp_links.constraints[name] = scp_link

        for name, link_cost in links.costs.items():
            print("=" * 60)
            print(f"link cost: {name}:")
            print("=" * 60)
            scp_class_name = f"scp_{link_cost.type}"
            scpLinkCostClass = getattr(scp_link_cost_type_module, scp_class_name)
            scp_link = scpLinkCostClass(link_cost)
            scp_link.build(self.scp_segments)
            self.scp_links.costs[name] = scp_link

        self.cp_cost  = sum(seg.cp_cost for seg in self.scp_segments.values())
        self.cp_cost += sum(link_cost.cp_cost for link_cost in self.scp_links.costs.values())
        self.cp_cost += sum(link_cnstr.cp_cost for link_cnstr in self.scp_links.constraints.values())

        self.cp_constraints = [c for s in self.scp_segments.values() for c in s.cp_constraints]

        for link_constraint in self.scp_links.constraints.values():
            self.cp_constraints += link_constraint.cp_constraints

        self.cp_subproblem = cp.Problem(cp.Minimize(self.cp_cost), self.cp_constraints)
        total_param_scalars = sum(p.size for p in self.cp_subproblem.parameters())

        self._converged = False

        print("subproblem stats:")
        print("------------------------------------------------------------")
        print(f"total number of segments: {len(self.scp_segments)}")
        print(f"total number of cvxpy parameters: {total_param_scalars}")
        print(f"total number of cvxpy constraints: {len(self.cp_constraints)}")
        print(f"total number of linking constraints: {len(self.scp_links.constraints)}")
        print(f"is DPP: {self.cp_subproblem.is_dcp(dpp=True)}")

    def update_cvxpy_parameters(self) -> None:
        for scp_segment in self.scp_segments.values():
            scp_segment.update_cvxpy_parameters()

        for link in self.scp_links.constraints.values():
            link.update_cvxpy_parameters()

    def update_current_iter_data(self) -> None:
        parse_time = self.cp_subproblem.compilation_time * 1000.0
        solve_time = self.cp_subproblem.solver_stats.solve_time * 1000.0

        for scp_segment in self.scp_segments.values():
            scp_segment.current_iter_data.parse_time = parse_time
            scp_segment.current_iter_data.solve_time = solve_time
            scp_segment.read_solution()

        alpha = self.joint_line_search()

        for scp_segment in self.scp_segments.values():
            scp_segment.cp_subproblem_status = self.cp_subproblem.status
            scp_segment.apply_step(alpha)

        for link_constraint in self.scp_links.constraints.values():
            link_constraint.apply_step(alpha)

        self._converged = (
            all(s.current_iter_data.converged for s in self.scp_segments.values())
            and all(link.converged for link in self.scp_links.constraints.values())
        )

        for scp_segment in self.scp_segments.values():
            scp_segment.update_W_dual(alpha)

        for link_constraint in self.scp_links.constraints.values():
            link_constraint.update_W_dual(alpha)

        for scp_segment in self.scp_segments.values():
            scp_segment.record_iter_data()

    def joint_line_search(self, c1=1e-6, beta=0.5, max_iter=20, alpha_min=0.1):
        segments = self.scp_segments
        links    = self.scp_links.constraints

        phi_0, dphi = 0.0, 0.0
        for seg in segments.values():
            v, g = seg.merit_grad_at_zero()
            phi_0 += v
            dphi += g
        for link in links.values():
            v, g = link.merit_grad_at_zero(segments)
            phi_0 += v
            dphi += g

        slope = min(dphi, -abs(dphi) * 1e-6)

        alpha = 1.0
        for _ in range(max_iter):
            phi = sum(seg.evaluate_merit_at_alpha(alpha) for seg in segments.values())
            phi += sum(link.evaluate_merit_at_alpha(segments, alpha) for link in links.values())
            if np.isfinite(phi) and phi <= phi_0 + c1 * alpha * slope:
                return alpha
            alpha *= beta
            if alpha < alpha_min:
                return alpha_min

        return alpha_min

    def solve(self):

        print("-" * 172)
        print("  Iteration |  Discretization |   Solve   |    Parse   |  log(dx/eps) | log(vb_ineq/eps) | log(vb_term/eps) | log(vb_dyn/eps) | Solve status | alpha |  Time of    |   Cost    ")
        print("            |    time [ms]    | time [ms] |  time [ms] |     (state)  |    (ncvx_ineq)   |      (terminal)  |    (dynamics)   |              |       |  Flight [s] |           ")
        print("-" * 172)

        max_iter = int(self.method_config.flags.iter_max)
        solve_start = None

        for i in range(max_iter + 1):
            self.update_cvxpy_parameters()
            self.cp_subproblem.solve(warm_start=False, **self.method_config.solver_opts)

            if self.cp_subproblem.status not in {"optimal", "optimal_inaccurate"}:
                print(f"Terminated from non-optimal convex subproblem! Status: {self.cp_subproblem.status}")
                break

            self.update_current_iter_data()
            self.display_status()

            if i == 0:
                solve_start = time.perf_counter()

            if self._converged:
                print("Terminated from convergence criteria!")
                break

        ran_iterations = any(s.iter_data_list[-1].iter_num > 0 for s in self.scp_segments.values())
        if ran_iterations and not self._converged:
            print("Terminated from hitting maximum iterations!")

        if solve_start is not None:
            total_elapsed_ms = (time.perf_counter() - solve_start) * 1000.0
            print(f"\nTotal elapsed time (from iteration 2 onward): {total_elapsed_ms:.1f} ms")

    def display_status(self) -> None:
        multi = len(self.scp_segments) > 1
        for scp_segment in self.scp_segments.values():
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
