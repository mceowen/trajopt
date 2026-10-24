"""Solve the joint CVXPY problem without changing the reference trajectory."""
from time import perf_counter

import cvxpy as cp

from trajopt.methods.state import TrialStep


class CVXPYBackend:
    def __init__(self, prepared):
        self.prepared = prepared

    def solve(self, options, *, reuse_workspace=False):
        problem = self.prepared.cp_subproblem
        options = dict(options)
        options.setdefault('warm_start', reuse_workspace)
        started = perf_counter()
        error = None
        raised = False
        try:
            problem.solve(**options)
            if problem.status == cp.USER_LIMIT:
                error = 'Convex subproblem reached the solver iteration limit'
        except cp.error.SolverError as exc:
            error, raised = str(exc), True
        elapsed = perf_counter() - started
        # CVXPY retains its previous status/stats when a later solve raises.
        # They do not describe this failed attempt; wall time remains measured.
        stats = None if raised else problem.solver_stats
        status = None if raised else problem.status
        timings = dict(backend_wall_s=elapsed,
                       canonicalization_s=0. if raised else float(problem.compilation_time or 0.),
                       numerical_s=float(stats.solve_time or 0.) if stats is not None else 0.)
        deltas = {}
        if error is None and status in {'optimal', 'optimal_inaccurate'}:
            for name, segment in self.prepared.segments.items():
                deltas[name] = dict(dz=segment.dz.value, dnu=segment.dnu.value)
        return TrialStep(status, deltas, timings, error)
