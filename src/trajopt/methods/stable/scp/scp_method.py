"""Solve one or more coupled trajectory phases with SCP."""
from __future__ import annotations

from time import perf_counter

import cvxpy as cp
import numpy as np

from trajopt.common._snapshot import fingerprint
from trajopt.methods.stable.scp.transcribe import PreparedSolve, transcribe
from trajopt.methods.common.trust_region import select_step
from trajopt.methods.common.utils.bindings import update_solver_inputs
from trajopt.methods.common.utils.logging import (
    SolveReporter,
    record_iteration,
    record_solve,
)
from trajopt.methods.common.utils.solverstate import (
    SolverState,
    TrialStep,
    reset_solver,
    restore_checkpoint,
    restore_on_error,
    save_checkpoint,
)
from trajopt.methods.common.utils.specs import prepare_solve
from trajopt.methods.common.utils.trajectory_conversion import export_set


class SCPMethod:
    """Run SCP with the selected discretization, trust region, and penalty rules.

    Use solve() to run the method. Set budget={"max_attempts": 1} for one trial.
    Use reset() for a new initial guess. Use resume() to restore a saved solve.
    """

    def __init__(self, *, problem=None, method=None, initial_guess=None, prepared=None):
        # Prepare the phase problems and their connections, or use a prepared problem.
        if prepared is None:
            if problem is None or method is None:
                raise TypeError('Pass problem and method, or a prepared solve')
            prepared = transcribe(problem, method, initial_guess=initial_guess)
        else:
            if not isinstance(prepared, PreparedSolve):
                raise TypeError('prepared must be a PreparedSolve')
            if problem is not None or method is not None or initial_guess is not None:
                raise ValueError('Pass either prepared or problem/method/initial_guess')
        self.prepared = prepared
        self.phases = prepared.phases
        self.method_spec = prepared.method
        self.method_config = self.method_spec.config
        self.cp_subproblem = prepared.cp_subproblem
        self.cp_cost = prepared.cp_cost
        self.cp_constraints = prepared.cp_constraints
        # Keep the phase results together so that all phases use the same solve history.
        self.state = SolverState({name: phase.state for name, phase in self.phases.items()})
        self.result = self.last_trial = None
        self._terminal_reason = None
        self._timings = {}
        self._in_solve = False
        self._reference_inputs = fingerprint(self.problem.config)
        self.initial_trajectory = export_set(self.phases, initial=True)
        if len(self.phases) == 1:
            self.subproblem = next(iter(self.phases.values()))
        quiet = bool(self.method_config.hyperparams.flags.get('quiet', False))
        self.reporter = SolveReporter(multi=len(self.phases) > 1, quiet=quiet)
        self.reporter.subproblem_stats(num_phases=len(self.phases),
            num_params=sum(p.size for p in self.cp_subproblem.parameters()),
            num_constraints=len(self.cp_constraints), is_dpp=self.cp_subproblem.is_dcp(dpp=True))

    def solve(self, verbose=None, *, initial_guess=None, budget=None, reuse_workspace=False):
        """Repeat the convex solve until convergence or a limit stops the method.

        Continue from the current trajectory unless a new initial guess is supplied.
        Each attempt counts toward the limit, including a rejected step.
        """
        limits = prepare_solve(self, verbose=verbose, initial_guess=initial_guess, budget=budget)
        # Save the result and elapsed time when the solve ends, including after an error.
        with record_solve(self, limits.wall_time_s) as run:
            if limits.max_attempts and limits.wall_time_s > 0:
                # Prepare the numerical functions. This time counts toward the time limit.
                self.warmup_jax()
            self.reporter.header()

            for _ in range(limits.max_attempts):
                # Check the time before each attempt. An attempt in progress can exceed the limit.
                if perf_counter() - run.started >= limits.wall_time_s:
                    run.reason = 'Wall-time budget exhausted'
                    break

                with record_iteration(self) as record:
                    # Form the convex approximation at the current trajectory and solve it once.
                    self.update_cvxpy_parameters()
                    self.last_trial = self.solve_convex_subproblem(reuse_workspace=reuse_workspace)
                    # Check the solve result before checking the proposed trajectory change.
                    usable, record.reason = self.check_solver_status()
                    if usable:
                        accepted, record.reason = self.check_step_acceptance()
                        if accepted:
                            # Change the trajectory only after the step is accepted.
                            record.alpha = self.update_current_iter_data()
                            record.accepted = True
                            self.state.consecutive_rejects = 0

                # Record the attempt before checking whether to stop.
                if self._terminal_reason is not None:
                    run.reason = self._terminal_reason
                    break
                if self.state.converged:
                    run.reason = 'Convergence criteria satisfied'
                    break

        return self.result

    def check_solver_status(self):
        """Apply the recovery rule after a solver error. Stop on other solve failures."""
        trial = self.last_trial
        if trial.error is not None:
            error = cp.error.SolverError(trial.error)
            try:
                if self.method_spec.on_error is None:
                    raise error
                # The recovery rule can change the settings for the next attempt.
                reason = self.method_spec.on_error(self.phases.values(), error)
            except cp.error.SolverError as error:
                self._terminal_reason = str(error)
                raise
            return False, reason
        if trial.status not in {'optimal', 'optimal_inaccurate'}:
            self._terminal_reason = f'Convex subproblem terminated with status {trial.status}'
            return False, self._terminal_reason
        return True, ''

    def check_step_acceptance(self):
        """Check the proposed step without changing the current trajectory."""
        for name, phase in self.phases.items():
            delta = self.last_trial.deltas[name]
            phase._dz_new = None if delta['dz'] is None else np.array(delta['dz'], copy=True)
            phase._dnu_new = None if delta['dnu'] is None else np.array(delta['dnu'], copy=True)

        finite = all(delta['dz'] is not None and delta['dnu'] is not None
                     and np.isfinite(delta['dz']).all() and np.isfinite(delta['dnu']).all()
                     for delta in self.last_trial.deltas.values())
        # Apply the selected acceptance rule. It can change the trust region after a rejection.
        usable, message = ((True, None) if self.method_spec.on_step is None
                           else self.method_spec.on_step(self.phases.values()))

        if finite and usable:
            return True, ''

        reason = message or 'Candidate contains missing or nonfinite values'
        if not finite and self.method_spec.on_step is None:
            self._terminal_reason = reason
        self.state.consecutive_rejects += 1
        limit = self.method_spec.max_rejects
        if limit is not None and self.state.consecutive_rejects >= limit:
            self._terminal_reason = (f'Terminated: {self.state.consecutive_rejects} '
                                     'consecutive trust-region rejects')
            reason = f'{reason}; {self._terminal_reason}'
        return False, reason

    def update_current_iter_data(self):
        """Apply the step, check convergence, and update the weights for each phase."""
        trial = self.last_trial
        # Restore all phases to their previous values if any part of the update fails.
        with restore_on_error(self.phases, self.state):
            # Keep the current connection tolerances for the convergence check below.
            coupling_eps = [np.array(c.constraint.penalties.eps, copy=True)
                            for c in self.prepared.couplings]
            # Use the same step length for all phases to preserve their connections.
            alpha = select_step(self.phases.values(), self.method_spec)
            for phase in self.phases.values():
                phase.current_iter_data.parse_time = trial.timings['canonicalization_s'] * 1000.
                phase.current_iter_data.solve_time = trial.timings['numerical_s'] * 1000.
                phase.cp_subproblem_status = trial.status
                phase.apply_step(alpha)
                phase.evaluate_step()
                phase.check_convergence()

                # Check convergence before changing the trust region and penalty weights.
                # Update both, including on the final iteration.
                phase.update_trust_region()
                phase.update_constraint_penalties()

            # Require convergence in every phase and at every phase connection.
            self.state.converged = all(s.current_iter_data.converged for s in self.phases.values())
            self.state.converged = self.state.converged and all(
                np.all(np.abs(c.residual()) <= eps)
                for c, eps in zip(self.prepared.couplings, coupling_eps))
            for phase in self.phases.values():
                phase.record_iter_data()
        return alpha

    def update_cvxpy_parameters(self):
        """Update each phase's convex approximation at its current trajectory."""
        for phase in self.phases.values():
            phase.update_cvxpy_parameters()

    def solve_convex_subproblem(self, *, reuse_workspace=False):
        """Solve the convex problem once. Return the proposed step, status, and solve times."""
        problem = self.cp_subproblem
        options = dict(self.method_config.solver_opts)
        # Use the solver's warm_start setting if it is supplied.
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
        # A failed solve can leave old CVXPY values. Do not use those values.
        stats = None if raised else problem.solver_stats
        status = None if raised else problem.status
        timings = dict(backend_wall_s=elapsed,
                       canonicalization_s=0. if raised else float(problem.compilation_time or 0.),
                       numerical_s=float(stats.solve_time or 0.) if stats is not None else 0.)
        deltas = {}
        if error is None and status in {'optimal', 'optimal_inaccurate'}:
            for name, phase in self.phases.items():
                deltas[name] = dict(dz=phase.dz.value, dnu=phase.dnu.value)
        return TrialStep(status, deltas, timings, error)

    def warmup_jax(self):
        """Prepare the numerical functions before the first timed iteration."""
        started = perf_counter()
        self.update_cvxpy_parameters()
        elapsed = perf_counter() - started
        self._timings['warmup_s'] = self._timings.get('warmup_s', 0.) + elapsed
        self._timings['last_call_warmup_s'] = elapsed
        # During solve(), the total elapsed time already includes this preparation.
        if not self._in_solve:
            self._timings['total_s'] = self._timings.get('total_s', 0.) + elapsed

    @property
    def problem(self):
        """Return the physical problem definition used by this solve."""
        return self.prepared.problem

    def export_guess(self, *, include_aux=True):
        """Return the current trajectory in physical units."""
        return export_set(self.phases, include_aux=include_aux)

    def reset(self, *, initial_guess=None):
        """Set the initial guess and clear the iteration data."""
        reset_solver(self, initial_guess=initial_guess)

    def update_inputs(self, updates, *, initial_guess=None):
        """Change numeric inputs and restart from the current or supplied trajectory."""
        return update_solver_inputs(self, updates, initial_guess=initial_guess)

    def checkpoint(self):
        """Save the trajectory, weights, and iteration data."""
        return save_checkpoint(self)

    def resume(self, checkpoint):
        """Restore a saved solve with the same problem, mesh, and method."""
        restore_checkpoint(self, checkpoint)

    def display_status(self):
        """Show the current iteration results for each phase."""
        for name, phase in self.phases.items():
            self.reporter.row(phase.current_iter_data, phase_name=name if len(self.phases) > 1 else None)
