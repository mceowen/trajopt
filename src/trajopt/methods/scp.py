"""Run SCP on one or more coupled trajectory segments."""
from __future__ import annotations

import copy
from numbers import Integral
from time import perf_counter

import cvxpy as cp
import numpy as np

from trajopt.utils._snapshot import thaw, fingerprint
from trajopt.methods.backends.cvxpy import CVXPYBackend
from trajopt.methods.diagnostics import objective_terms
from trajopt.methods.common.penalties import (capture_penalty_state, install_penalty_state,
                                             penalty_descriptor)
from trajopt.methods.initial_guess.preparation import dimensional_samples, segment_guesses
from trajopt.methods.reporting import SolveReporter
from trajopt.methods.trajectory_conversion import export_set, export_trajectory
from trajopt.methods.policies.steps import select_step
from trajopt.methods.prepare import PreparedSolve, prepare, _without_display_flags
from trajopt.methods.provenance import environment
from trajopt.methods.specs import MethodSpec
from trajopt.methods.state import AlgorithmCheckpoint, SolverState
from trajopt.outputs.results import AttemptRecord, SolveResult
from trajopt.trajectories import TrajectorySet
from trajopt.utils.tools import recursive_attrdict


def _attempt_limit(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
        raise ValueError('max_attempts must be a nonnegative integer')
    return int(value)


def _capture_segment(segment, *, owned_history=True):
    return dict(current=copy.deepcopy(segment.current_iter_data),
                history=copy.deepcopy(segment.iter_data_list) if owned_history else list(segment.iter_data_list),
                scalar_trace=copy.deepcopy(segment.state.scalar_trace),
                initial_guess=copy.deepcopy(segment.initial_guess),
                Ts_init=copy.deepcopy(segment.Ts_init), cost_init=copy.deepcopy(segment.cost_init),
                cp_subproblem_status=segment.cp_subproblem_status,
                s_mult=segment.s_mult, lm_mu=getattr(segment, 'lm_mu', None),
                constraints={name: dict(penalties=capture_penalty_state(c.penalties),
                    lagrangian_dual=copy.deepcopy(getattr(c, 'lagrangian_dual', None)),
                    g_nl=copy.deepcopy(getattr(c, 'g_nl', None)))
                    for name, c in segment.constraints.items()})


def _install_segment(segment, data):
    segment.current_iter_data = data['current']
    segment.iter_data_list = list(data['history'])
    segment.state.scalar_trace = list(data.get('scalar_trace', []))
    segment.initial_guess = data['initial_guess']
    segment.Ts_init = data['Ts_init']
    segment.cost_init = data['cost_init']
    segment.cp_subproblem_status = data['cp_subproblem_status']
    segment.s_mult = data['s_mult']
    if data['lm_mu'] is not None:
        segment.lm_mu = data['lm_mu']
    else:
        segment.__dict__.pop('lm_mu', None)
    for name, values in data['constraints'].items():
        constraint = segment.constraints[name]
        install_penalty_state(constraint.penalties, values['penalties'])
        for key in ('lagrangian_dual', 'g_nl'):
            if values[key] is not None:
                setattr(constraint, key, values[key])
            else:
                constraint.__dict__.pop(key, None)
    segment._dz_new = segment._dnu_new = None


class SCPDriver:
    """Solve the coupled trajectory segments with SCP.

    Method presets select SCvx, Auto-SCvx, and their numerical settings.
    ``attempt()`` executes one convex solve. ``step()`` retries until it accepts
    an iteration or reaches the attempt limit.

    ``reset()`` imports a physical trajectory and clears the algorithm state.
    ``resume()`` restores a compatible checkpoint. Backend workspace reuse
    requires a separate option.
    """

    def __init__(self, *, problem=None, method=None, initial_guess=None, prepared=None):
        if prepared is None:
            if problem is None or method is None:
                raise TypeError('Pass problem and method, or a prepared solve')
            prepared = prepare(problem, method, initial_guess=initial_guess)
        else:
            if not isinstance(prepared, PreparedSolve):
                raise TypeError('prepared must be a PreparedSolve')
            if problem is not None or method is not None or initial_guess is not None:
                raise ValueError('Pass either prepared or problem/method/initial_guess')
        self.prepared = prepared
        self.segments = prepared.segments
        self.method_spec = prepared.method
        self.method_config = self.method_spec.config
        self.cp_subproblem = prepared.cp_subproblem
        self.cp_cost = prepared.cp_cost
        self.cp_constraints = prepared.cp_constraints
        self.backend = CVXPYBackend(prepared)
        self.state = SolverState({name: segment.state for name, segment in self.segments.items()})
        self.result = self.last_trial = None
        self._terminal_reason = None
        self._timings = {}
        self._in_solve = False
        self._reference_inputs = fingerprint(self.problem.config)
        self.initial_trajectory = export_set(self.segments, initial=True)
        if len(self.segments) == 1:
            self.subproblem = next(iter(self.segments.values()))
        quiet = bool(self.method_config.hyperparams.flags.get('quiet', False))
        self.reporter = SolveReporter(multi=len(self.segments) > 1, quiet=quiet)
        self.reporter.subproblem_stats(num_phases=len(self.segments),
            num_params=sum(p.size for p in self.cp_subproblem.parameters()),
            num_constraints=len(self.cp_constraints), is_dpp=self.cp_subproblem.is_dcp(dpp=True))

    @property
    def problem(self):
        return self.prepared.problem

    def _check_reference_inputs(self):
        if self._reference_inputs != fingerprint(self.problem.config):
            raise RuntimeError('Prepared inputs changed; reset the algorithm before solving or checkpointing')

    def export_guess(self, *, include_aux=True):
        return export_set(self.segments, include_aux=include_aux)

    def reset(self, *, initial_guess=None):
        supplied = segment_guesses(initial_guess, self.segments)
        prepared, previous = {}, None
        for name, segment in self.segments.items():
            value = segment.prepare_guess(supplied.get(name), previous)
            prepared[name] = value
            previous = dimensional_samples(segment.problem, value.z, value.nu)
        for name, segment in self.segments.items():
            segment.install_initial_guess(prepared[name])
        for variable in self.cp_subproblem.variables():
            variable.value = None
        self.state.attempts.clear()
        self.state.consecutive_rejects = 0
        self.state.converged = False
        self.result = self.last_trial = None
        self._terminal_reason = None
        self._timings = {}
        self._reference_inputs = fingerprint(self.problem.config)
        self.initial_trajectory = export_set(self.segments, initial=True)

    def _checkpoint_signature(self):
        return fingerprint((self.prepared.signature, self.problem.config,
                            _without_display_flags(self.method_config)))

    def update_inputs(self, updates, *, initial_guess=None):
        """Update supported numeric inputs and restart the prepared solver.

        Use the current physical reference as the default initial guess.
        If initialization fails, restore the previous inputs. Structural
        changes require a new prepared problem.
        """
        source = self.export_guess(include_aux=False) if initial_guess is None else initial_guess
        previous = self.problem
        reverse = {}
        for name, values in updates.items():
            if name not in previous.segments:
                raise ValueError(f'Unknown segment {name!r}')
            config = previous.segments[name].config
            reverse[name] = {}
            if 'params' in values:
                reverse[name]['params'] = config.params
            if 'initial_state' in values:
                constraint = next(c for c in config.constraints.values() if c['type'] == 'initial_state')
                reverse[name]['initial_state'] = constraint.value
        self.prepared.update_inputs(updates)
        try:
            self.reset(initial_guess=source)
        except Exception:
            self.prepared.update_inputs(reverse)
            raise
        return self

    def checkpoint(self):
        self._check_reference_inputs()
        return AlgorithmCheckpoint(self._checkpoint_signature(),
            {name: _capture_segment(segment) for name, segment in self.segments.items()},
            tuple(self.state.attempts), self.state.converged, self._timings,
            consecutive_rejects=self.state.consecutive_rejects)

    def resume(self, checkpoint):
        if not isinstance(checkpoint, AlgorithmCheckpoint):
            raise TypeError('resume requires an AlgorithmCheckpoint; use reset for a trajectory')
        if checkpoint.signature != self._checkpoint_signature():
            raise ValueError('Checkpoint is incompatible with this problem, inputs, mesh, or method')
        if set(checkpoint.segments) != set(self.segments):
            raise ValueError('Checkpoint segment identities differ')
        data = recursive_attrdict(thaw(checkpoint.segments))
        for name, segment in self.segments.items():
            current = data[name]['current']
            for field, shape in (('z_opt', segment.cp_params.z_ref.shape),
                                 ('nu_opt', segment.cp_params.nu_ref.shape)):
                if np.shape(current[field]) != shape or not np.isfinite(current[field]).all():
                    raise ValueError(f'Invalid checkpoint {name}.{field}')
        for name, segment in self.segments.items():
            _install_segment(segment, data[name])
        self.state.attempts = list(checkpoint.attempts)
        self.state.converged = checkpoint.converged
        self.state.consecutive_rejects = checkpoint.consecutive_rejects
        self._reference_inputs = fingerprint(self.problem.config)
        self._timings = thaw(checkpoint.timings)
        self.result = self.last_trial = None
        self._terminal_reason = None
        self.initial_trajectory = export_set(self.segments, initial=True)

    def update_cvxpy_parameters(self):
        for segment in self.segments.values():
            segment.update_cvxpy_parameters()

    def warmup_jax(self):
        started = perf_counter()
        self.update_cvxpy_parameters()
        elapsed = perf_counter() - started
        self._timings['warmup_s'] = self._timings.get('warmup_s', 0.) + elapsed
        self._timings['last_call_warmup_s'] = elapsed
        if not self._in_solve:
            self._timings['total_s'] = self._timings.get('total_s', 0.) + elapsed

    def _recover(self, error):
        if self.method_spec.on_error is None:
            raise error
        return self.method_spec.on_error(self.segments.values(), error)

    def attempt(self, *, reuse_workspace=False):
        """Execute one attempt and record whether its candidate was accepted."""
        if self._terminal_reason is not None:
            raise RuntimeError('Solve terminated; reset or resume before another attempt')
        self._check_reference_inputs()
        count = len(self.state.attempts)
        started = perf_counter()
        self.last_trial = None
        try:
            return self._attempt(reuse_workspace=reuse_workspace)
        except Exception as exc:
            if len(self.state.attempts) == count:
                trial = self.last_trial
                timings = {} if trial is None else dict(trial.timings)
                timings['total_s'] = perf_counter() - started
                self.state.attempts.append(AttemptRecord(count + 1,
                    None if trial is None else trial.status, False,
                    self.state.iterations, f'{type(exc).__name__}: {exc}', timings=timings))
            self._terminal_reason = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            if not self._in_solve:
                self._timings['total_s'] = self._timings.get('total_s', 0.) + perf_counter() - started

    def _attempt(self, *, reuse_workspace=False):
        started = perf_counter()
        self.update_cvxpy_parameters()
        local_s = {name: float(s.current_iter_data.discretization_time) / 1000.
                   for name, s in self.segments.items()}
        trial = self.backend.solve(self.method_config.solver_opts, reuse_workspace=reuse_workspace)
        self.last_trial = trial
        timings = dict(trial.timings, local_model_s=local_s)
        accepted, reason, alpha, error_to_raise = False, '', None, None
        candidate_rejected = False
        if trial.error is not None:
            error = cp.error.SolverError(trial.error)
            try:
                reason = self._recover(error)
            except cp.error.SolverError as exc:
                reason, error_to_raise = str(exc), exc
                self._terminal_reason = reason
        elif trial.status not in {'optimal', 'optimal_inaccurate'}:
            reason = f'Convex subproblem terminated with status {trial.status}'
            self._terminal_reason = reason
        else:
            finite = all(d['dz'] is not None and d['dnu'] is not None
                         and np.isfinite(d['dz']).all() and np.isfinite(d['dnu']).all()
                         for d in trial.deltas.values())
            # Adaptive acceptance evaluates the nonlinear merit along this trial.
            # Cache it before invoking the policy, just as read_solution did upstream.
            for name, segment in self.segments.items():
                delta = trial.deltas[name]
                segment._dz_new = None if delta['dz'] is None else np.array(delta['dz'], copy=True)
                segment._dnu_new = None if delta['dnu'] is None else np.array(delta['dnu'], copy=True)
            usable, message = ((True, None) if self.method_spec.on_step is None
                               else self.method_spec.on_step(self.segments.values()))
            if not finite or not usable:
                candidate_rejected = True
                reason = message or 'Candidate contains missing or nonfinite values'
                if not finite and self.method_spec.on_step is None:
                    self._terminal_reason = reason
            else:
                before = {name: _capture_segment(s, owned_history=False) for name, s in self.segments.items()}
                was_converged = self.state.converged
                coupling_eps = [np.array(c.constraint.penalties.eps, copy=True)
                                for c in self.prepared.couplings]
                try:
                    alpha = select_step(self.segments.values(), self.method_spec)
                    for segment in self.segments.values():
                        segment.current_iter_data.parse_time = trial.timings['canonicalization_s'] * 1000.
                        segment.current_iter_data.solve_time = trial.timings['numerical_s'] * 1000.
                        segment.cp_subproblem_status = trial.status
                        segment.apply_step(alpha)
                    self.state.converged = all(s.current_iter_data.converged for s in self.segments.values())
                    self.state.converged = self.state.converged and all(
                        np.all(np.abs(c.residual()) <= eps)
                        for c, eps in zip(self.prepared.couplings, coupling_eps))
                    for segment in self.segments.values():
                        segment.record_iter_data()
                    accepted = True
                except Exception:
                    self.state.converged = was_converged
                    for name, segment in self.segments.items():
                        _install_segment(segment, before[name])
                    raise
        if accepted:
            self.state.consecutive_rejects = 0
        elif candidate_rejected:
            self.state.consecutive_rejects += 1
            limit = self.method_spec.max_rejects
            if limit is not None and self.state.consecutive_rejects >= limit:
                self._terminal_reason = (f'Terminated: {self.state.consecutive_rejects} '
                                         'consecutive trust-region rejects')
                reason = f'{reason}; {self._terminal_reason}'
        timings['total_s'] = perf_counter() - started
        diagnostics = dict(candidate_step_norms={name: {
            key: None if value is None else float(np.max(np.abs(value)))
            for key, value in delta.items()} for name, delta in trial.deltas.items()})
        record = AttemptRecord(len(self.state.attempts) + 1, trial.status, accepted,
                               self.state.iterations, reason, alpha, timings, diagnostics)
        self.state.attempts.append(record)
        if reason:
            self.reporter.message(reason)
        if accepted:
            self.display_status()
        if error_to_raise is not None:
            raise error_to_raise
        return record

    def step(self, *, max_attempts=None, reuse_workspace=False):
        limit = _attempt_limit(self.method_spec.iter_max + 1 if max_attempts is None else max_attempts)
        if limit < 1:
            raise ValueError('max_attempts must be positive')
        for _ in range(limit):
            record = self.attempt(reuse_workspace=reuse_workspace)
            if record.accepted or self._terminal_reason is not None:
                break
        return record

    def solve(self, verbose=None, *, initial_guess=None, budget=None, reuse_workspace=False):
        self.method_spec = MethodSpec.from_config(self.method_config)
        budget = {} if budget is None else dict(budget)
        if set(budget) - {'max_attempts', 'wall_time_s'}:
            raise ValueError('Budget supports max_attempts and wall_time_s')
        limit = _attempt_limit(budget.get('max_attempts', self.method_spec.iter_max + 1))
        deadline = float(budget.get('wall_time_s', np.inf))
        if limit < 0 or deadline < 0 or np.isnan(deadline):
            raise ValueError('Solve budgets must be nonnegative and not NaN')
        retention = self.method_config.get('history', 'scalars')
        if retention not in {'none', 'scalars', 'full'}:
            raise ValueError('history must be none, scalars, or full')
        if initial_guess is not None:
            self.reset(initial_guess=initial_guess)
        self._check_reference_inputs()
        if verbose is not None:
            self.reporter.quiet = not verbose
        for segment in self.segments.values():
            segment.method_config.history = retention
        started = perf_counter()
        prior_elapsed = self._timings.get('total_s', 0.)
        self._timings['last_call_warmup_s'] = 0.
        self._terminal_reason = None
        self._in_solve = True
        reason, error = 'Attempt budget exhausted', None
        try:
            if limit and deadline > 0:
                self.warmup_jax()
            self.reporter.header()
            for _ in range(limit):
                if perf_counter() - started >= deadline:
                    reason = 'Wall-time budget exhausted'
                    break
                self.attempt(reuse_workspace=reuse_workspace)
                if self._terminal_reason is not None:
                    reason = self._terminal_reason
                    break
                if self.state.converged:
                    reason = 'Convergence criteria satisfied'
                    break
        except Exception as exc:
            error = exc
            reason = f'{type(exc).__name__}: {exc}'
            self._terminal_reason = reason
        elapsed = perf_counter() - started
        self._in_solve = False
        self._timings.update(total_s=prior_elapsed + elapsed, last_call_s=elapsed,
                             preparation_s=self.prepared.preparation_s,
                             deadline_overrun_s=max(0., elapsed - deadline))
        self.result = self._make_result(reason, failed=self._terminal_reason is not None)
        self.reporter.footer(reason=reason, total_ms=elapsed * 1000.,
            disc_ms=self.result.timings['local_model_s'] * 1000.,
            solve_ms=self.result.timings['numerical_s'] * 1000.)
        self.reporter.trajectory_summary([(name, s.current_iter_data.t_start,
            s.current_iter_data.t_final) for name, s in self.segments.items()])
        if error is not None:
            raise error
        return self.result

    def _make_result(self, reason, *, failed=False):
        history_option = self.method_config.get('history', 'scalars')
        if history_option not in {'none', 'scalars', 'full'}:
            raise ValueError('history must be none, scalars, or full')
        data, diagnostics = {}, {}
        for name, segment in self.segments.items():
            iterations = segment.iter_data_list
            if history_option != 'full':
                iterations = [iterations[0], iterations[-1]] if len(iterations) > 1 else iterations
            data[name] = dict(initial_guess=segment.initial_guess, iter_data_list=iterations,
                scalar_history=segment.state.scalar_trace if history_option != 'none' else [],
                mesh=dict(tau=segment.mesh.tau, discretize=segment.mesh.discretize, segments=segment.mesh.segments),
                hyperparams=segment.hyperparams,
                constraints={key: dict(name=c.name, type=c.type, shape=c.shape,
                    penalties=penalty_descriptor(c.penalties))
                    for key, c in segment.constraints.items()})
            it = segment.current_iter_data
            diagnostics[name] = dict(**objective_terms(segment), reported_cost=float(it.get('cost', 0.)),
                penalty_cost=float(it.get('penalty_cost', 0.)), stopping_ratios=it.get('chk', {}),
                dynamics_defect_nd=it.get('defect'),
                constraint_residuals_nd={key: c.g_nl for key, c in segment.constraints.items()
                                         if getattr(c, 'g_nl', None) is not None})
        diagnostics['couplings'] = {f'{c.spec.target}.{c.spec.name}': c.residual()
                                    for c in self.prepared.couplings}
        attempts = tuple(self.state.attempts)
        timings = dict(self._timings,
            numerical_s=sum(a.timings.get('numerical_s', 0.) for a in attempts),
            canonicalization_s=sum(a.timings.get('canonicalization_s', 0.) for a in attempts),
            local_model_s=sum(sum(a.timings.get('local_model_s', {}).values()) for a in attempts))
        initial = self.initial_trajectory
        try:
            trajectory = export_set(self.segments) if self.state.iterations else None
        except ValueError as exc:
            trajectory, failed = None, True
            reason = f'Unusable accepted trajectory (nonfinite values or invalid time map): {exc}'
        history = []
        if history_option == 'full':
            for index in range(1, min(len(s.iter_data_list) for s in self.segments.values())):
                try:
                    history.append(TrajectorySet({name: export_trajectory(s, data=s.iter_data_list[index])
                                                  for name, s in self.segments.items()}))
                except ValueError:
                    diagnostics.setdefault('unavailable_history', []).append(index)
        status = ('failed' if failed or trajectory is None else
                  'converged' if self.state.converged else 'not_converged')
        return SolveResult(trajectory, initial, status, reason,
            attempts[-1].status if attempts else None,
            self.state.iterations, attempts, timings, diagnostics,
            dict(algorithm='SCPDriver', prepared_signature=self.prepared.signature,
                 environment=environment(),
                 timing_scope='seconds since reset/resume; total_s excludes preparation; '
                              'last_call_s is the latest solve call; component timings overlap total_s'),
            tuple(history), self.problem, self.method_spec, data)

    def display_status(self):
        for name, segment in self.segments.items():
            self.reporter.row(segment.current_iter_data, phase_name=name if len(self.segments) > 1 else None)
