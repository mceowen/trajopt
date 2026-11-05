"""Report solve progress, evaluate result diagnostics, and record the environment."""
import copy
import hashlib
import platform
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter

import cvxpy as cp
import numpy as np

from trajopt.common._snapshot import readonly_array
from trajopt.common.results import AttemptRecord, SolveResult
from trajopt.common.trajectory import TrajectorySet
from trajopt.methods.common.penalties import penalty_descriptor
from trajopt.methods.common.utils.trajectory_conversion import (
    export_set,
    export_trajectory,
)

_SEP = " │ "


class _Column:
    """One table column: its key, header text, width and number format."""

    def __init__(self, key, header, width, kind):
        self.key = key
        self.header = header
        self.width = width
        self.kind = kind

    def format_value(self, value):
        if value is None:
            text = "—"
        elif self.kind == "int":
            text = f"{int(value):d}"
        elif self.kind == "f2":
            text = f"{value:.2f}"
        elif self.kind == "f3":
            text = f"{value:.3f}"
        elif self.kind == "f4":
            text = f"{value:.4f}"
        elif self.kind == "sci":
            text = f"{value:.4e}"
        elif self.kind == "g":
            text = f"{value:.8g}"
        else:  # "str"
            text = str(value)
        return _center(text, self.width)

    def format_header(self):
        return _center(self.header, self.width)


def _center(text, width):
    """Center text in width, truncating it if it does not fit."""
    if len(text) > width:
        text = text[:width]
    return f"{text:^{width}}"


# The residual columns hold the ratio metric/eps, which reaches 1 at convergence.
_BASE_COLUMNS = [
    _Column("iter",    "Iter",        5,  "int"),
    _Column("disc",    "Disc [ms]",   10, "f2"),
    _Column("solve",   "Solve [ms]",  10, "f2"),
    _Column("parse",   "Parse [ms]",  10, "f2"),
    _Column("dx",      "dx/eps",      12, "sci"),
    _Column("vb_ineq", "vb_ineq/eps", 12, "sci"),
    _Column("vb_eq",   "vb_eq/eps",   12, "sci"),
    _Column("vb_term", "vb_term/eps", 12, "sci"),
    _Column("vb_dyn",  "vb_dyn/eps",  12, "sci"),
    _Column("status",  "Status",      12, "str"),
    _Column("alpha",   "alpha",       8,  "f4"),
    _Column("ratio",   "ratio",       8,  "f4"),
    _Column("s_mult",  "s_mult",      10, "sci"),
    _Column("duration", "Duration [s]", 12, "f3"),
    _Column("cost",    "Cost",        14, "g"),
    _Column("penalty", "Penalty",     14, "g"),
]

_SEG_COLUMN = _Column("seg", "Phase", 12, "str")


class SolveReporter:
    """Prints the SCP solve progress table. multi adds a Phase column, quiet silences it."""

    def __init__(self, *, multi=False, quiet=False):
        self.quiet = quiet
        self.columns = ([_SEG_COLUMN] + _BASE_COLUMNS) if multi else list(_BASE_COLUMNS)

    def _header_line(self):
        return _SEP.join(col.format_header() for col in self.columns)

    def message(self, text=""):
        """Print a single line of text."""
        if not self.quiet:
            print(text)

    def subproblem_stats(self, *, num_phases, num_params, num_constraints, is_dpp):
        if self.quiet:
            return
        print("subproblem stats:")
        print("------------------------------------------------------------")
        print(f"total number of phases: {num_phases}")
        print(f"total number of cvxpy parameters: {num_params}")
        print(f"total number of cvxpy constraints: {num_constraints}")
        print(f"is DPP: {is_dpp}")

    def header(self):
        if self.quiet:
            return
        line = self._header_line()
        divider = "─" * len(line)
        print(divider)
        print(line)
        print(divider)

    def row(self, iter_data, phase_name=None):
        if self.quiet:
            return
        values = self._extract(iter_data, phase_name)
        cells = [col.format_value(values.get(col.key)) for col in self.columns]
        print(_SEP.join(cells))

    def footer(self, *, reason=None, total_ms=0.0, disc_ms=0.0, solve_ms=0.0):
        if self.quiet:
            return
        print("─" * len(self._header_line()))
        if reason:
            print(reason)
        print(f"Total SCP time: {total_ms:.1f} ms (discretize: {disc_ms:.1f}, solve: {solve_ms:.1f})")

    def trajectory_summary(self, phases):
        """Print each phase's start, end and duration, then the time of flight."""
        if self.quiet or not phases:
            return
        if len(phases) > 1:
            name_w = max(len(name) for name, _, _ in phases)
            print("Phase times")
            for name, t_start, t_final in phases:
                print(f"  {name:<{name_w}}  {t_start:10.3f} → {t_final:10.3f} s   ({t_final - t_start:.3f} s)")
        print(f"Time of flight: {phases[-1][2] - phases[0][1]:.3f} s")

    @staticmethod
    def _extract(iter_data, phase_name):
        chk = iter_data.chk
        return {
            "seg":     phase_name,
            "iter":    int(getattr(iter_data, "iter_num")),
            "disc":    float(getattr(iter_data, "discretization_time")),
            "solve":   float(getattr(iter_data, "solve_time")),
            "parse":   float(getattr(iter_data, "parse_time")),
            "dx":      float(getattr(chk, "dz")),
            "vb_ineq": float(getattr(chk, "nonconvex_inequality")),
            "vb_eq":   float(getattr(chk, "nonconvex_equality")),
            "vb_term": float(getattr(chk, "final_state")),
            "vb_dyn":  float(getattr(chk, "dynamics")),
            "status":  str(getattr(iter_data, "status")),
            "alpha":   float(getattr(iter_data, "alpha", 1.0)),
            "ratio":   float(getattr(iter_data, "tr_ratio", 1.0)),
            "s_mult":  float(getattr(iter_data, "s_mult", 1.0)),
            "duration": float(getattr(iter_data, "t_final")) - float(getattr(iter_data, "t_start")),
            "cost":    float(getattr(iter_data, "cost")),
            "penalty": float(getattr(iter_data, "penalty_cost", 0.0)),
        }


def _number(value):
    """Sum a finite numeric value or a constant CVXPY expression."""
    if isinstance(value, cp.Expression):
        if value.variables() or value.parameters():
            raise ValueError('Objective callback returned a live optimization expression')
        value = value.value
    if value is None:
        raise ValueError('Objective callback did not produce a value')
    array = np.asarray(value, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError('Objective contribution is nonfinite')
    return float(np.sum(array))


def _evaluate_cost(phase, cost, z, nu, params):
    """Evaluate the configured SCP cost from the supplied trajectory arrays."""
    kind = cost.type
    idx = phase.index_map.indices
    if kind == 'nonconvex_running':
        gamma = getattr(cost, 'gamma_idx', None)
        if gamma is None or not len(idx.z.running_cost):
            return 0., 'No running-cost accumulator is installed; recipe adds no cost'
        return (_number(cost.w*z[-1, idx.z.running_cost[gamma]]),
                'Accepted optimized running-cost accumulator; no independent quadrature')
    if kind in {'nonconvex_terminal', 'nonconvex_minimax'}:
        nodes = np.atleast_1d(cost.nodes)
        values = np.array([np.atleast_1d(cost.fcn_znu(z[k], nu[k], params)) for k in nodes])
        if kind == 'nonconvex_minimax':
            if not len(nodes):
                raise ValueError('A minimax cost needs at least one evaluation node')
            return (_number(cost.w*np.max(values, axis=0)),
                    'Componentwise maximum of nonlinear callback values at configured accepted nodes')
        # The current terminal recipe does not apply a separate w factor.
        return (_number(values), 'Nonlinear callback at configured accepted nodes; configured output scale applied')
    if kind in {'convex_terminal', 'convex_running'}:
        # Match the implemented recipe: normalized x/u and literal time zero.
        # Constructing Constants allows CP-aware callbacks without using any
        # backend variable, reference parameter, or previous solve expression.
        nodes = np.atleast_1d(cost.nodes)
        node_weights = np.ones(len(z))
        if kind == 'convex_running':
            if phase.mesh.weights is not None:
                node_weights = np.concatenate(([0.], len(z)*phase.mesh.weights))
            nodes = [k for k in range(len(z)) if k in nodes]
        value = 0.
        for k in nodes:
            if node_weights[k] == 0:
                continue
            x = cp.Constant(z[k, idx.z.state])
            u = cp.Constant(nu[k, idx.nu.control])
            value += _number(node_weights[k]*cost.w*cost.fcn_dim(x, u, 0, params))
        return value, 'Configured nodal convex callback on normalized x/u with time zero, matching recipe'
    if kind == 'min_time':
        if not phase.free_final_time:
            return 0., 'Fixed-time recipe adds no minimum-time objective'
        from trajopt.methods.stable.scp.cost_types import (
            min_time_all_nodes,
            min_time_final_node,
        )
        recipe = phase.fcns.cost.min_time
        if recipe is min_time_final_node:
            return _number(z[-1, idx.z.time]), 'Final normalized time, matching min_time_final_node'
        if recipe is min_time_all_nodes:
            return _number(z[:, idx.z.time]), 'Sum of normalized nodal times, matching min_time_all_nodes'
        raise ValueError('Unrecognized minimum-time recipe')
    if kind == 'min_norm_terminal':
        target = cost.value if cost.value is not None else np.zeros(len(cost.idx))
        return _number(np.linalg.norm(z[-1, cost.idx]-target)), 'Terminal normalized-state Euclidean norm'
    if kind == 'final_state':
        return _number(cost.w*z[-1, cost.idx]), 'Weighted selected final normalized-state coordinates'
    if kind == 'final_control':
        return _number(cost.w*nu[-1, cost.idx]), 'Weighted selected final normalized-control coordinates'
    if kind in {'regularization', 'rate_regularization'}:
        if cost.set == 'state':
            values = z
        elif cost.set == 'control':
            values = nu
        else:
            raise ValueError(f'Unrecognized regularization set {cost.set!r}')
        if kind == 'rate_regularization':
            values = np.diff(values[:, cost.idx], axis=0)
            source = 'Adjacent normalized-coordinate differences on selected indices; no time division'
        else:
            source = 'All augmented normalized coordinates, matching the implemented regularization recipe'
        if cost.norm_type == 'l2':
            value = np.sum(values**2)
        elif cost.norm_type == 'l1':
            value = np.sum(np.abs(values))
        else:
            raise ValueError(f'Unrecognized regularization norm {cost.norm_type!r}')
        return _number(cost.w*value), source
    raise ValueError(f'Unsupported objective type {kind!r}')


def objective_terms(phase):
    """Evaluate the configured objective on the accepted reference trajectory.

    Terms use the configured scales and weights. Their physical units can
    differ. The objective excludes solver penalties. Running costs use the
    accepted cost accumulators without further integration.

    If any term cannot be evaluated, the total objective is unavailable.
    This function does not change the solver state.
    """
    z = readonly_array(phase.current_iter_data.z_opt)
    nu = readonly_array(phase.current_iter_data.nu_opt)
    params = copy.deepcopy(phase.params)
    terms, unsupported = {}, []
    for name, wrapper in phase.costs.items():
        cost = wrapper.cost
        term = {'type': cost.type, 'value': None,
                'units': 'configured normalized objective contribution'}
        if hasattr(cost, 'scale'):
            term['configured_scale'] = cost.scale
        try:
            term['value'], term['source'] = _evaluate_cost(phase, cost, z, nu, params)
        except Exception as exc:
            unsupported.append(name)
            term['source'] = 'unavailable'
            term['reason'] = f'{type(exc).__name__}: {exc}'
        terms[name] = term
    return {'objective': None if unsupported else float(sum(term['value'] for term in terms.values())),
            'objective_terms': terms,
            'objective_kind': 'configured scaled objective on accepted reference, excluding algorithm penalties and proximal costs',
            'unsupported_objective_terms': tuple(unsupported)}


@lru_cache(maxsize=1)
def collect_environment():
    packages = {}
    for name in ('trajopt', 'numpy', 'scipy', 'jax', 'jaxlib', 'cvxpy', 'clarabel'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    root = Path(__file__).resolve().parents[3]
    digest = hashlib.sha256()
    for path in sorted(root.rglob('*.py')):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return dict(python=platform.python_version(), platform=platform.platform(),
                packages=packages, package_source_sha256=digest.hexdigest())


def make_result(solver, reason, *, failed=False):
    history_option = solver.method_config.get('history', 'scalars')
    if history_option not in {'none', 'scalars', 'full'}:
        raise ValueError('history must be none, scalars, or full')
    data, diagnostics = {}, {}
    for name, phase in solver.phases.items():
        iterations = phase.iter_data_list
        if history_option != 'full':
            iterations = [iterations[0], iterations[-1]] if len(iterations) > 1 else iterations
        data[name] = dict(initial_guess=phase.initial_guess, iter_data_list=iterations,
            scalar_history=phase.state.scalar_trace if history_option != 'none' else [],
            mesh=dict(tau=phase.mesh.tau, discretize=phase.mesh.discretize, segments=phase.mesh.segments),
            hyperparams=phase.hyperparams,
            constraints={key: dict(name=c.name, type=c.type, shape=c.shape,
                penalties=penalty_descriptor(c.penalties))
                for key, c in phase.constraints.items()})
        it = phase.current_iter_data
        diagnostics[name] = dict(**objective_terms(phase), reported_cost=float(it.get('cost', 0.)),
            penalty_cost=float(it.get('penalty_cost', 0.)), stopping_ratios=it.get('chk', {}),
            dynamics_defect_nd=it.get('defect'),
            constraint_residuals_nd={key: c.g_nl for key, c in phase.constraints.items()
                                     if getattr(c, 'g_nl', None) is not None})
    diagnostics['couplings'] = {f'{c.spec.target}.{c.spec.name}': c.residual()
                                for c in solver.prepared.couplings}
    attempts = tuple(solver.state.attempts)
    timings = dict(solver._timings,
        numerical_s=sum(a.timings.get('numerical_s', 0.) for a in attempts),
        canonicalization_s=sum(a.timings.get('canonicalization_s', 0.) for a in attempts),
        local_model_s=sum(sum(a.timings.get('local_model_s', {}).values()) for a in attempts))
    initial = solver.initial_trajectory
    try:
        trajectory = export_set(solver.phases) if solver.state.iterations else None
    except ValueError as exc:
        trajectory, failed = None, True
        reason = f'Unusable accepted trajectory (nonfinite values or invalid time map): {exc}'
    history = []
    if history_option == 'full':
        for index in range(1, min(len(s.iter_data_list) for s in solver.phases.values())):
            try:
                history.append(TrajectorySet({name: export_trajectory(s, data=s.iter_data_list[index])
                                              for name, s in solver.phases.items()}))
            except ValueError:
                diagnostics.setdefault('unavailable_history', []).append(index)
    status = ('failed' if failed or trajectory is None else
              'converged' if solver.state.converged else 'not_converged')
    return SolveResult(trajectory, initial, status, reason,
        attempts[-1].status if attempts else None,
        solver.state.iterations, attempts, timings, diagnostics,
        dict(algorithm='SCPMethod', prepared_signature=solver.prepared.signature,
             environment=collect_environment(),
             timing_scope='seconds since reset/resume; total_s excludes preparation; '
                          'last_call_s is the latest solve call; component timings overlap total_s'),
        tuple(history), solver.problem, solver.method_spec, data)


@dataclass
class _SolveLog:
    started: float
    reason: str = 'Attempt budget exhausted'


@dataclass
class _IterationLog:
    accepted: bool = False
    reason: str = ''
    alpha: float | None = None


@contextmanager
def record_solve(solver, wall_time_s):
    """Measure the solve and save its result. Save the result if an error occurs."""
    run = _SolveLog(perf_counter())
    prior_elapsed = solver._timings.get('total_s', 0.)
    solver._timings['last_call_warmup_s'] = 0.
    solver._terminal_reason = None
    solver._in_solve = True
    try:
        yield run
    except Exception as error:
        run.reason = f'{type(error).__name__}: {error}'
        solver._terminal_reason = run.reason
        raise
    finally:
        elapsed = perf_counter() - run.started
        solver._in_solve = False
        solver._timings.update(total_s=prior_elapsed + elapsed, last_call_s=elapsed,
                               preparation_s=solver.prepared.preparation_s,
                               deadline_overrun_s=max(0., elapsed - wall_time_s))
        solver.result = make_result(solver, run.reason, failed=solver._terminal_reason is not None)
        solver.reporter.footer(reason=run.reason, total_ms=elapsed * 1000.,
            disc_ms=solver.result.timings['local_model_s'] * 1000.,
            solve_ms=solver.result.timings['numerical_s'] * 1000.)
        solver.reporter.trajectory_summary([
            (name, phase.current_iter_data.t_start, phase.current_iter_data.t_final)
            for name, phase in solver.phases.items()])


@contextmanager
def record_iteration(solver):
    """Record one trial, including a rejected trial or a failed update."""
    entry = _IterationLog()
    started = perf_counter()
    solver.last_trial = None
    failed = False
    try:
        yield entry
    except Exception as error:
        failed = True
        entry.reason = solver._terminal_reason or f'{type(error).__name__}: {error}'
        solver._terminal_reason = f'{type(error).__name__}: {error}'
        raise
    finally:
        trial = solver.last_trial
        timings = {} if trial is None else dict(trial.timings)
        timings['total_s'] = perf_counter() - started
        diagnostics = {}
        if trial is not None and not failed:
            timings['local_model_s'] = {
                name: float(phase.current_iter_data.discretization_time) / 1000.
                for name, phase in solver.phases.items()}
            diagnostics['candidate_step_norms'] = {
                name: {key: None if value is None else float(np.max(np.abs(value)))
                       for key, value in delta.items()}
                for name, delta in trial.deltas.items()}
        record = AttemptRecord(len(solver.state.attempts) + 1,
            None if trial is None else trial.status, entry.accepted,
            solver.state.iterations, entry.reason, entry.alpha, timings, diagnostics)
        solver.state.attempts.append(record)
        if entry.reason:
            solver.reporter.message(entry.reason)
        if entry.accepted:
            solver.display_status()
