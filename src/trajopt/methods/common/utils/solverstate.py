"""Save and restore the trajectory, iteration data, and penalty values."""
import copy
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np

from trajopt.common._snapshot import fingerprint, freeze, thaw
from trajopt.common.tools import recursive_attrdict
from trajopt.methods.common.initial_guess import dimensional_samples, segment_guesses
from trajopt.methods.common.penalties import (
    capture_penalty_state,
    install_penalty_state,
)
from trajopt.methods.common.utils.trajectory_conversion import export_set


@dataclass
class SegmentState:
    current: object = None
    history: list = field(default_factory=list)
    scalar_trace: list = field(default_factory=list)
    s_mult: float = 1.0


@dataclass
class SolverState:
    segments: Mapping
    attempts: list = field(default_factory=list)
    converged: bool = False
    consecutive_rejects: int = 0

    @property
    def iterations(self):
        return max((state.current.iter_num for state in self.segments.values()), default=0)


@dataclass(frozen=True)
class TrialStep:
    status: str | None
    deltas: Mapping
    timings: Mapping
    error: str | None = None

    def __post_init__(self):
        for name in ('deltas', 'timings'):
            object.__setattr__(self, name, freeze(getattr(self, name)))


@dataclass(frozen=True)
class AlgorithmCheckpoint:
    """Saved state for the same problem, inputs, mesh, and method.

    Check these settings before restoring the state. The saved state does
    not include the CVXPY solver's working data.
    """
    signature: str
    segments: Mapping
    attempts: tuple = ()
    converged: bool = False
    timings: Mapping = field(default_factory=dict)
    consecutive_rejects: int = 0

    def __post_init__(self):
        object.__setattr__(self, 'segments', freeze(self.segments))
        object.__setattr__(self, 'attempts', tuple(self.attempts))
        object.__setattr__(self, 'timings', freeze(self.timings))


def capture_segment(segment, *, owned_history=True):
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


def install_segment(segment, data):
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


@contextmanager
def restore_on_error(segments, state):
    """Restore all phases if the update fails."""
    before = {name: capture_segment(segment, owned_history=False)
              for name, segment in segments.items()}
    was_converged = state.converged
    try:
        yield
    except Exception:
        state.converged = was_converged
        for name, segment in segments.items():
            install_segment(segment, before[name])
        raise


def reset_solver(solver, *, initial_guess=None):
    """Set the initial guess and clear the iteration data."""
    supplied = segment_guesses(initial_guess, solver.segments)
    prepared, previous = {}, None
    for name, segment in solver.segments.items():
        value = segment.prepare_guess(supplied.get(name), previous)
        prepared[name] = value
        previous = dimensional_samples(segment.problem, value.z, value.nu)
    for name, segment in solver.segments.items():
        segment.install_initial_guess(prepared[name])
    for variable in solver.cp_subproblem.variables():
        variable.value = None
    solver.state.attempts.clear()
    solver.state.consecutive_rejects = 0
    solver.state.converged = False
    solver.result = solver.last_trial = None
    solver._terminal_reason = None
    solver._timings = {}
    solver._reference_inputs = fingerprint(solver.problem.config)
    solver.initial_trajectory = export_set(solver.segments, initial=True)


def save_checkpoint(solver):
    check_reference_inputs(solver)
    return AlgorithmCheckpoint(_checkpoint_signature(solver),
        {name: capture_segment(segment) for name, segment in solver.segments.items()},
        tuple(solver.state.attempts), solver.state.converged, solver._timings,
        consecutive_rejects=solver.state.consecutive_rejects)


def restore_checkpoint(solver, checkpoint):
    if not isinstance(checkpoint, AlgorithmCheckpoint):
        raise TypeError('resume requires an AlgorithmCheckpoint; use reset for a trajectory')
    if checkpoint.signature != _checkpoint_signature(solver):
        raise ValueError('Checkpoint is incompatible with this problem, inputs, mesh, or method')
    if set(checkpoint.segments) != set(solver.segments):
        raise ValueError('Checkpoint segment identities differ')
    data = recursive_attrdict(thaw(checkpoint.segments))
    for name, segment in solver.segments.items():
        current = data[name]['current']
        for field, shape in (('z_opt', segment.cp_params.z_ref.shape),
                             ('nu_opt', segment.cp_params.nu_ref.shape)):
            if np.shape(current[field]) != shape or not np.isfinite(current[field]).all():
                raise ValueError(f'Invalid checkpoint {name}.{field}')
    for name, segment in solver.segments.items():
        install_segment(segment, data[name])
    solver.state.attempts = list(checkpoint.attempts)
    solver.state.converged = checkpoint.converged
    solver.state.consecutive_rejects = checkpoint.consecutive_rejects
    solver._reference_inputs = fingerprint(solver.problem.config)
    solver._timings = thaw(checkpoint.timings)
    solver.result = solver.last_trial = None
    solver._terminal_reason = None
    solver.initial_trajectory = export_set(solver.segments, initial=True)


def check_reference_inputs(solver):
    if solver._reference_inputs != fingerprint(solver.problem.config):
        raise RuntimeError('Prepared inputs changed; reset the algorithm before solving or checkpointing')


def _checkpoint_signature(solver):
    from trajopt.methods.common.scp.transcribe import _without_display_flags
    return fingerprint((solver.prepared.signature, solver.problem.config,
                        _without_display_flags(solver.method_config)))
