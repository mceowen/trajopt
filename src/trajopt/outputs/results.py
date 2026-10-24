"""Store solve results separately from the solver's mutable data."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from trajopt.utils._snapshot import freeze
from trajopt.trajectories import TrajectorySet

if TYPE_CHECKING:
    from trajopt.formulations.specs import ProblemSpec
    from trajopt.methods.specs import MethodSpec


@dataclass(frozen=True)
class AttemptRecord:
    number: int
    status: str | None
    accepted: bool
    iteration: int
    reason: str = ''
    alpha: float | None = None
    timings: Mapping = field(default_factory=dict)
    diagnostics: Mapping = field(default_factory=dict)

    def __post_init__(self):
        for name in ('timings', 'diagnostics'):
            object.__setattr__(self, name, freeze(getattr(self, name)))


@dataclass(frozen=True)
class SolveResult:
    """Solve status and any accepted trajectory.

    The initial guess is separate from the accepted trajectory. All public
    timing values are in seconds. Component timings can overlap with elapsed
    timings. Do not add them to ``total_s``.
    """
    trajectory: TrajectorySet | None
    initial_trajectory: TrajectorySet
    status: str
    reason: str
    solver_status: str | None
    iterations: int
    attempts: tuple[AttemptRecord, ...] = ()
    timings: Mapping = field(default_factory=dict)
    diagnostics: Mapping = field(default_factory=dict)
    provenance: Mapping = field(default_factory=dict)
    history: tuple[TrajectorySet, ...] = ()
    problem: ProblemSpec | None = None
    method: MethodSpec | None = None
    segment_data: Mapping = field(default_factory=dict)

    def __post_init__(self):
        if self.status not in {'converged', 'not_converged', 'failed'}:
            raise ValueError(f'Unknown solve status {self.status!r}')
        if self.iterations == 0 and self.trajectory is not None:
            raise ValueError('No accepted iteration: keep the trajectory in initial_trajectory')
        for name in ('timings', 'diagnostics', 'provenance', 'segment_data'):
            object.__setattr__(self, name, freeze(getattr(self, name)))
        object.__setattr__(self, 'attempts', tuple(self.attempts))
        object.__setattr__(self, 'history', tuple(self.history))

    @property
    def converged(self):
        return self.status == 'converged'

    @property
    def usable(self):
        return self.trajectory is not None and self.iterations > 0

    @property
    def rejected_attempts(self):
        return sum(not attempt.accepted for attempt in self.attempts)
