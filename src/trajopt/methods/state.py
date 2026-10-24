"""Algorithm state and checkpoints for resuming a compatible solve."""
from collections.abc import Mapping
from dataclasses import dataclass, field

from trajopt.utils._snapshot import freeze


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
    """Algorithm state for a matching problem, inputs, mesh, and method.

    The driver checks compatibility before it restores the state. The
    checkpoint does not contain the backend workspace.
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
