"""Results of sequential trajectory optimization stages."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generic, TypeVar
from uuid import uuid4

from trajopt.trajectories import TrajectorySet

T = TypeVar('T')


@dataclass
class StageResult(Generic[T]):
    name: str
    config_path: Path
    config: dict
    status: str = 'failed'
    reason: str = ''
    solver_status: str | None = None
    iterations: int = 0
    trajectory: TrajectorySet | dict = field(default_factory=dict)
    layouts: dict = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    seed_from: str | None = None
    transferred_phases: tuple[str, ...] = ()
    output: T | None = None
    invocation_id: str = field(default_factory=lambda: uuid4().hex)
    execution_status: str = 'failed'

    @property
    def converged(self) -> bool:
        return self.status == 'converged'


@dataclass
class FlowResult:
    stages: dict[str, StageResult] = field(default_factory=dict)
    completed: bool = False
    failed_stage: str | None = None
    elapsed_s: float = 0.
    selected_stage: str | None = None

    @property
    def selected(self):
        return self.stages.get(self.selected_stage)

    @property
    def final(self) -> StageResult | None:
        return next(reversed(self.stages.values())) if self.stages else None

    @property
    def converged(self) -> bool:
        """The final stage converged and all preceding stages permitted continuation."""
        return self.completed and self.final is not None and self.final.converged
