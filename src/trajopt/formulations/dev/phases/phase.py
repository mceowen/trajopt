from trajopt.formulations.common.problem import Problem
from trajopt.utils.tools import AttrDict


class Phase(Problem):
    """One phase of a multi-phase trajectory: a Problem named by its position in the sequence."""

    _KIND = "phase"

    def __init__(self, name: str, phase_config: AttrDict) -> None:
        super().__init__(phase_config, name=name)
