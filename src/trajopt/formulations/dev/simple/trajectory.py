from trajopt.formulations.common.problem import Problem


class Trajectory(Problem):
    """A single flat trajectory: one Problem, no sub-phases."""

    _KIND = "trajectory"
