from trajopt.formulations.common.problem import Problem


class Trajectory(Problem):
    """A single flat trajectory: one Problem, no sub-segments."""

    _KIND = "trajectory"
