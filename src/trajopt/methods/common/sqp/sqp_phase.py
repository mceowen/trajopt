from trajopt.methods.common.scp.phase import SCPPhase
from trajopt.methods.common.sqp.sqp_subproblem import SQPSubproblem


class SQPPhase(SQPSubproblem, SCPPhase):
    """SQPSubproblem for one phase of a multi-phase trajectory."""
