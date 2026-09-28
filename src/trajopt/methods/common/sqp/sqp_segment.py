from trajopt.methods.common.scp.segment import SCPSegment
from trajopt.methods.common.sqp.sqp_subproblem import SQPSubproblem


class SQPSegment(SQPSubproblem, SCPSegment):
    """SQPSubproblem for one phase of a multi-phase trajectory."""
