import numpy as np
from trajopt.core.constraints.constraints import Constraints


class SubproblemConstraints(Constraints):
    """
    Constraints used by the SCP subproblem.
    Identical to Problem.constraints, with SCP-specific data added.
    Assumes `problem.constraints.config_list` exists.
    """

    def __init__(self, problem, method):
        super().__init__(problem.constraints.config_list, problem.params)

        # Mirror Problem initialization
        self.add_params(problem.params)
        self.resolve_functions(problem.params, problem.fcns)

        # # Initialize per-constraint SCP data
        # for c in self.all:
        #     d = int(c.dimension)

        #     c.W    = np.zeros(d)
        #     c.dual = np.zeros(d)
        #     c.vb   = np.zeros(d)

        breakpoint()