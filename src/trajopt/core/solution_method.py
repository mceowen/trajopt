import trajopt.methods.scp.scp as scp
from trajopt.core.scaling.nondim import Nondim
from trajopt.utils.tools import AttrDict

class SolutionMethod:
    def __init__(self, problem, config, index_map):

        # ===============================================================
        # load config params
        # ===============================================================
        self.problem       = problem
        self.index_map     = problem.index_map

        self.flags         = config.method.flags
        self.initial_guess = config.method.guess
        self.conv          = config.method.conv
        self.penalty       = config.method.weights
        self.solver_opts   = config.method.solver_opts

        self.conv_data     = AttrDict()

        # update index_map
        self.index_map.update_index_map(problem=self.problem, method=self)
        self.nondim = Nondim(problem)

        self.initial_guess.x = None
        self.initial_guess.u = None
        self.initial_guess.z = None
        self.initial_guess.nu = None

    def initialize(self):
        scp.initialize(self)

    def solve(self):
        results = scp.run_scp(self)
        return results