import numpy as np

import trajopt; import importlib; importlib.reload(trajopt)

from trajopt.core.Problem   import Problem
from trajopt.core.Method    import Method
from trajopt.core.modules.method.indices import Indices  


class Trajopt:
    def __init__(self, config, subprob=None):

        # example
        self.name       = 'name'

        self.problem    = Problem(config)
        self.indices    = Indices(self.problem)
        self.method     = Method(self.problem, config)