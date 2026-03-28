import numpy as np
from trajopt.core.constraints.constraints import Constraints
from trajopt.core.costs.costs import Costs
from trajopt.core.trajectories.trajectories import Trajectories
from trajopt.utils.config_loader import resolve_function_from_path
from trajopt.utils.tools import AttrDict

class Problem:

    def __init__(self, config, index_map=None):

        # ------------------------------------------------------------
        # Config
        # ------------------------------------------------------------

        self.config = config
        self.index_map = index_map

        # ------------------------------------------------------------
        # Functions
        # ------------------------------------------------------------

        fcn_config = config.problem.fcns
        self.fcns = AttrDict()
        
        for name, path in fcn_config.items():
            self.fcns[name] = resolve_function_from_path(path)

        # ------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------

        self.params = config.problem.params

        print("problem configuration: ")
        print("------------------------------------------------------------")
        # ------------------------------------------------------------
        # Constraints
        # ------------------------------------------------------------

        self.constraints = Constraints(self.config, index_map, fcns=self.fcns)

        # ------------------------------------------------------------
        # Cost
        # ------------------------------------------------------------

        self.costs = Costs(self.config, index_map, fcns=self.fcns)
        print("------------------------------------------------------------")

        # ------------------------------------------------------------
        # Bind fcns dict to constraint/cost functions that accept it
        # ------------------------------------------------------------

        self.constraints.resolve_functions(self.fcns)
        self.costs.resolve_functions(self.fcns)

        # ------------------------------------------------------------
        # Trajectories (similar to constraints but used for analysis)
        # ------------------------------------------------------------
        self.trajectories = Trajectories(self.config, index_map, fcns=self.fcns)
        self.trajectories.resolve_functions(self.fcns)