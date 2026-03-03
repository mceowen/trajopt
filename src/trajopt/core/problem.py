import numpy as np
from trajopt.core.constraints.constraints import Constraints
from trajopt.core.costs.costs import Costs
from trajopt.utils.config_loader import resolve_function
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
            self.fcns[name] = resolve_function(path)

        # ------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------

        self.params = config.problem.params

        # ------------------------------------------------------------
        # Constraints
        # ------------------------------------------------------------

        self.constraints = Constraints(self.config)

        # ------------------------------------------------------------
        # Cost
        # ------------------------------------------------------------

        self.costs = Costs(self.config)

        # ------------------------------------------------------------
        # Add constraint/cost config and resolve functions
        # ------------------------------------------------------------

        self.constraints.resolve_functions(self.fcns)
        self.costs.resolve_functions(self.fcns)
        # self.costs.make_epigraph_constraints()