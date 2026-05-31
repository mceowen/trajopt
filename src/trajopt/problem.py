from trajopt.constraints.constraints import Constraints
from trajopt.costs.costs import Costs
from trajopt.trajectories.trajectories import Trajectories
from trajopt.indexing.index_map import IndexMap
from trajopt.nondim.nondim import Nondim
from trajopt.utils.tools import AttrDict, recursive_attrdict, resolve_function_from_string

class Problem:

    def __init__(self, config):

        # ------------------------------------------------------------
        # Config
        # ------------------------------------------------------------

        self.config    = config
        self.index_map = IndexMap(self.config)
        self.nondim    = Nondim(self.config, self.index_map)

        # ------------------------------------------------------------
        # Functions
        # ------------------------------------------------------------

        fcn_config = config.problem.fcns
        self.fcns = AttrDict()
        
        for name, path in fcn_config.items():
            self.fcns[name] = resolve_function_from_string(path)

        # ------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------

        self.params = recursive_attrdict(config.problem.params)

        print("problem configuration: ")
        print("------------------------------------------------------------")
        # ------------------------------------------------------------
        # Constraints
        # ------------------------------------------------------------

        self.constraints = Constraints(self.config, self.index_map, fcns=self.fcns)

        # ------------------------------------------------------------
        # Cost
        # ------------------------------------------------------------

        self.costs = Costs(self.config, self.index_map, fcns=self.fcns)
        print("------------------------------------------------------------")
        print("\n")

        # ------------------------------------------------------------
        # Bind fcns dict to constraint/cost functions that accept it
        # ------------------------------------------------------------

        self.constraints.resolve_functions(self.fcns)
        self.costs.resolve_functions(self.fcns)

        # ------------------------------------------------------------
        # Trajectories (similar to constraints but used for analysis)
        # ------------------------------------------------------------
        self.trajectories = Trajectories(self.config, self.index_map, fcns=self.fcns)
        self.trajectories.resolve_functions(self.fcns)

        # nondimensionalize constraints
        self.constraints.nondim_constraints(self.nondim)

        # finalize the augmented z layout and dynamics now that every constraint knows its dimension
        self.constraints.augment()

        # nondimensionalize and convexify costs
        self.costs.nondim_costs(self.nondim)
        self.costs.convexify_costs()