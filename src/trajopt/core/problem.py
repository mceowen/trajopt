from trajopt.core.constraints.constraints import Constraints
from trajopt.core.costs.costs import Costs
from trajopt.core.trajectories.trajectories import Trajectories
from trajopt.utils.tools import AttrDict, recursive_attrdict, resolve_function_from_string

class Problem:

    def __init__(self, config, index_map):

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
            try:
                self.fcns[name] = resolve_function_from_string(path)
            except Exception as e:
                raise type(e)(
                    f"error loading fcns.{name}: '{path}'\n"
                    f"  {e}"
                ) from None

        # ------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------

        self.params = recursive_attrdict(config.problem.params)

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