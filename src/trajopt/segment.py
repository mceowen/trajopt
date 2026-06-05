
from trajopt.index_map import IndexMap
from trajopt.nondim import Nondim
import trajopt.constraints.constraint_types as constraint_type_module
import trajopt.costs.cost_types as cost_type_module
import trajopt.trajplots.trajplot_types as trajplot_type_module
from trajopt.utils.tools import AttrDict, resolve_function_from_string

class Segment:

    def __init__(self, name: str, segment_config: AttrDict) -> None:

        self.name = name
        self.num_nodes = segment_config.num_nodes

        self.index_map = IndexMap(segment_config)
        self.nondim    = Nondim(segment_config, self.index_map)

        self.fcns  = AttrDict()
        for fcn_name, path in segment_config.fcns.items():
            self.fcns[fcn_name] = resolve_function_from_string(path)

        self.params = segment_config.params
        self.guess  = segment_config.get("guess", AttrDict())

        print(f"segment '{self.name}' configuration:")
        print("------------------------------------------------------------")

        self.constraints = AttrDict()
        for cnstr_name, cnstr_config in segment_config.constraints.items():
            cnstr_config.name = cnstr_name
            cnstr_type = cnstr_config.type
            constraintClass = getattr(constraint_type_module, cnstr_type)
            self.constraints[cnstr_name] = constraintClass(cnstr_config, self)

        self.costs = AttrDict()
        for cost_name, cost_config in segment_config.get("costs", AttrDict()).items():
            cost_config.name = cost_name
            cost_type = cost_config.type
            costClass = getattr(cost_type_module, cost_type)
            self.costs[cost_name] = costClass(cost_config, self)

        self.trajplots = AttrDict()
        for trajplot_name, trajplot_config in segment_config.get("trajplots", AttrDict()).items():
            trajplot_config.name = trajplot_name
            trajplot_type = trajplot_config.type
            trajplotClass = getattr(trajplot_type_module, trajplot_type)
            self.trajplots[trajplot_name] = trajplotClass(trajplot_config, self)

        print("------------------------------------------------------------")
        print("\n")
