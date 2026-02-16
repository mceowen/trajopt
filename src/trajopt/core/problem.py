import numpy as np
from trajopt.core.constraints.constraints import Constraints
from trajopt.core.costs.costs import Costs
from trajopt.utils.config_loader import resolve_function

# ████████████████████████████████████████████████████████████████████████████

# TODO:
# need to remove constriant bookkeeping section (54-79)
# i kept it there for compatibility with the rest of the code

# ████████████████████████████████████████████████████████████████████████████

class Problem:

    def __init__(self, problem_config, index_map=None):

        # ------------------------------------------------------------
        # Config
        # ------------------------------------------------------------

        from trajopt.utils.tools import AttrDict
        self.config = AttrDict(problem_config['config'])
        self.index_map = index_map

        # ------------------------------------------------------------
        # Functions
        # ------------------------------------------------------------
        
        fcns_config = self.config.mission.pop('fcns', {})
        self.fcns = AttrDict()
        for name, path in fcns_config.items():
            self.fcns[name] = resolve_function(path)

        # ------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------

        self.params = AttrDict(problem_config['params'])

        # ------------------------------------------------------------
        # Constraints
        # ------------------------------------------------------------

        constraint_config_list = problem_config["constraints"]
        self.constraints = Constraints(constraint_config_list, self.config)

        # ------------------------------------------------------------
        # Cost
        # ------------------------------------------------------------

        cost_config_list = problem_config["costs"]
        self.costs = Costs(cost_config_list, self.config)

        # ------------------------------------------------------------
        # Add constraint/cost config and resolve functions
        # ------------------------------------------------------------

        self.constraints.resolve_functions(self.fcns)
        self.costs.resolve_functions(self.fcns)
        # self.costs.make_epigraph_constraints()

        
    def update_from_config(self, varied_paths, nondim):
        mission_config = self.config['mission']
        for path in varied_paths:
            parts = path.split('.')
            if parts[0] == 'constraints' and len(parts) >= 3:
                name, field = parts[1], parts[2]
                val = mission_config.get('constraints', {}).get(name, {}).get(field)
                if val is not None:
                    for c in self.constraints.get():
                        if c.name == name:
                            setattr(c, field, np.atleast_1d(val))
                            c.nondim_constraint(nondim)
                            break
            
            elif parts[0] == 'params' and len(parts) >= 2:
                target = self.params
                source = mission_config.get('params', {})
                for part in parts[1:-1]:
                    target = target.setdefault(part, {})
                    source = source.get(part, {})
                if parts[-1] in source:
                    target[parts[-1]] = source[parts[-1]]