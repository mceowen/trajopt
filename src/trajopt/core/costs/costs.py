import trajopt.core.costs.costs_library as costs_library
from pprint import pprint
import inspect
from functools import partial
import trajopt.library.methods.convexify as convexify
import trajopt.utils.tools as tools

class Costs:
    def __init__(self, cost_config_list, params):

        self.costs_list = []
        self.cost_ids = {'all': []}

        print(f"costs:")

        # build cost_ids mapping
        for i, cost_config in enumerate(cost_config_list):
            cost_type = cost_config["type"]
            cost_name = cost_config["name"]

            print(f"  {i}: {cost_name}: {cost_type}")
            cost_params = {k:v for k, v in cost_config.items() if k != "type"}
            costClass = getattr(costs_library, cost_type)
            self.costs_list.append(costClass(**cost_params, params=params))

            if cost_type not in self.cost_ids:
                self.cost_ids[cost_type] = []

            
            self.cost_ids[cost_type].append(i)
            self.cost_ids['all'].append(i)

        # print("cost_ids: \n")
        # pprint(self.cost_ids)

    def get(self, cost_type):
        
        cost_ids = self.cost_ids.get(cost_type, [])
        costs = [self.costs_list[i] for i in cost_ids]

        return costs

    def has(self, cost_type):

        return cost_type in self.cost_ids.keys()

    def add_params(self, problem_params):

        for cost in self.costs_list:
            if "params" in cost.__dict__:
                if cost.params is not None:
                    problem_params = tools.deep_update(problem_params, cost.params)

    def resolve_functions(self, params, fcns):
        for cost in self.costs_list:
            # Check fcn_dim (the raw function) since fcn may be None until nondim wrapping
            if getattr(cost, 'fcn_dim', None) is not None:
                sig = inspect.signature(cost.fcn_dim)
                param_names = sig.parameters.keys()

                kwargs_to_bind = {}
                if 'params' in param_names:
                    kwargs_to_bind['params'] = params
                if 'fcns' in param_names:
                    kwargs_to_bind['fcns'] = fcns

                if kwargs_to_bind:
                    cost.fcn_dim = partial(cost.fcn_dim, **kwargs_to_bind)

    def nondim_costs(self, nondim):
        # apply scaling to each cost so that they are nondim
        # by the time it gets to the discretization and solver
        for cost in self.costs_list:
            cost.nondim_cost(nondim)

        print("costs nondimmed!")

    def convexify_costs(self):
        for cost in self.costs_list:
            if getattr(cost, 'fcn', None) is not None:
                cost.fcn_jit, cost.dfcn_dz_jit, cost.dfcn_du_jit = convexify.linearize_jax(cost.fcn)

        print("costs convexified!")