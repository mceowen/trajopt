import trajopt.core.modules.model.costs_library as costs_library
from pprint import pprint

class Costs:
    def __init__(self, cost_config_list):

        self.costs_list = []
        self.cost_ids = {'all': []}

        print(f"costs:")

        # build constraint_ids mapping
        for i, cost_config in enumerate(cost_config_list):
            cost_type = cost_config["type"]
            print(f"  {i}: {cost_type}")
            cost_params = {k:v for k, v in cost_config.items() if k != "type"}
            costClass = getattr(costs_library, cost_type)
            self.costs_list.append(costClass(**cost_params))

            if cost_type not in self.cost_ids:
                self.cost_ids[cost_type] = []

            
            self.cost_ids[cost_type].append(i)
            self.cost_ids['all'].append(i)

        print("cost_ids: \n")
        pprint(self.cost_ids)

    def get(self, cost_type):
        
        cost_ids = self.cost_ids.get(cost_type, [])
        costs = [self.costs_list[i] for i in cost_ids]

        return costs

    def has(self, cost_type):

        return cost_type in self.cost_ids.keys()