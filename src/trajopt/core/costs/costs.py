import inspect
from functools import partial

from trajopt.core.costs import cost_types


class Costs:
    def __init__(self, config, index_map, fcns=None):

        self.cost_list = []
        self.cost_type_list = []

        print("\ncosts:")

        # build cost_ids mapping
        for i, (cost_name, cost_config_i) in enumerate(config.problem.costs.items()):
            print(f"  {i}: {cost_name}: {cost_name}")
            self.register_cost(cost_config_i, index_map, fcns=fcns)

    def register_cost(self, cost_config, index_map, fcns=None):
        """Register a cost object in the costs list given a cost configuration.

        Args:
            cost_config: Cost configuration dictionary.
            index_map: Index map object.
            fcns: Resolved functions dictionary.

        Returns:
            None.

        """
        cost_type = cost_config["type"]
        costClass = getattr(cost_types, cost_type)
        self.cost_list.append(costClass(cost_config, index_map, fcns=fcns))
        if cost_type not in self.cost_type_list:
            self.cost_type_list.append(cost_type)

    def get(self, **kwargs):
        """Get all costs that match given keyword arguments.

        Args:
            **kwargs: Keyword arguments to match against cost attributes.

        Returns:
            List of costs that match the given keyword arguments.

        """
        selected_costs = [cost for cost in self.cost_list if all(getattr(cost, k, None) == v for k, v in kwargs.items())]
        return selected_costs

    def has(self, **kwargs):
        """Check if any costs match given keyword arguments.

        Args:
            **kwargs: Keyword arguments to match against cost attributes.

        Returns:
            True if any costs match all given keyword arguments, False otherwise.

        """
        return any(all(getattr(cost, k, None) == v for k, v in kwargs.items()) for cost in self.cost_list)

    def resolve_functions(self, fcns):
        """Bind user-provided functions to cost objects and wrap 'fcns' dictionary.

        Args:
            fcns: Dictionary of user-provided functions.

        Returns:
            None.

        """
        for cost in self.cost_list:
            if getattr(cost, "fcn_dim", None) is not None:
                sig = inspect.signature(cost.fcn_dim)
                param_names = sig.parameters.keys()

                kwargs_to_bind = {}
                if "fcns" in param_names:
                    kwargs_to_bind = {"fcns": fcns}

                if kwargs_to_bind:
                    cost.fcn_dim = partial(cost.fcn_dim, **kwargs_to_bind)

    def nondim_costs(self, nondim):
        """Non-dimensionalize all costs.

        Args:
            nondim: Non-dimensionalization object.

        Returns:
            None.

        """
        for cost in self.cost_list:
            cost.nondim_cost(nondim)

    def convexify_costs(self):
        """Convexify all costs. If a cost has a 'convexify_cost' method, call it.

        Args:
            None.

        Returns:
            None.

        """
        for cost in self.cost_list:
            if getattr(cost, "convexify_cost", None) is not None:
                cost.convexify_cost()
