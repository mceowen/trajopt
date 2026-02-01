from trajopt.core.constraints.constraints import Constraints
from trajopt.core.costs.costs import Costs
from trajopt.utils.config_loader import resolve_function

# ████████████████████████████████████████████████████████████████████████████

# TODO:
# need to remove constriant bookkeeping section (54-79)
# i kept it there for compatibility with the rest of the code

# ████████████████████████████████████████████████████████████████████████████

class Problem:

    def __init__(self, problem_config):

        # ------------------------------------------------------------
        # Config
        # ------------------------------------------------------------

        self.config = problem_config['config']
        self.n = self.config['model']['dimensions']['n']
        self.m = self.config['model']['dimensions']['m']

        # ------------------------------------------------------------
        # Functions
        # ------------------------------------------------------------
        
        fcns_config = self.config['mission'].pop('fcns', {})
        
        self.fcns = {}
        for name, path in fcns_config.items():
            self.fcns[name] = resolve_function(path)

        # ------------------------------------------------------------
        # Parameters (these can be varied during montecarlo)
        # ------------------------------------------------------------

        self.params = problem_config['params']

        # ------------------------------------------------------------
        # Constraints
        # ------------------------------------------------------------

        constraint_config_list = problem_config["constraints"]
        self.constraints = Constraints(constraint_config_list, self.config, self.params)

        # ------------------------------------------------------------
        # Cost
        # ------------------------------------------------------------

        cost_config_list = problem_config["costs"]
        self.costs = Costs(cost_config_list, self.config)

        # ------------------------------------------------------------
        # Add constraint/cost config and resolve functions
        # ------------------------------------------------------------
        
        self.constraints.add_params(self.params)
        self.costs.add_params(self.params)

        self.constraints.resolve_functions(self.params, self.fcns)
        self.costs.resolve_functions(self.params, self.fcns)

        # ------------------------------------------------------------
        # CONSTRAINT BOOK KEEPING
        # TODO: MOVE THIS TO INDEX_MAP AND DONT HARDCODE GROUPS
        # ------------------------------------------------------------

        # constraint book keeping
        self.n_ineq = sum(constraint.dimension for constraint in self.constraints.get('nodal', 'nonconvex_inequality'))

        # TODO: should the algorithm need to distinguish between path, nfz, and custom, can we collapse into n_ineq?
        # TODO: ADD this to constraints class lol, ideally, shouldn't need any loops
        self.n_path = sum(constraint.dimension for constraint in self.constraints.get('nodal', 'nonconvex_inequality') if constraint.group == "path")
        self.n_nfz = sum(constraint.dimension for constraint in self.constraints.get('nodal', 'nonconvex_inequality') if constraint.group == "nfz")
        self.n_custom = sum(constraint.dimension for constraint in self.constraints.get('nodal', 'nonconvex_inequality') if constraint.group == "custom")

        if self.constraints.has('ct'):
            self.n_ctcs = sum(constraint.dimension for constraint in self.constraints.get('ct', 'all'))
        else:
            self.n_ctcs = 0

        self.nz = self.n + self.n_ctcs

        # TODO: same here
        self.n_term = sum(constraint.dimension for constraint in self.constraints.get('nodal', 'equality_bc') if constraint.boundary == "final" and constraint.set == "state")
        self.n_term_ineq = sum(constraint.dimension for constraint in self.constraints.get('nodal', 'inequality_bc') if constraint.boundary == "final" and constraint.set == "state")
        self.n_term_ctcs = self.n_ctcs
        self.n_term_total = self.n_term + self.n_term_ineq + self.n_ctcs