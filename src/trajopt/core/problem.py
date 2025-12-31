from typing import Dict, Any, Optional, List
from pprint import pprint
import numpy as np
import importlib

import trajopt.core.methods.convexify as convexify
from trajopt.core.constraints import Constraints
from trajopt.core.costs import Costs
import trajopt.core.utils.tools as tools

# ████████████████████████████████████████████████████████████████████████████

# TODO: STILL UNDER CONSTRUCTION, RUNS AND "CONVERGES" for:
# examples/lander_6dof/standalone_prototype.ipynb

# ████████████████████████████████████████████████████████████████████████████

class Problem:

    def __init__(self, problem_config):

        # params already built by load_trajopt (includes model, mission, constraint params)
        self.params = problem_config['params']
        self.fcns = {}

        self.n = self.params['model']['dimensions']['n']
        self.m = self.params['model']['dimensions']['m']

        # ████████████████████████████████████████████████████████████████████████████
        # █                                                                          █
        # █                         O C P   D E F I N I T I O N                      █
        # █                                                                          █
        # ████████████████████████████████████████████████████████████████████████████

        # ------------------------------------------------------------
        # Constraints
        # ------------------------------------------------------------

        constraint_config_list = problem_config["constraints"]
        self.constraints = Constraints(constraint_config_list)

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

        # ------------------------------------------------------------
        # Cost
        # ------------------------------------------------------------

        cost_config_list = problem_config["costs"]
        self.costs = Costs(cost_config_list)

        self.constraints.resolve_functions(self.params, self.fcns)
        self.costs.resolve_functions(self.params, self.fcns)

        # ------------------------------------------------------------
        # Augment CTCS dynamics
        # ------------------------------------------------------------

        if self.constraints.has('ct'):
            dynamics = self.constraints.get('dynamics').fcn
            f_ctcs, dfcn_dz_ctcs, dfcn_du_ctcs = convexify.linearize_jax_ctcs(dynamics, self.constraints, self.n)

            lin_dyn_ctcs = lambda t, z, nu: (f_ctcs(z, nu), dfcn_dz_ctcs(z, nu), dfcn_du_ctcs(z, nu))

            self.lin_dyn = lin_dyn_ctcs

        for constraint in self.constraints.constraints_list:
            if "params" in constraint.__dict__:
                if constraint.params is not None:
                    self.params = tools.deep_update(self.params, constraint.params)

        print("\n")
        print("constraints config:")
        pprint(problem_config["constraints"])
        print("\n")

        print("\n")
        print("costs config:")
        pprint(problem_config["costs"])
        print("\n")