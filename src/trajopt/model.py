
import numpy as np
import importlib
from typing import Dict, Any, Optional, List

import numpy as np
import importlib

import trajopt.utils.set_defaults           as defaults
import trajopt.utils.tools                  as tools
import trajopt.algorithm.initial_guess      as guess
import trajopt.algorithm.convergence        as convergence
import trajopt.algorithm.convexification    as convexify
import trajopt.utils.nondim                 as nondim

class Model:

    def __init__(self, problem):
        
        self.params = problem['params']

        # point to selected model module
        model_name = self.params['model_name']
        model_module = importlib.import_module(f"trajopt.models.{model_name}")

        # set dynamics
        self._dynamics = model_module.system_dynamics

        # set ltv dynamics
        if self.params["bools"]["auto_jac"]:
            self._lin_dyn = convexify.generate_jacobians(self.dynamics)
        else:
            self._lin_dyn = model_module.analytical_linsys

        # set nonlinear inequality constraints
        self._nonlinear_inequality_constraints = model_module.nonlinear_inequality_constraints

        # set linearized constraints
        if self.params["bools"]["auto_jac_cnst"]:
            self._lin_constr = convexify.generate_jacobians(self.nonlinear_inequality_constraints)
        else:
            self._lin_constr = model_module.analytical_inequality_constraints

    #===============================================================
    # member functions point to selected fcns from selected module
    #===============================================================

    def dynamics(self, ts, zs, us, t_vec=None):
        return self._dynamics(ts, zs, us, self, t_vec)
    
    def lin_dyn(self, ts, zs, us):
        return self._lin_dyn(ts, zs, us, self)
    
    def nonlinear_inequality_constraints(self, ts, zs, us):
        return self._nonlinear_inequality_constraints(ts, zs, us, self)
    
    def lin_constr(self, ts, zs, us):
        return self._lin_constr(ts, zs, us, self)