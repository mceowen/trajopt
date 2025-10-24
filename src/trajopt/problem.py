import trajopt.algorithm.initial_guess      as guess
import trajopt.utils.config_loader          as cfg

from trajopt.mission import Mission
from trajopt.model import Model
import numpy as np

# new problem class structure (current version is using the ocp dictionary below)
class Problem:
    def __init__(self, example_name):
        self.params = cfg.load_params(example_name)
        self.params["case_flag"] = 1

        self.mission = Mission(self)
        self.model   = Model(self)
        # self.method  = Method(self)
        
# old ocp problem dictionary (still in use)
def ocp(example_name):
    """
    Define the optimal control problem (OCP):
    - cost
    - constraints
    - dynamics

    Parameters:
        config : configuration dictionary or object

    Returns:
        problem : dictionary describing the OCP
    """
    problem = {}

    params = cfg.load_params(example_name)
    
    params["case_flag"] = 1    # case1
    problem["params"] = params

    # at this point, we should have the params["mission"] and params["model"] dictionaries

    problem["mission"]    = Mission(problem)
    problem["model"]      = Model(problem)

    # Cost function
    problem["cost"]       = problem["mission"].cost
    problem["lin_cost"]   = problem["mission"].lin_cost

    # dynamics
    problem["xdot"]       = problem["model"].dynamics
    problem["lin_dyn"]    = problem["model"].lin_dyn

    problem["P"]          = problem["model"].nonlinear_inequality_constraints
    problem["lin_constr"] = problem["model"].lin_constr

    # Algorithm - custom formulation
    problem["custom_inputs"]        = problem["mission"].custom_inputs
    problem["custom_variables"]     = problem["mission"].custom_subprob_variables
    problem["custom_constraints"]   = problem["mission"].custom_subprob_constraints
    problem["custom_cost"]          = problem["mission"].custom_subprob_cost

    # TEMP: this would go in method class and configs
    #======================================
    # Initialize trajectory (initial guess)
    #======================================
    if params["method"]["bools"]["free_final_time"] and (params["method"]["bools"].get("buff_dyn")=="term"):
        us_range = np.ones((2, 1)) @ ((-params["mission"]["ge"].reshape(1, -1) * params["mission"]["mass"])+ np.array([0.08, 0.08, 0.0])) / params["method"]["nondim"]["nf"]
        
        # need to manually set the left-hand side vector to a column vector for multiplacation to work
        params = guess.nonlinear_initial_guess(us_range, problem)

    else:
        params = guess.straight_line_initial_guess(params) 
        params["us_init"] =  np.tile(-params["mission"]["ge"] * params["mission"]["mass"], (params["method"]["N"], 1)) / params["method"]["nondim"]["nf"]

    if params["method"]["bools"]["ctcs"]:
        params = guess.ctcs_initial_guess(params)

    problem["cost_init"]  = problem["cost"](params["method"]["ts_init"], params["method"]["zs_init"], params["method"]["us_init"])

    return problem

