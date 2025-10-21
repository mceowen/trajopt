import trajopt.utils.set_defaults           as defaults
import trajopt.utils.tools                  as tools
import trajopt.algorithm.initial_guess      as guess
import trajopt.algorithm.convergence        as convergence
import trajopt.algorithm.convexification    as convexify
import trajopt.utils.nondim                 as nondim
import trajopt.config.main_config           as main_config
import trajopt.utils.config_loader          as cfg

from trajopt.mission import Mission
from trajopt.model import Model
import numpy as np

# TEMP: REPLACE WITH PROBLEM CLASS
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

    params = cfg.load_params_from_example(f"trajopt.examples.{example_name}")
    params                  = defaults.set_params_constraint_default(params)

    # at this point, we should have the params['mission'] and params['model'] dictionaries

    problem['mission']    = Mission(problem)
    problem['model']      = Model(problem)

    mission               = problem['mission']
    model                 = problem['model']

    # TEMP: REPLACE WITH METHOD CLASS
    params = main_config.method_params(params)

    # Default state/control bounds
    problem               = defaults.set_problem_default(problem)
    
    # Cost function
    problem["cost"]       = mission.cost
    problem['lin_cost']   = mission.lin_cost

    # dynamics
    problem["xdot"]       = model.dynamics
    problem['lin_dyn']    = model.lin_dyn

    # Nonconvex inequality constraints
    problem["path_lim"]   = params["path_lim"]

    problem["P"]          = model.nonlinear_inequality_constraints
    problem['lin_constr'] = model.lin_constr

    # Algorithm - custom formulation
    problem["custom_inputs"]        = mission.custom_inputs
    problem["custom_variables"]     = mission.custom_subprob_variables
    problem["custom_constraints"]   = mission.custom_subprob_constraints
    problem["custom_cost"]          = mission.custom_subprob_cost

    
    # TEMP: this would go in method and take in the model 

    #======================================
    # Initialize trajectory (initial guess)
    #======================================
    if params['bools']['free_final_time'] and (params['bools'].get('buff_dyn')=='term'):
        us_range = np.ones((2, 1)) @ ((-params['ge'].reshape(1, -1) * params['mass'])+ np.array([0.08, 0.08, 0.0])) / params['nondim']['nf']
        
        # need to manually set the left-hand side vector to a column vector for multiplacation to work
        params = guess.nonlinear_initial_guess(us_range, problem)

    else:
        params = guess.straight_line_initial_guess(params) 
        params['us_init'] =  np.tile(-params['ge'] * params['mass'], (params['N'], 1)) / params['nondim']['nf']

    if params['bools']['ctcs']:
        params = guess.ctcs_initial_guess(params)

    problem["cost_init"]  = problem["cost"](params["ts_init"], params["zs_init"], params["us_init"])

    problem["ts_init"] = params["ts_init"]
    problem["zs_init"] = params["zs_init"]
    problem["us_init"] = params["us_init"]

    return problem

# class Problem:
    
#     def __init__(self, config):

#         self.mission = Mission(config)
#         self.model = Model(config, self.mission)
#         self.method = Method(config, self.model)

