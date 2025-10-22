import trajopt.utils.set_defaults           as defaults
import trajopt.utils.tools                  as tools
import trajopt.algorithm.initial_guess      as guess
import trajopt.algorithm.convergence        as convergence
import trajopt.algorithm.convexification    as convexify
import trajopt.utils.nondim                 as nondim
import trajopt.temp_config.temp_config      as temp_config


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
    params = {}

    params["mission"] = {}
    params["model"] = {}
    params["method"] = {}

    # example configs
    example_pkg = f"trajopt.examples.{example_name}"
    example = {k: tools.load_yaml(example_pkg, f"{k}.yaml") for k in ["mission", "model", "method"]}

    example_mission_params = example["mission"]
    example_model_params   = example["model"]
    example_method_params  = example["method"]

    mission_name = example_mission_params["mission_name"]
    model_name   = example_model_params["model_name"]
    method_name  = example_method_params["method_name"]

    # base configs
    base_pkgs = {
        "mission": "trajopt.base_configs.missions",
        "model":  "trajopt.base_configs.models",
        "method": "trajopt.base_configs.methods",
    }

    base = {
        k: tools.load_yaml(base_pkgs[k], f"{name}.yaml")
        for k, name in {
            "mission": mission_name,
            "model":   model_name,
            "method":  method_name,
        }.items()
    }

    base_mission_params = base["mission"]
    base_model_params   = base["model"]
    base_method_params  = base["method"]

    # general default configs
    default = {k: tools.load_yaml(base_pkgs[k], "default.yaml") for k in base_pkgs}

    default_mission_params = default["mission"]
    default_model_params   = default["model"]
    default_method_params  = default["method"]

    params["mission"].update(default_mission_params)
    params["mission"] = tools.deep_update(params["mission"], base_mission_params)
    params["mission"] = tools.deep_update(params["mission"], example_mission_params)

    params["model"].update(default_model_params)
    params["model"] = tools.deep_update(params["model"], base_model_params)
    params["model"] = tools.deep_update(params["model"], example_model_params)

    params["method"].update(default_method_params)
    params["method"] = tools.deep_update(params["method"], base_method_params)
    params["method"] = tools.deep_update(params["method"], example_method_params)

    # set nondim params
    params = nondim.set_nondim_params(params["model"]["z_types"],
                            params["model"]["u_types"],
                            params["model"]["anchor_types"],
                            params["model"]["anchor_scales"],
                            params,
                            base_unit_labels=params["model"]["base_unit_labels"])
    
    params["case_flag"]                           = 1    # case1

    problem["params"] = params

    # at this point, we should have the params["mission"] and params["model"] dictionaries

    problem["mission"]    = Mission(problem)
    problem["model"]      = Model(problem)
    
    problem["params"] = defaults.set_params_constraint_default(problem["params"])

    # TEMP: REPLACE WITH METHOD CLASS
    problem["params"] = temp_config.method_params(problem["params"])

    # Default state/control bounds
    problem               = defaults.set_problem_default(problem)

    # Cost function
    problem["cost"]       = problem["mission"].cost
    problem["lin_cost"]   = problem["mission"].lin_cost

    # dynamics
    problem["xdot"]       = problem["model"].dynamics
    problem["lin_dyn"]    = problem["model"].lin_dyn

    # Nonconvex inequality constraints
    problem["path_lim"]   = problem["params"]["mission"]["path_lim"]

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

    problem["ts_init"] = params["method"]["ts_init"]
    problem["zs_init"] = params["method"]["zs_init"]
    problem["us_init"] = params["method"]["us_init"]

    return problem

