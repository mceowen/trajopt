import trajopt.core.modules.utils.tools as tools
import numpy as np

def load_configs(example_name, local=False):
    
    # Define paths based on local flag
    if local:
        example_pkg = f"trajopt.local.examples.{example_name}"
    else:
        example_pkg = f"trajopt.examples.{example_name}"

    problem_dict = tools.load_yaml(example_pkg, "problem.yaml")

    config_dicts = {"mission": {}, "model": {}}
    
    # Load mission and model configs specified in problem.yaml
    for config_type in ["mission", "model"]:
        # Load default and example configs
        name = problem_dict[f"{config_type}_name"]
        default_config = tools.load_yaml(f"trajopt.core.modules.{config_type}.configs", "default.yaml")
        base_config    = tools.load_yaml(f"trajopt.core.modules.{config_type}.configs", f"{name}.yaml")
        # example_config = problem_config[config_type]
        
        # Merge default -> example
        config_dicts[config_type] = tools.deep_update(config_dicts[config_type], default_config)
        config_dicts[config_type] = tools.deep_update(config_dicts[config_type], base_config)
        # config_dicts[config_type] = tools.deep_update(config_dicts[config_type], example_config)

        config_dicts[config_type]['name'] = name
        
        # Load planet and vehicle for mission
        if config_type == "mission":
            config_dicts["mission"]["planet"] = tools.load_yaml(
                f"trajopt.core.modules.mission.configs.planet", 
                f"{config_dicts['mission']['planet']}.yaml"
            )
            config_dicts["mission"]["vehicle"] = tools.load_yaml(
                f"trajopt.core.modules.mission.configs.vehicle",
                f"{config_dicts['mission']['vehicle']}.yaml"
            )
        
        # Evaluate expressions
        tools.eval_expressions(config_type, config_dicts)

    # Load method config specified in method.yaml
    example_method_config = tools.load_yaml(example_pkg, "method.yaml")
    default_method_config = tools.load_yaml(f"trajopt.core.modules.method.configs", "default.yaml")
    base_method_config    = tools.load_yaml(f"trajopt.core.modules.method.configs", f"{example_method_config['base_method']}.yaml")

    method_dict = tools.deep_update(default_method_config, base_method_config)
    method_dict = tools.deep_update(method_dict, example_method_config)

    # Evaluate expressions in method config
    config_dicts["method"] = method_dict
    tools.eval_expressions("method", config_dicts)

    problem_dict["mission"] = config_dicts["mission"]
    problem_dict["model"] = config_dicts["model"]

    config = {"problem": problem_dict, "method": method_dict}
    
    return config

def gen_mc_variations(example_name, local=False):

    mc_variations = {}

    if local:
        mc_variations = tools.load_yaml(f"trajopt.local.examples.{example_name}.variations", "mission.yaml")
    else:
        mc_variations = tools.load_yaml(f"trajopt.examples.{example_name}.variations", "mission.yaml")

    # generate the realizations for the mission variations
    mc_variations["realizations"] = [{}]
    for i in range(0, mc_variations["num_variations"]):
        realization_dict = {}
        for rv_name, rv_properties in mc_variations["random_vars"].items():
            
            rv_var_type = rv_properties.get("variation_type", "uniform")

            if rv_var_type == "uniform":
                lb = rv_properties["lb"]
                ub = rv_properties["ub"]
                
                realization = lb + (ub - lb) * np.random.random(lb.shape)
            
            # TODO (carlos): add normal dispersions as well
            # elif rv_var_type == "normal":
            #     realization = 
            
            realization_dict[rv_name] = realization
        
        mc_variations["realizations"].append(realization_dict)
    return mc_variations

def load_mv_variations(example_name, local=False):

    if local:
        mv_variations = tools.load_yaml(f"trajopt.local.examples.{example_name}.variations", "method.yaml")
    else:
        mv_variations = tools.load_yaml(f"trajopt.examples.{example_name}.variations", "method.yaml")

    return mv_variations
