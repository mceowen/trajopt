import trajopt.utils.tools as tools
import numpy as np

def load_configs(example_name, local=False):
    
    # Define paths based on local flag
    if local:
        example_pkg = f"trajopt.local.examples.{example_name}"
        config_base = "trajopt.local.configs"
    else:
        example_pkg = f"trajopt.examples.{example_name}"
        config_base = "trajopt.core.configs"
    
    config = {"example_name": example_name}
    
    # Load configs for mission, model, method
    for config_type in ["mission", "model", "method"]:
        # Load default and example configs
        default_cfg = tools.load_yaml(f"{config_base}.{config_type}", "default.yaml")
        example_cfg = tools.load_yaml(example_pkg, f"{config_type}.yaml")
        
        # Merge default -> example
        config[config_type] = tools.deep_update({}, default_cfg)
        config[config_type] = tools.deep_update(config[config_type], example_cfg)
        
        # Load planet and vehicle for mission
        if config_type == "mission":
            config["mission"]["planet"] = tools.load_yaml(
                f"{config_base}.mission.planet", 
                f"{config['mission']['planet_name']}.yaml"
            )
            config["mission"]["vehicle"] = tools.load_yaml(
                f"{config_base}.mission.vehicle",
                f"{config['mission']['vehicle_name']}.yaml"
            )
        
        # Evaluate expressions
        tools.eval_expressions(config_type, config)
    
    return config

def gen_mc_variations(example_name):

    mc_variations = {}

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

def load_mv_variations(example_name):

    return tools.load_yaml(f"trajopt.examples.{example_name}.variations", "method.yaml")
