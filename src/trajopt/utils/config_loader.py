import trajopt.utils.tools                  as tools
import numpy as np

def load_configs(example_name):

    config = {}

    config["example_name"] = example_name

    config["mission"] = {}
    config["model"]   = {}
    config["method"]  = {}

    # example configs
    example_pkg = f"trajopt.examples.{example_name}"
    example = {
        k: tools.load_yaml(example_pkg, f"{k}.yaml")
        for k in [
            "mission", "model", "method"
            ]
        }

    # base configs
    base_pkgs = {
        "mission": "trajopt.core.configs.missions",
        "model":  "trajopt.core.configs.models",
        "method": "trajopt.core.configs.methods",
    }

    base = {
        k: tools.load_yaml(base_pkgs[k], f"{name}.yaml")
        for k, name in {
            "mission": example["mission"]["mission_name"],
            "model":   example["model"]["model_name"],
            "method":  example["method"]["method_name"],
        }.items()
    }

    # general default configs
    default = {
        k: tools.load_yaml(base_pkgs[k], "default.yaml")
        for k in base_pkgs
        }

    # update config with defaults -> base -> example config
    for param_type in ['mission', 'model', 'method']:
        for update_type in [default, base, example]:
            config[param_type] = tools.deep_update(config[param_type], update_type[param_type])

        # update mission config first with planet and vehicle dicts
        if param_type == "mission":
            planet_name  = config["mission"]["planet_name"]
            vehicle_name = config["mission"]["vehicle_name"]
            config["mission"]["planet"]  = tools.load_yaml(f"{base_pkgs['mission']}.planet", f"{planet_name}.yaml")
            config["mission"]["vehicle"] = tools.load_yaml(f"{base_pkgs['mission']}.vehicle", f"{vehicle_name}.yaml")

        # evaluate expressions
        tools.eval_expressions(param_type, config)

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
