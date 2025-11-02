import trajopt.utils.tools                  as tools
import numpy as np

def load_configs(example_name):

    config = {}

    config["mission"] = {}
    config["model"]   = {}
    config["method"]  = {}

    # example configs
    example_pkg = f"trajopt.example_configs.{example_name}"
    example = {
        k: tools.load_yaml(example_pkg, f"{k}.yaml")
        for k in [
            "mission", "model", "method"
            ]
        }

    # base configs
    base_pkgs = {
        "mission": "trajopt.base_configs.missions",
        "model":  "trajopt.base_configs.models",
        "method": "trajopt.base_configs.methods",
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

        # update mission config with planet and vehicle dicts
        if param_type == "mission":
            planet_name = config["mission"]["planet_name"]
            vehicle_name = config["mission"]["vehicle_name"]
            config["mission"]["planet"]  = tools.load_yaml(f"{base_pkgs['mission']}.planet", f"{planet_name}.yaml")
            config["mission"]["vehicle"] = tools.load_yaml(f"{base_pkgs['mission']}.vehicle", f"{vehicle_name}.yaml")

    return config