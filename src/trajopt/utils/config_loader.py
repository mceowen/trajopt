import trajopt.utils.set_defaults           as defaults
import trajopt.utils.tools                  as tools
import trajopt.algorithm.initial_guess      as guess
import trajopt.algorithm.convergence        as convergence
import trajopt.algorithm.convexification    as convexify
import trajopt.utils.nondim                 as nondim
import numpy as np

def load_configs(example_name):

    config = {}

    config["mission"] = {}
    config["model"] = {}
    config["method"] = {}

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
            if f'{update_type}' == 'default':
                config[param_type].update(default[param_type])
            else:
                config[param_type] = tools.deep_update(config[param_type], update_type[param_type])

    return config