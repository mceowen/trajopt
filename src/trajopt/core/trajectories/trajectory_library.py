import numpy as np
import jax
import jax.numpy as jnp
from trajopt.utils.config_loader import resolve_function_from_path

def _resolve_fcn(fcn_string, fcns=None):
    """resolve function string from either fcns dictionary or direct path"""
    if fcns and isinstance(fcn_string, str) and fcn_string.startswith('fcns.'):
        key = fcn_string.split('.', 1)[1]
        if key not in fcns:
            raise KeyError(f"Function reference '{fcn_string}' not found in fcns dict. "
                           f"Available functions: {list(fcns.keys())}")
        return fcns[key]

    else:
        return resolve_function_from_path(fcn_string)

class spatial:
    def __init__(self, cnstr_config, index_map, fcns=None, **kwargs):
        # required config
        self.type       = "spatial"
        self.name       = cnstr_config["name"]
        self.group      = cnstr_config.get("group", None)
        self.units      = cnstr_config.get("units", None)

        self.fcn_string = cnstr_config["fcn"]

        # optional configs
        self.backend    = cnstr_config.get("backend", "jax")

        self.fcn_dim = _resolve_fcn(self.fcn_string, fcns)

    def compile_function(self):
        if self.backend == "jax":
            self.fcn_batched = jax.jit(jax.vmap(self.fcn_dim, in_axes=(0, 0, 0, None)))

    def compute_trajectory_values(self, t, z, nu, params):
        if self.backend == "jax":
            t_jax = jnp.asarray(t)
            z_jax = jnp.asarray(z)
            nu_jax = jnp.asarray(nu)
            values = np.asarray(self.fcn_batched(t_jax, z_jax, nu_jax, params))

        return {"values": values, "limits": None}
    
class time_series:
    def __init__(self, cnstr_config, index_map, fcns=None, **kwargs):
        # required config
        self.type       = "time_series"
        self.name       = cnstr_config["name"]
        self.group      = cnstr_config.get("group", None)
        self.units      = cnstr_config.get("units", None)

        self.fcn_string = cnstr_config["fcn"]

        # optional configs
        self.backend    = cnstr_config.get("backend", "jax")

        self.fcn_dim = _resolve_fcn(self.fcn_string, fcns)

    def compile_function(self):
        if self.backend == "jax":
            self.fcn_batched = jax.jit(jax.vmap(self.fcn_dim, in_axes=(0, 0, 0, None)))

    def compute_trajectory_values(self, t, z, nu, params):
        if self.backend == "jax":
            t_jax = jnp.asarray(t)
            z_jax = jnp.asarray(z)
            nu_jax = jnp.asarray(nu)
            values = np.asarray(self.fcn_batched(t_jax, z_jax, nu_jax, params))

        return {"values": values, "limits": None}