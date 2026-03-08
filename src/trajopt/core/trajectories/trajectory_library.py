import numpy as np
import cvxpy as cp
import jax
import jax.numpy as jnp
from trajopt.utils.config_loader import resolve_function

class general:
    def __init__(self, cnstr_config, index_map):
        # required config
        self.type       = "nonconvex"
        self.name       = cnstr_config["name"]
        self.group      = cnstr_config.get("group", None)
        self.units      = cnstr_config.get("units", None)

        self.fcn_string = cnstr_config["fcn"]

        # optional configs
        self.backend    = cnstr_config.get("backend", "jax")

        self.fcn_dim = resolve_function(self.fcn_string)

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