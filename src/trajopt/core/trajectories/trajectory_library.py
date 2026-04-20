import numpy as np
import jax
import jax.numpy as jnp
import trajopt.utils.tools as tools

class spatial:
    def __init__(self, cnstr_config, index_map, fcns=None, **kwargs):
        self.type       = "spatial"
        self.name       = cnstr_config["name"]
        self.group      = cnstr_config.get("group", None)
        self.units      = cnstr_config.get("units", None)
        self.title      = cnstr_config.get("title", None)
        self.xlabel     = cnstr_config.get("xlabel", None)
        self.ylabel     = cnstr_config.get("ylabel", None)
        self.tick_nbins = cnstr_config.get("tick_nbins", None)
        self.markers    = cnstr_config.get("markers", None)
        self.backend    = cnstr_config.get("backend", "jax")
        self.index_map  = index_map
        self.fcn_dim    = tools.resolve_fcn(cnstr_config["fcn"], fcns)

    def compile_function(self):
        if self.backend == "jax":
            def fcn(z, nu, params):
                x, t, beta = self.index_map.unpack_z(z)
                u, s       = self.index_map.unpack_nu(nu)
                return self.fcn_dim(t, x, u, params)
            self.fcn_batched = jax.jit(jax.vmap(fcn, in_axes=(0, 0, None)))

    def compute_trajectory_values(self, z, nu, params):
        values = np.asarray(self.fcn_batched(jnp.asarray(z), jnp.asarray(nu), params))
        return {"values": values, "limits": None}
    
class time_series:
    def __init__(self, cnstr_config, index_map, fcns=None, **kwargs):
        self.type       = "time_series"
        self.name       = cnstr_config["name"]
        self.group      = cnstr_config.get("group", None)
        self.units      = cnstr_config.get("units", None)
        self.title      = cnstr_config.get("title", None)
        self.xlabel     = cnstr_config.get("xlabel", None)
        self.ylabel     = cnstr_config.get("ylabel", None)
        self.tick_nbins = cnstr_config.get("tick_nbins", None)
        self.backend    = cnstr_config.get("backend", "jax")
        self.index_map  = index_map
        self.upper_limit    = cnstr_config.get("upper_limit", None)
        self.lower_limit    = cnstr_config.get("lower_limit", None)
        self.upper_limit_fn = None
        self.lower_limit_fn = None

        if isinstance(self.upper_limit, str) and fcns:
            self.upper_limit_fn = tools.resolve_fcn(self.upper_limit, fcns)
            self.upper_limit = None
        if isinstance(self.lower_limit, str) and fcns:
            self.lower_limit_fn = tools.resolve_fcn(self.lower_limit, fcns)
            self.lower_limit = None

        self.fcn_dim = tools.resolve_fcn(cnstr_config["fcn"], fcns)

    def _wrap(self, fn):
        def fcn(z, nu, params):
            x, t, beta = self.index_map.unpack_z(z)
            u, s       = self.index_map.unpack_nu(nu)
            return fn(t, x, u, params)
        return jax.jit(jax.vmap(fcn, in_axes=(0, 0, None)))

    def compile_function(self):
        if self.backend == "jax":
            self.fcn_batched = self._wrap(self.fcn_dim)
            if self.upper_limit_fn:
                self.upper_limit_fn_batched = self._wrap(self.upper_limit_fn)
            if self.lower_limit_fn:
                self.lower_limit_fn_batched = self._wrap(self.lower_limit_fn)

    def compute_trajectory_values(self, z, nu, params):
        z_jax, nu_jax = jnp.asarray(z), jnp.asarray(nu)
        values = np.asarray(self.fcn_batched(z_jax, nu_jax, params))
        upper  = self.upper_limit if not self.upper_limit_fn else np.asarray(self.upper_limit_fn_batched(z_jax, nu_jax, params)).flatten()
        lower  = self.lower_limit if not self.lower_limit_fn else np.asarray(self.lower_limit_fn_batched(z_jax, nu_jax, params)).flatten()
        return {"values": values, "limits": {"upper": upper, "lower": lower}}