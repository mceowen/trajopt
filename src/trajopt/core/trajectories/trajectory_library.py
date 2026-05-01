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
        self.zlabel     = cnstr_config.get("zlabel", None)
        self.tick_nbins = cnstr_config.get("tick_nbins", None)
        self.markers    = cnstr_config.get("markers", None)
        self.invert_x   = cnstr_config.get("invert_x", False)
        self.index_map  = index_map
        self.fcn_dim    = tools.resolve_function_from_string(cnstr_config["fcn"], fcns)

        raw_quivers = cnstr_config.get("quivers", {}) or {}
        self.quiver_configs      = list(raw_quivers.values()) if isinstance(raw_quivers, dict) else list(raw_quivers)
        self.quiver_fcn_dims     = [tools.resolve_function_from_string(qcfg["fcn"], fcns) for qcfg in self.quiver_configs]
        self.quiver_fcns_batched = []

    def _wrap(self, fn):
        def fcn(z, nu, params):
            x, t, beta = self.index_map.unpack_z(z)
            u, s       = self.index_map.unpack_nu(nu)
            return fn(t, x, u, params)
        return jax.jit(jax.vmap(fcn, in_axes=(0, 0, None)))

    def compile_function(self):
        self.fcn_batched = self._wrap(self.fcn_dim)
        self.quiver_fcns_batched = [self._wrap(f) for f in self.quiver_fcn_dims]

    def compute_trajectory_values(self, z, nu, params):
        z_jax, nu_jax = jnp.asarray(z), jnp.asarray(nu)
        values = np.asarray(self.fcn_batched(z_jax, nu_jax, params))
        quivers = [
            {"dirs": np.asarray(qfn(z_jax, nu_jax, params)), "config": qcfg}
            for qfn, qcfg in zip(self.quiver_fcns_batched, self.quiver_configs)
        ]
        return {"values": values, "limits": None, "quivers": quivers}


class time_series:
    def __init__(self, cnstr_config, index_map, fcns=None, **kwargs):
        self.type       = "time_series"
        self.name       = cnstr_config["name"]
        self.group      = cnstr_config.get("group", None)
        self.units      = cnstr_config.get("units", None)
        self.title      = cnstr_config.get("title", None)
        self.xlabel     = cnstr_config.get("xlabel", None)
        self.ylabel     = cnstr_config.get("ylabel", None)
        self.zlabel     = cnstr_config.get("zlabel", None)
        self.tick_nbins = cnstr_config.get("tick_nbins", None)
        self.show_iters = cnstr_config.get("show_iters", None)
        self.index_map  = index_map
        self.upper_limit    = cnstr_config.get("upper_limit", None)
        self.lower_limit    = cnstr_config.get("lower_limit", None)
        self.upper_limit_fn = None
        self.lower_limit_fn = None

        if isinstance(self.upper_limit, str) and fcns:
            self.upper_limit_fn = tools.resolve_function_from_string(self.upper_limit, fcns)
            self.upper_limit = None
        if isinstance(self.lower_limit, str) and fcns:
            self.lower_limit_fn = tools.resolve_function_from_string(self.lower_limit, fcns)
            self.lower_limit = None

        self.fcn_dim = tools.resolve_function_from_string(cnstr_config["fcn"], fcns)

    def _wrap(self, fn):
        def fcn(z, nu, params):
            x, t, beta = self.index_map.unpack_z(z)
            u, s       = self.index_map.unpack_nu(nu)
            return fn(t, x, u, params)
        return jax.jit(jax.vmap(fcn, in_axes=(0, 0, None)))

    def compile_function(self):
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
