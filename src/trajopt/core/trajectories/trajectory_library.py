import numpy as np
import jax
import jax.numpy as jnp
import trajopt.utils.tools as tools


def _casadi_to_numpy(val):
    if hasattr(val, 'full'):
        return val.full().flatten().astype(float)
    return np.atleast_1d(np.asarray(val, dtype=float)).flatten()


def _eval_casadi_batched(fn, t, x, u, params):
    N = x.shape[0]
    results = []
    for k in range(N):
        tk = float(t[k]) if np.ndim(t) > 0 else float(t)
        xk = x[k]
        uk = u[k] if k < u.shape[0] else u[-1]
        val = fn(tk, xk, uk, params)
        results.append(_casadi_to_numpy(val))
    return np.array(results)


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
        self.backend    = cnstr_config.get("backend", "jax")
        self.index_map  = index_map
        self.fcn_dim    = tools.resolve_fcn(cnstr_config["fcn"], fcns)

        raw_quivers = cnstr_config.get("quivers", {}) or {}
        self.quiver_configs   = list(raw_quivers.values()) if isinstance(raw_quivers, dict) else list(raw_quivers)
        self.quiver_fcn_dims  = [tools.resolve_fcn(qcfg["fcn"], fcns) for qcfg in self.quiver_configs]
        self.quiver_fcns_batched = []

    def _wrap(self, fn):
        def fcn(z, nu, params):
            x, t, beta = self.index_map.unpack_z(z)
            u, s       = self.index_map.unpack_nu(nu)
            return fn(t, x, u, params)
        return jax.jit(jax.vmap(fcn, in_axes=(0, 0, None)))

    def compile_function(self):
        if self.backend == "jax":
            self.fcn_batched = self._wrap(self.fcn_dim)
            self.quiver_fcns_batched = [self._wrap(f) for f in self.quiver_fcn_dims]
        elif self.backend == "casadi":
            pass

    def compute_trajectory_values(self, z, nu, params):
        if self.backend == "casadi":
            x, t = z[:, self.index_map.indices.z.state], z[:, self.index_map.indices.z.time].squeeze(-1)
            u = nu[:, self.index_map.indices.nu.control]
            values = _eval_casadi_batched(self.fcn_dim, t, x, u, params)
            quivers = [
                {"dirs": _eval_casadi_batched(qfn, t, x, u, params), "config": qcfg}
                for qfn, qcfg in zip(self.quiver_fcn_dims, self.quiver_configs)
            ]
            return {"values": values, "limits": None, "quivers": quivers}

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
        elif self.backend == "casadi":
            pass

    def compute_trajectory_values(self, z, nu, params):
        if self.backend == "casadi":
            x, t = z[:, self.index_map.indices.z.state], z[:, self.index_map.indices.z.time].squeeze(-1)
            u = nu[:, self.index_map.indices.nu.control]
            values = _eval_casadi_batched(self.fcn_dim, t, x, u, params)
            upper = self.upper_limit
            lower = self.lower_limit
            if self.upper_limit_fn:
                upper = _eval_casadi_batched(self.upper_limit_fn, t, x, u, params).flatten()
            if self.lower_limit_fn:
                lower = _eval_casadi_batched(self.lower_limit_fn, t, x, u, params).flatten()
            return {"values": values, "limits": {"upper": upper, "lower": lower}}

        z_jax, nu_jax = jnp.asarray(z), jnp.asarray(nu)
        values = np.asarray(self.fcn_batched(z_jax, nu_jax, params))
        upper  = self.upper_limit if not self.upper_limit_fn else np.asarray(self.upper_limit_fn_batched(z_jax, nu_jax, params)).flatten()
        lower  = self.lower_limit if not self.lower_limit_fn else np.asarray(self.lower_limit_fn_batched(z_jax, nu_jax, params)).flatten()
        return {"values": values, "limits": {"upper": upper, "lower": lower}}