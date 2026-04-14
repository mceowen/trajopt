import re
import inspect
from functools import partial

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

def _parse_limit_expr(expr_string, fcns):
    """Parse a limit expression like '0.001 * fcns.q_s' into a JAX-compatible callable."""
    expr_string = expr_string.strip()

    # coeff * fcns.name
    match = re.match(r'^([+-]?\d*\.?\d+)\s*\*\s*fcns\.(\w+)$', expr_string)
    if match:
        coeff, name = float(match.group(1)), match.group(2)
    else:
        # fcns.name * coeff
        match = re.match(r'^fcns\.(\w+)\s*\*\s*([+-]?\d*\.?\d+)$', expr_string)
        if match:
            name, coeff = match.group(1), float(match.group(2))
        else:
            # bare fcns.name
            match = re.match(r'^fcns\.(\w+)$', expr_string)
            if match:
                name, coeff = match.group(1), 1.0
            else:
                raise ValueError(f"Cannot parse limit expression: '{expr_string}'")

    fn = fcns[name]
    sig = inspect.signature(fn)
    if 'fcns' in sig.parameters:
        fn = partial(fn, fcns=fcns)

    def limit_fn(t, z, nu, params):
        return coeff * fn(t, z, nu, params)

    return limit_fn

class spatial:
    def __init__(self, cnstr_config, index_map, fcns=None, **kwargs):
        # required config
        self.type       = "spatial"
        self.name       = cnstr_config["name"]
        self.group      = cnstr_config.get("group", None)
        self.units      = cnstr_config.get("units", None)
        self.title      = cnstr_config.get("title", None)
        self.xlabel     = cnstr_config.get("xlabel", None)
        self.ylabel     = cnstr_config.get("ylabel", None)
        self.tick_nbins = cnstr_config.get("tick_nbins", None)

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
        self.title      = cnstr_config.get("title", None)
        self.xlabel     = cnstr_config.get("xlabel", None)
        self.ylabel     = cnstr_config.get("ylabel", None)
        self.tick_nbins = cnstr_config.get("tick_nbins", None)

        self.fcn_string = cnstr_config["fcn"]

        # optional configs
        self.backend      = cnstr_config.get("backend", "jax")
        self.upper_limit  = cnstr_config.get("upper_limit", None)
        self.lower_limit  = cnstr_config.get("lower_limit", None)

        self.upper_limit_fn = None
        self.lower_limit_fn = None

        if isinstance(self.upper_limit, str) and fcns:
            self.upper_limit_fn = _parse_limit_expr(self.upper_limit, fcns)
            self.upper_limit = None
        if isinstance(self.lower_limit, str) and fcns:
            self.lower_limit_fn = _parse_limit_expr(self.lower_limit, fcns)
            self.lower_limit = None

        self.fcn_dim = _resolve_fcn(self.fcn_string, fcns)

    def compile_function(self):
        if self.backend == "jax":
            self.fcn_batched = jax.jit(jax.vmap(self.fcn_dim, in_axes=(0, 0, 0, None)))
            if self.upper_limit_fn:
                self.upper_limit_fn_batched = jax.jit(jax.vmap(self.upper_limit_fn, in_axes=(0, 0, 0, None)))
            if self.lower_limit_fn:
                self.lower_limit_fn_batched = jax.jit(jax.vmap(self.lower_limit_fn, in_axes=(0, 0, 0, None)))

    def compute_trajectory_values(self, t, z, nu, params):
        if self.backend == "jax":
            t_jax = jnp.asarray(t)
            z_jax = jnp.asarray(z)
            nu_jax = jnp.asarray(nu)
            values = np.asarray(self.fcn_batched(t_jax, z_jax, nu_jax, params))

        upper = self.upper_limit
        lower = self.lower_limit

        if self.upper_limit_fn:
            upper = np.asarray(self.upper_limit_fn_batched(t_jax, z_jax, nu_jax, params)).flatten()
        if self.lower_limit_fn:
            lower = np.asarray(self.lower_limit_fn_batched(t_jax, z_jax, nu_jax, params)).flatten()

        limits = {"upper": upper, "lower": lower}
        return {"values": values, "limits": limits}