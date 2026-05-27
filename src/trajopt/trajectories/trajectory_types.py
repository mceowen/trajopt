from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from trajopt.indexing import IndexMap
from trajopt.utils import tools
from trajopt.utils.tools import AttrDict


class spatial:
    """Trajectory type for 3D spatial plots."""

    def __init__(self, traj_config: AttrDict, index_map: IndexMap, fcns: AttrDict | None = None, **kwargs) -> None:
        """Initialize spatial trajectory from config."""
        self.type       = "spatial"
        self.name       = traj_config["name"]
        self.group      = traj_config.get("group", None)
        self.units      = traj_config.get("units", None)
        self.title      = traj_config.get("title", None)
        self.xlabel     = traj_config.get("xlabel", None)
        self.ylabel     = traj_config.get("ylabel", None)
        self.zlabel     = traj_config.get("zlabel", None)
        self.tick_nbins = traj_config.get("tick_nbins", None)
        self.markers    = traj_config.get("markers", None)
        self.invert_x   = traj_config.get("invert_x", False)
        self.index_map  = index_map
        self.fcn_dim    = tools.resolve_function_from_string(traj_config["fcn"], fcns)

        raw_quivers = traj_config.get("quivers", {}) or {}
        self.quiver_configs      = list(raw_quivers.values()) if isinstance(raw_quivers, dict) else list(raw_quivers)
        self.quiver_fcn_dims     = [tools.resolve_function_from_string(qcfg["fcn"], fcns) for qcfg in self.quiver_configs]
        self.quiver_origin_fcn_dims = [
            tools.resolve_function_from_string(qcfg["origin_fcn"], fcns) if "origin_fcn" in qcfg else None
            for qcfg in self.quiver_configs
        ]
        self.quiver_fcns_batched = []
        self.quiver_origin_fcns_batched = []

    def _wrap(self, fn: Callable) -> Callable:
        """Wrap fn as a batched, JIT-compiled (z, nu, params) -> output function."""
        def fcn(z, nu, params):
            x, t, beta = self.index_map.unpack_z(z)
            u, s       = self.index_map.unpack_nu(nu)
            return fn(t, x, u, params)
        return jax.jit(jax.vmap(fcn, in_axes=(0, 0, None)))

    def compile_function(self) -> None:
        """Compile batched JIT functions for trajectory and quiver evaluations."""
        self.fcn_batched = self._wrap(self.fcn_dim)
        self.quiver_fcns_batched = [self._wrap(f) for f in self.quiver_fcn_dims]
        self.quiver_origin_fcns_batched = [
            self._wrap(f) if f is not None else None
            for f in self.quiver_origin_fcn_dims
        ]

    def compute_trajectory_values(self, z: np.ndarray | Array, nu: np.ndarray | Array, params: AttrDict) -> dict:
        """Evaluate the spatial trajectory and quivers over the time grid."""
        z_jax, nu_jax = jnp.asarray(z), jnp.asarray(nu)
        values = np.asarray(self.fcn_batched(z_jax, nu_jax, params))
        quivers = [
            {
                "dirs": np.asarray(qfn(z_jax, nu_jax, params)),
                "origins": np.asarray(ofn(z_jax, nu_jax, params)) if ofn is not None else None,
                "config": qcfg,
            }
            for qfn, ofn, qcfg in zip(self.quiver_fcns_batched, self.quiver_origin_fcns_batched, self.quiver_configs)
        ]
        return {"values": values, "limits": None, "quivers": quivers}


class time_series:
    """Trajectory type for time-series plots."""

    def __init__(self, traj_config: AttrDict, index_map: IndexMap, fcns: AttrDict | None = None, **kwargs) -> None:
        """Initialize time-series trajectory from config."""
        self.type       = "time_series"
        self.name       = traj_config["name"]
        self.group      = traj_config.get("group", None)
        self.units      = traj_config.get("units", None)
        self.title      = traj_config.get("title", None)
        self.xlabel     = traj_config.get("xlabel", None)
        self.ylabel     = traj_config.get("ylabel", None)
        self.zlabel     = traj_config.get("zlabel", None)
        self.tick_nbins = traj_config.get("tick_nbins", None)
        self.show_iters = traj_config.get("show_iters", None)
        self.index_map  = index_map
        self.upper_limit    = traj_config.get("upper_limit", None)
        self.lower_limit    = traj_config.get("lower_limit", None)
        self.upper_limit_fn = None
        self.lower_limit_fn = None

        if isinstance(self.upper_limit, str) and fcns:
            self.upper_limit_fn = tools.resolve_function_from_string(self.upper_limit, fcns)
            self.upper_limit = None
        if isinstance(self.lower_limit, str) and fcns:
            self.lower_limit_fn = tools.resolve_function_from_string(self.lower_limit, fcns)
            self.lower_limit = None

        self.fcn_dim = tools.resolve_function_from_string(traj_config["fcn"], fcns)

    def _wrap(self, fn: Callable) -> Callable:
        """Wrap fn as a batched, JIT-compiled (z, nu, params) -> output function."""
        def fcn(z, nu, params):
            x, t, beta = self.index_map.unpack_z(z)
            u, s       = self.index_map.unpack_nu(nu)
            return fn(t, x, u, params)
        return jax.jit(jax.vmap(fcn, in_axes=(0, 0, None)))

    def compile_function(self) -> None:
        """Compile batched JIT functions for trajectory and optional limit evaluations."""
        self.fcn_batched = self._wrap(self.fcn_dim)
        if self.upper_limit_fn:
            self.upper_limit_fn_batched = self._wrap(self.upper_limit_fn)
        if self.lower_limit_fn:
            self.lower_limit_fn_batched = self._wrap(self.lower_limit_fn)

    def compute_trajectory_values(self, z: np.ndarray | Array, nu: np.ndarray | Array, params: AttrDict) -> dict:
        """Evaluate the time-series trajectory values and limits over the time grid."""
        z_jax, nu_jax = jnp.asarray(z), jnp.asarray(nu)
        values = np.asarray(self.fcn_batched(z_jax, nu_jax, params))
        upper  = self.upper_limit if not self.upper_limit_fn else np.asarray(self.upper_limit_fn_batched(z_jax, nu_jax, params)).flatten()
        lower  = self.lower_limit if not self.lower_limit_fn else np.asarray(self.lower_limit_fn_batched(z_jax, nu_jax, params)).flatten()
        return {"values": values, "limits": {"upper": upper, "lower": lower}}
