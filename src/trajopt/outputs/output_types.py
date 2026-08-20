import jax
import jax.numpy as jnp
import numpy as np

from trajopt.utils import tools


def resolve_extractor(output_config, index_map, fcns):
    """Return the function an output evaluates, named by fcn: or by component: and idx:."""
    if "fcn" in output_config:
        return tools.resolve_function_from_string(output_config.fcn, fcns)

    name = output_config.get("component")
    if name is None:
        raise ValueError(
            f"output '{output_config.name}' needs either fcn: or component:"
        )

    for set_name, components in index_map.components.items():
        if name in components:
            idx = components[name]
            break
    else:
        known = sorted(n for c in index_map.components.values() for n in c)
        raise ValueError(
            f"output '{output_config.name}' names unknown component '{name}'; "
            f"known components are {known}"
        )

    if "idx" in output_config:
        idx = idx[np.atleast_1d(output_config.idx)]
    idx = jnp.asarray(idx)

    if set_name == "state":
        return lambda x, u, t, params: jnp.atleast_1d(x)[idx]
    return lambda x, u, t, params: jnp.atleast_1d(u)[idx]


class spatial:
    def __init__(self, output_config, segment):
        index_map = segment.index_map
        nondim = segment.nondim
        fcns = segment.fcns

        self.type       = "spatial"
        self.name       = output_config.name
        self.group      = output_config.get("group")
        self.units      = output_config.get("units")
        self.title      = output_config.get("title")
        self.xlabel     = output_config.get("xlabel")
        self.ylabel     = output_config.get("ylabel")
        self.zlabel     = output_config.get("zlabel")
        self.tick_nbins = output_config.get("tick_nbins")
        self.markers    = output_config.get("markers")
        self.invert_x   = output_config.get("invert_x", False)
        self.index_map  = index_map

        self.fcn_txu_dim    = resolve_extractor(output_config, index_map, fcns)
        self.M_state_nd2d   = jnp.asarray(nondim.M.state.nd2d)
        self.M_ctrl_nd2d    = jnp.asarray(nondim.M.control.nd2d)
        self.time_scale     = float(nondim.time_scale)
        self.fcn_batched    = jax.jit(jax.vmap(self.fcn_znu, in_axes=(0, 0, None)))

        raw_quivers = output_config.get("quivers", {}) or {}
        quiver_configs = list(raw_quivers.values()) if isinstance(raw_quivers, dict) else list(raw_quivers)
        self._quiver_batches = []
        for qcfg in quiver_configs:
            q_fcn = tools.resolve_function_from_string(qcfg["fcn"], fcns)

            def quiver_znu(z, nu, params, fcn=q_fcn):
                x, t, _, u, _ = index_map.unpack_znu(z, nu)
                return fcn(self.M_state_nd2d @ x, self.M_ctrl_nd2d @ u, jnp.asarray(t) * self.time_scale, params)

            q_batch = jax.jit(jax.vmap(quiver_znu, in_axes=(0, 0, None)))

            o_batch = None
            if "origin_fcn" in qcfg:
                o_fcn = tools.resolve_function_from_string(qcfg["origin_fcn"], fcns)

                def origin_znu(z, nu, params, fcn=o_fcn):
                    x, t, _, u, _ = index_map.unpack_znu(z, nu)
                    return fcn(self.M_state_nd2d @ x, self.M_ctrl_nd2d @ u, jnp.asarray(t) * self.time_scale, params)

                o_batch = jax.jit(jax.vmap(origin_znu, in_axes=(0, 0, None)))

            self._quiver_batches.append((q_batch, o_batch, qcfg))

    def fcn_txu_nd(self, x, u, t, params):
        return self.fcn_txu_dim(
            self.M_state_nd2d @ x,
            self.M_ctrl_nd2d @ u,
            jnp.asarray(t) * self.time_scale,
            params,
        )

    def fcn_znu(self, z, nu, params):
        x, t, _, u, _ = self.index_map.unpack_znu(z, nu)
        return self.fcn_txu_nd(x, u, t, params)

    def compute_values(self, z, nu, params):
        z_jax, nu_jax = jnp.asarray(z), jnp.asarray(nu)
        values = np.asarray(self.fcn_batched(z_jax, nu_jax, params))
        quivers = [
            {
                "dirs": np.asarray(q_batch(z_jax, nu_jax, params)),
                "origins": np.asarray(o_batch(z_jax, nu_jax, params)) if o_batch is not None else None,
                "config": tuple(qcfg.items()),
            }
            for q_batch, o_batch, qcfg in self._quiver_batches
        ]
        return {"values": values, "limits": None, "quivers": quivers}


class time_series:
    def __init__(self, output_config, segment):
        index_map = segment.index_map
        nondim = segment.nondim
        fcns = segment.fcns

        self.type       = "time_series"
        self.name       = output_config.name
        self.group      = output_config.get("group")
        self.units      = output_config.get("units")
        self.title      = output_config.get("title")
        self.xlabel     = output_config.get("xlabel")
        self.ylabel     = output_config.get("ylabel")
        self.zlabel     = output_config.get("zlabel")
        self.tick_nbins = output_config.get("tick_nbins")
        self.show_iters = output_config.get("show_iters")
        self.index_map  = index_map

        self.fcn_txu_dim  = resolve_extractor(output_config, index_map, fcns)
        self.M_state_nd2d = jnp.asarray(nondim.M.state.nd2d)
        self.M_ctrl_nd2d  = jnp.asarray(nondim.M.control.nd2d)
        self.time_scale   = float(nondim.time_scale)
        self.fcn_batched  = jax.jit(jax.vmap(self.fcn_znu, in_axes=(0, 0, None)))

        self.upper_limit = output_config.get("upper_limit")
        self.lower_limit = output_config.get("lower_limit")
        self.trigger_line = output_config.get("trigger_line")
        self.upper_limit_batched = None
        self.lower_limit_batched = None

        if isinstance(self.upper_limit, str):
            upper_fcn = tools.resolve_function_from_string(self.upper_limit, fcns)
            self.upper_limit = None

            def upper_znu(z, nu, params):
                x, t, _, u, _ = index_map.unpack_znu(z, nu)
                return upper_fcn(self.M_state_nd2d @ x, self.M_ctrl_nd2d @ u, jnp.asarray(t) * self.time_scale, params)

            self.upper_limit_batched = jax.jit(jax.vmap(upper_znu, in_axes=(0, 0, None)))

        if isinstance(self.lower_limit, str):
            lower_fcn = tools.resolve_function_from_string(self.lower_limit, fcns)
            self.lower_limit = None

            def lower_znu(z, nu, params):
                x, t, _, u, _ = index_map.unpack_znu(z, nu)
                return lower_fcn(self.M_state_nd2d @ x, self.M_ctrl_nd2d @ u, jnp.asarray(t) * self.time_scale, params)

            self.lower_limit_batched = jax.jit(jax.vmap(lower_znu, in_axes=(0, 0, None)))

    def fcn_txu_nd(self, x, u, t, params):
        return self.fcn_txu_dim(
            self.M_state_nd2d @ x,
            self.M_ctrl_nd2d @ u,
            jnp.asarray(t) * self.time_scale,
            params,
        )

    def fcn_znu(self, z, nu, params):
        x, t, _, u, _ = self.index_map.unpack_znu(z, nu)
        return self.fcn_txu_nd(x, u, t, params)

    def compute_values(self, z, nu, params):
        z_jax, nu_jax = jnp.asarray(z), jnp.asarray(nu)
        values = np.asarray(self.fcn_batched(z_jax, nu_jax, params))
        upper = self.upper_limit
        lower = self.lower_limit
        if self.upper_limit_batched is not None:
            upper = np.asarray(self.upper_limit_batched(z_jax, nu_jax, params)).flatten()
        if self.lower_limit_batched is not None:
            lower = np.asarray(self.lower_limit_batched(z_jax, nu_jax, params)).flatten()
        return {"values": values, "limits": {"upper": upper, "lower": lower}, "quivers": []}


class event:
    """A value at one instant of the trajectory.

    before_end: or after_start: sets the offset in seconds from the end or the
    start of the segment. The value is interpolated between the two adjacent
    nodes. It is then repeated at every node.
    """

    def __init__(self, output_config, segment):
        index_map = segment.index_map
        nondim = segment.nondim
        fcns = segment.fcns

        self.type       = "event"
        self.name       = output_config.name
        self.group      = output_config.get("group")
        self.units      = output_config.get("units")
        self.title      = output_config.get("title")
        self.xlabel     = output_config.get("xlabel")
        self.ylabel     = output_config.get("ylabel")
        self.zlabel     = output_config.get("zlabel")
        self.tick_nbins = output_config.get("tick_nbins")
        self.show_iters = output_config.get("show_iters")
        self.index_map  = index_map

        self.before_end  = output_config.get("before_end")
        self.after_start = output_config.get("after_start")
        if (self.before_end is None) == (self.after_start is None):
            raise ValueError(
                f"output '{self.name}' needs exactly one of before_end: or after_start:"
            )

        self.fcn_txu_dim  = resolve_extractor(output_config, index_map, fcns)
        self.M_state_nd2d = jnp.asarray(nondim.M.state.nd2d)
        self.M_ctrl_nd2d  = jnp.asarray(nondim.M.control.nd2d)
        self.time_scale   = float(nondim.time_scale)
        self.fcn_batched  = jax.jit(jax.vmap(self.fcn_znu, in_axes=(0, 0, None)))

    def fcn_txu_nd(self, x, u, t, params):
        return self.fcn_txu_dim(
            self.M_state_nd2d @ x,
            self.M_ctrl_nd2d @ u,
            jnp.asarray(t) * self.time_scale,
            params,
        )

    def fcn_znu(self, z, nu, params):
        x, t, _, u, _ = self.index_map.unpack_znu(z, nu)
        return self.fcn_txu_nd(x, u, t, params)

    def sample_time(self, t):
        """Return the instant to report. It stays inside the segment."""
        if self.before_end is not None:
            t_event = t[-1] - float(self.before_end)
        else:
            t_event = t[0] + float(self.after_start)
        return float(np.clip(t_event, t[0], t[-1]))

    def compute_values(self, z, nu, params):
        z = np.asarray(z)
        values = np.asarray(self.fcn_batched(jnp.asarray(z), jnp.asarray(nu), params))
        values = values.reshape(len(z), -1)
        t = z[:, self.index_map.indices.z.time].flatten() * self.time_scale
        t_event = self.sample_time(t)
        at_event = np.array([np.interp(t_event, t, values[:, j]) for j in range(values.shape[1])])
        return {"values": np.tile(at_event, (len(t), 1)), "limits": None, "quivers": []}
