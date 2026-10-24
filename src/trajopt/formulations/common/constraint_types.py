import cvxpy as cp
import jax.numpy as jnp
import numpy as np

from trajopt.utils import tools
from trajopt.formulations.common.constraint import Constraint


class initial_state(Constraint):
    def __init__(self, cnstr_config: dict, phase) -> None:
        self.name = cnstr_config.name
        self.type = "initial_state"

        nondim = phase.nondim
        raw_value = cnstr_config["value"]

        if "idx" in cnstr_config:
            self.idx = cnstr_config.idx
            self._value_dim = np.atleast_1d(raw_value)
        else:
            self.idx = [i for i, v in enumerate(raw_value) if v is not None]
            self._value_dim = np.atleast_1d([v for v in raw_value if v is not None])

        self.dimension = len(self.idx)
        self._nondim = nondim

    @property
    def value(self):
        return self._nondim.M.state.d2nd[np.ix_(self.idx, self.idx)] @ self._value_dim


class final_state(Constraint):
    def __init__(self, cnstr_config: dict, phase) -> None:
        self.name = cnstr_config.name
        self.type = "final_state"

        nondim = phase.nondim
        raw_value = cnstr_config.value

        if "idx" in cnstr_config:
            self.idx = cnstr_config["idx"]
            self._value_dim = np.atleast_1d(raw_value)
        else:
            self.idx = [i for i, v in enumerate(raw_value) if v is not None]
            self._value_dim = np.atleast_1d([v for v in raw_value if v is not None])

        self.dimension = len(self.idx)
        self._nondim = nondim

    @property
    def value(self):
        return self._nondim.M.state.d2nd[np.ix_(self.idx, self.idx)] @ self._value_dim


class initial_control(Constraint):
    def __init__(self, cnstr_config: dict, phase) -> None:
        self.name = cnstr_config.name
        self.type = "initial_control"

        nondim = phase.nondim
        raw_value = cnstr_config["value"]

        if "idx" in cnstr_config:
            self.idx = cnstr_config.idx
            self._value_dim = np.atleast_1d(raw_value)
        else:
            self.idx = [i for i, v in enumerate(raw_value) if v is not None]
            self._value_dim = np.atleast_1d([v for v in raw_value if v is not None])

        self.dimension = len(self.idx)
        self._nondim = nondim

    @property
    def value(self):
        return self._nondim.M.control.d2nd[np.ix_(self.idx, self.idx)] @ self._value_dim


class final_control(Constraint):
    def __init__(self, cnstr_config: dict, phase) -> None:
        self.name = cnstr_config.name
        self.type = "final_control"

        nondim = phase.nondim
        raw_value = cnstr_config.value

        if "idx" in cnstr_config:
            self.idx = cnstr_config["idx"]
            self._value_dim = np.atleast_1d(raw_value)
        else:
            self.idx = [i for i, v in enumerate(raw_value) if v is not None]
            self._value_dim = np.atleast_1d([v for v in raw_value if v is not None])

        self.dimension = len(self.idx)
        self._nondim = nondim

    @property
    def value(self):
        return self._nondim.M.control.d2nd[np.ix_(self.idx, self.idx)] @ self._value_dim


class state_limits(Constraint):
    def __init__(self, cnstr_config: dict, phase) -> None:
        index_map = phase.index_map
        nondim = phase.nondim

        self.name = cnstr_config.name
        self.type = "state_limits"

        raw_lower = cnstr_config.get("lower", [])
        raw_upper = cnstr_config.get("upper", [])

        self.lower_idx = [i for i, v in enumerate(raw_lower) if v is not None]
        self.upper_idx = [i for i, v in enumerate(raw_upper) if v is not None]
        self._lower_dim = np.atleast_1d([v for v in raw_lower if v is not None]) if self.lower_idx else np.array([])
        self._upper_dim = np.atleast_1d([v for v in raw_upper if v is not None]) if self.upper_idx else np.array([])

        self.dimension = len(self.lower_idx) + len(self.upper_idx)

        n_elem = index_map.n.state
        parts = []
        if self.lower_idx:
            parts.append(-np.eye(n_elem)[self.lower_idx, :])
        if self.upper_idx:
            parts.append(np.eye(n_elem)[self.upper_idx, :])
        self.M_select = np.vstack(parts) if parts else np.zeros((0, n_elem))
        self._nondim = nondim

    def _d2nd(self, idx):
        return self._nondim.M.state.d2nd[np.ix_(idx, idx)]

    @property
    def lower_value(self):
        return self._d2nd(self.lower_idx) @ self._lower_dim if self.lower_idx else self._lower_dim

    @property
    def upper_value(self):
        return self._d2nd(self.upper_idx) @ self._upper_dim if self.upper_idx else self._upper_dim

    @property
    def rhs(self):
        parts = [v for v in (self.lower_value if self.lower_idx else None,
                             self.upper_value if self.upper_idx else None) if v is not None]
        return np.concatenate(parts) if parts else np.array([])


class initial_state_limits(state_limits):
    def __init__(self, cnstr_config, phase):
        super().__init__(cnstr_config, phase)
        self.type = "initial_state_limits"


class final_state_limits(state_limits):
    def __init__(self, cnstr_config, phase):
        super().__init__(cnstr_config, phase)
        self.type = "final_state_limits"


class control_limits(Constraint):
    def __init__(self, cnstr_config: dict, phase) -> None:
        index_map = phase.index_map
        nondim = phase.nondim

        self.name = cnstr_config.name
        self.type = "control_limits"

        raw_lower = cnstr_config.get("lower", [])
        raw_upper = cnstr_config.get("upper", [])

        self.lower_idx = [i for i, v in enumerate(raw_lower) if v is not None]
        self.upper_idx = [i for i, v in enumerate(raw_upper) if v is not None]
        self._lower_dim = np.atleast_1d([v for v in raw_lower if v is not None]) if self.lower_idx else np.array([])
        self._upper_dim = np.atleast_1d([v for v in raw_upper if v is not None]) if self.upper_idx else np.array([])

        self.dimension = len(self.lower_idx) + len(self.upper_idx)

        n_elem = index_map.n.control
        parts = []
        if self.lower_idx:
            parts.append(-np.eye(n_elem)[self.lower_idx, :])
        if self.upper_idx:
            parts.append(np.eye(n_elem)[self.upper_idx, :])
        self.M_select = np.vstack(parts) if parts else np.zeros((0, n_elem))
        self._nondim = nondim

    @property
    def lower_value(self):
        idx = self.lower_idx
        return self._nondim.M.control.d2nd[np.ix_(idx, idx)] @ self._lower_dim

    @property
    def upper_value(self):
        idx = self.upper_idx
        return self._nondim.M.control.d2nd[np.ix_(idx, idx)] @ self._upper_dim

    @property
    def rhs(self):
        parts = [v for v in (self.lower_value if self.lower_idx else None,
                             self.upper_value if self.upper_idx else None) if v is not None]
        return np.concatenate(parts) if parts else np.array([])


class initial_control_limits(control_limits):
    def __init__(self, cnstr_config, phase):
        super().__init__(cnstr_config, phase)
        self.type = "initial_control_limits"


class final_control_limits(control_limits):
    def __init__(self, cnstr_config, phase):
        super().__init__(cnstr_config, phase)
        self.type = "final_control_limits"


class control_rate_limit(Constraint):
    def __init__(self, cnstr_config: dict, phase) -> None:
        index_map = phase.index_map
        nondim = phase.nondim

        self.type = "control_rate_limit"
        self.name = cnstr_config.name

        raw_value = cnstr_config["value"]
        if "idx" in cnstr_config:
            self.idx = cnstr_config.idx
            self._value_dim = np.atleast_1d(raw_value)
        else:
            self.idx = [i for i, v in enumerate(raw_value) if v is not None]
            self._value_dim = np.atleast_1d([v for v in raw_value if v is not None])

        self.dimension = len(self.idx)

        n_elem = index_map.n.control
        M_min = -np.eye(n_elem)[self.idx, :]
        M_max = np.eye(n_elem)[self.idx, :]
        self.M_select = np.vstack([M_min, M_max])
        self._nondim = nondim

    @property
    def value(self):
        nondim = self._nondim
        return nondim.time_scale * nondim.M.control.d2nd[np.ix_(self.idx, self.idx)] @ self._value_dim


class control_accel_limit(control_rate_limit):
    """|u_{k+1} - 2 u_k + u_{k-1}| <= value * dt_{k-1} * dt_k."""

    def __init__(self, cnstr_config, phase):
        super().__init__(cnstr_config, phase)
        self.type = "control_accel_limit"

    @property
    def value(self):
        nondim = self._nondim
        return nondim.time_scale**2 * nondim.M.control.d2nd[np.ix_(self.idx, self.idx)] @ self._value_dim


class initial_time(Constraint):
    """Sets the time the phase starts at, else it comes from guess.t_start."""
    def __init__(self, cnstr_config: dict, phase) -> None:
        self.type = "initial_time"
        self.name = cnstr_config.name
        self.dimension = 1

        self._value_dim = cnstr_config["value"]
        self._nondim = phase.nondim

    @property
    def value(self):
        return self._value_dim / self._nondim.time_scale


class final_time(Constraint):
    """Sets the time the phase ends at. value fixes it, lower and upper bound it."""
    def __init__(self, cnstr_config: dict, phase) -> None:
        index_map = phase.index_map
        nondim = phase.nondim

        self.type = "final_time"
        self.name = cnstr_config.name
        self.dimension = 1

        fixed = cnstr_config.get("value", None)
        self._fixed_dim = fixed
        self._lower_dim = cnstr_config.get("lower", fixed)
        self._upper_dim = cnstr_config.get("upper", fixed)

        self._N_all = index_map.N.all
        self._nondim = nondim

    @property
    def is_fixed(self):
        return self._fixed_dim is not None

    @property
    def fixed_value(self):
        return self._fixed_dim / self._nondim.time_scale if self.is_fixed else None

    @property
    def lower(self):
        return self._lower_dim / self._nondim.time_scale if self._lower_dim is not None else None

    @property
    def upper(self):
        return self._upper_dim / self._nondim.time_scale if self._upper_dim is not None else None

    @property
    def dt_min(self):
        return self.lower / (self._N_all - 1) if self.lower is not None else None

    @property
    def dt_max(self):
        return self.upper / (self._N_all - 1) if self.upper is not None else None


class convex_inequality(Constraint):
    def __init__(self, cnstr_config: dict, phase) -> None:
        index_map = phase.index_map
        nondim = phase.nondim
        fcns = phase.fcns
        params = phase.params

        self.type      = "convex_inequality"
        self.name      = cnstr_config.name
        self._upper_dim = cnstr_config.get("upper", None)
        self._lower_dim = cnstr_config.get("lower", None)
        self.index_map = index_map

        self.fcn_string = cnstr_config.fcn
        self.fcn_xu_dim = tools.resolve_function_from_string(self.fcn_string, fcns)

        out = self.fcn_xu_dim(np.ones((1, index_map.n.state)), np.ones((1, index_map.n.control)), params)
        self.dimension = 1 if out.ndim == 1 else out.shape[1]

        self.scale = np.atleast_1d(cnstr_config.get("scale", np.ones(self.dimension)))
        if self._upper_dim is not None and self._lower_dim is not None:
            self.dimension = 2 * self.dimension

        self.M_state_nd2d = np.asarray(nondim.M.state.nd2d)
        self.M_ctrl_nd2d  = np.asarray(nondim.M.control.nd2d)
        
        self.time_scale = nondim.time_scale

        self.M_out_d2nd = np.diag(1.0 / np.abs(self.scale))

    @property
    def upper(self):
        return self.M_out_d2nd @ np.atleast_1d(self._upper_dim) if self._upper_dim is not None else None

    @property
    def lower(self):
        return self.M_out_d2nd @ np.atleast_1d(self._lower_dim) if self._lower_dim is not None else None

    def fcn_txu_nd(self, x, u, t, params):
        M_out_diag = np.diag(self.M_out_d2nd)
        g = cp.multiply(M_out_diag, self.fcn_xu_dim(x @ self.M_state_nd2d.T, u @ self.M_ctrl_nd2d.T, params))
        pieces = []
        if self.lower is not None:
            pieces.append(self.lower - g)
        if self.upper is not None:
            pieces.append(g - self.upper)
        if not pieces:
            pieces.append(g)
        return cp.hstack(pieces)

    def fcn_znu(self, z, nu, params):
        x, t, _, u, _ = self.index_map.unpack_znu(z, nu)
        return self.fcn_txu_nd(x, u, t, params)


class initial_convex_inequality(convex_inequality):
    def __init__(self, cnstr_config, phase):
        super().__init__(cnstr_config, phase)
        self.type = "initial_convex_inequality"


class final_convex_inequality(convex_inequality):
    def __init__(self, cnstr_config, phase):
        super().__init__(cnstr_config, phase)
        self.type = "final_convex_inequality"


_RESERVED_CONSTRAINT_KEYS = {"type", "fcn", "name", "lower", "upper", "scale", "eps"}


class nonconvex_inequality(Constraint):
    def __init__(self, cnstr_config: dict, phase) -> None:
        index_map = phase.index_map
        nondim = phase.nondim
        fcns = phase.fcns
        params = phase.params

        self.type      = "nonconvex_inequality"
        self.name      = cnstr_config.name
        self._upper_dim = cnstr_config.get("upper", None)
        self._lower_dim = cnstr_config.get("lower", None)
        self.index_map = index_map

        # any non-standard fields (e.g. a fixed 'pos') are exposed to the
        # model fcn under params.<constraint name>, alongside its own params
        extra = {k: v for k, v in cnstr_config.items() if k not in _RESERVED_CONSTRAINT_KEYS}
        if extra:
            params[self.name] = tools.deep_merge(params.get(self.name, {}), extra)

        self.fcn_string = cnstr_config.fcn
        self.fcn_txu_dim = tools.resolve_function_from_string(self.fcn_string, fcns)

        out = self.fcn_txu_dim(np.ones(index_map.n.state), np.ones(index_map.n.control), 0.0, params)
        self.dimension = jnp.atleast_1d(out).shape[0]

        self.scale = np.atleast_1d(cnstr_config.get("scale", np.ones(self.dimension)))
        if self._upper_dim is not None and self._lower_dim is not None:
            self.dimension = 2 * self.dimension

        self.M_out_d2nd   = jnp.diag(1.0 / jnp.abs(jnp.asarray(self.scale)))
        self.M_state_nd2d = jnp.asarray(nondim.M.state.nd2d)
        self.M_ctrl_nd2d  = jnp.asarray(nondim.M.control.nd2d)
        self.time_scale     = nondim.time_scale

    @property
    def upper(self):
        return self.M_out_d2nd @ jnp.atleast_1d(self._upper_dim) if self._upper_dim is not None else None

    @property
    def lower(self):
        return self.M_out_d2nd @ jnp.atleast_1d(self._lower_dim) if self._lower_dim is not None else None

    def fcn_txu_nd(self, x, u, t, params):
        g = self.M_out_d2nd @ self.fcn_txu_dim(self.M_state_nd2d @ x, self.M_ctrl_nd2d @ u, self.time_scale * t, params)
        pieces = []
        if self.lower is not None:
            pieces.append(self.lower - g)
        if self.upper is not None:
            pieces.append(g - self.upper)
        if not pieces:
            pieces.append(g)
        return jnp.concatenate(pieces)

    def fcn_znu(self, z, nu, params):
        x, t, _, u, _ = self.index_map.unpack_znu(z, nu)
        return self.fcn_txu_nd(x, u, t, params)


class initial_nonconvex_inequality(nonconvex_inequality):
    def __init__(self, cnstr_config, phase):
        super().__init__(cnstr_config, phase)
        self.type = "initial_nonconvex_inequality"


class final_nonconvex_inequality(nonconvex_inequality):
    def __init__(self, cnstr_config, phase):
        super().__init__(cnstr_config, phase)
        self.type = "final_nonconvex_inequality"


class ctcs_nonconvex_inequality(nonconvex_inequality):
    def __init__(self, cnstr_config, phase):
        super().__init__(cnstr_config, phase)
        self.type = "ctcs_nonconvex_inequality"


class nonconvex_equality(Constraint):
    def __init__(self, cnstr_config: dict, phase) -> None:
        index_map = phase.index_map
        nondim = phase.nondim
        fcns = phase.fcns
        params = phase.params

        self.type      = "nonconvex_equality"
        self.name      = cnstr_config.name
        self.index_map = index_map

        self.fcn_string  = cnstr_config.fcn
        self.fcn_txu_dim = tools.resolve_function_from_string(self.fcn_string, fcns)

        out = self.fcn_txu_dim(np.ones(index_map.n.state), np.ones(index_map.n.control), 0.0, params)
        self.dimension = jnp.atleast_1d(out).shape[0]

        self.scale = np.atleast_1d(cnstr_config.get("scale", np.ones(self.dimension)))

        self.M_out_d2nd   = jnp.diag(1.0 / jnp.abs(jnp.asarray(self.scale)))
        self.M_state_nd2d = jnp.asarray(nondim.M.state.nd2d)
        self.M_ctrl_nd2d  = jnp.asarray(nondim.M.control.nd2d)
        self.time_scale     = nondim.time_scale

    def fcn_txu_nd(self, x, u, t, params):
        return self.M_out_d2nd @ self.fcn_txu_dim(self.M_state_nd2d @ x, self.M_ctrl_nd2d @ u, self.time_scale * t, params)

    def fcn_znu(self, z, nu, params):
        x, t, _, u, _ = self.index_map.unpack_znu(z, nu)
        return self.fcn_txu_nd(x, u, t, params)


class initial_nonconvex_equality(nonconvex_equality):
    def __init__(self, cnstr_config, phase):
        super().__init__(cnstr_config, phase)
        self.type = "initial_nonconvex_equality"


class final_nonconvex_equality(nonconvex_equality):
    def __init__(self, cnstr_config, phase):
        super().__init__(cnstr_config, phase)
        self.type = "final_nonconvex_equality"


def _continuity_indices(phase, kind, selection):
    """Resolve numeric indices or channel group names in the physical layout."""
    width = phase.index_map.n[kind]
    if selection is None:
        return np.arange(width, dtype=int)
    groups = phase.index_map.components[kind]
    if isinstance(selection, (str, int, np.integer)):
        selection = [selection]
    indices = []
    for item in selection:
        if isinstance(item, str):
            if item not in groups:
                raise ValueError(f"Unknown {kind} channel {item!r} in segment {phase.name!r}")
            indices.extend(int(i) for i in groups[item])
        elif isinstance(item, (int, np.integer)):
            indices.append(int(item))
        else:
            raise ValueError("Continuity channel selections must contain indices or channel names")
    if not indices or len(set(indices)) != len(indices) or any(i < 0 or i >= width for i in indices):
        raise ValueError(f"Invalid {kind} continuity indices {indices} for segment {phase.name!r}")
    return np.asarray(indices, dtype=int)


def _continuity_channel_metadata(phase, kind):
    metadata = [None] * phase.index_map.n[kind]
    groups = phase.index_map.components[kind]
    phase_config = getattr(phase, "config", {})
    config = phase_config.get(kind, {})
    default_frame = phase_config.get("reference_frame", phase_config.get("frame"))
    for name, indices in groups.items():
        group = config.get(name, {})
        frame = group.get("reference_frame", group.get("frame", default_frame))
        for local, index in enumerate(indices):
            metadata[int(index)] = (name, local, group.get("units", group.get("unit")), frame)
    return metadata


class _continuity(Constraint):
    """Segment endpoint differences in physical units, divided by residual scales."""
    groups = ()

    def __init__(self, cnstr_config, phase):
        self.name = cnstr_config.name
        self.phase_name = cnstr_config.phase
        self.idx = cnstr_config.get("target_idx", cnstr_config.get("idx"))
        self._config = cnstr_config
        self.dimension = 0
        for kind in self.groups:
            if kind == "time":
                self.dimension += phase.index_map.n.time
            else:
                _, selection, _ = self._selections(kind)
                self.dimension += len(_continuity_indices(phase, kind, selection))
        if len(self.groups) == 1 and self.groups[0] in {"state", "control"}:
            _, selection, _ = self._selections(self.groups[0])
            self.idx = (_continuity_indices(phase, self.groups[0], selection).tolist()
                        if selection is not None else None)

    def _selections(self, kind):
        mapping = self._config.get(f"{kind}_map")
        if mapping is not None:
            if not hasattr(mapping, "get") or "source" not in mapping or "target" not in mapping:
                raise ValueError(f"{kind}_map requires source and target channel selections")
            return mapping["source"], mapping["target"], True
        common = self._config.get("idx")
        explicit = "source_idx" in self._config or "target_idx" in self._config
        return (self._config.get("source_idx", common),
                self._config.get("target_idx", common), explicit)

    def _layout(self, source, target, kind):
        if kind == "time":
            left = source.problem.config.get("time", {})
            right = target.problem.config.get("time", {})
            self._validate_units(left.get("units", left.get("unit")),
                                 right.get("units", right.get("unit")), kind)
            return (source.index_map.indices.z.time, target.index_map.indices.z.time,
                    np.atleast_1d(source.nondim.time_scale),
                    np.atleast_1d(target.nondim.time_scale))
        source_selection, target_selection, explicit = self._selections(kind)
        si = _continuity_indices(source.problem, kind, source_selection)
        ti = _continuity_indices(target.problem, kind, target_selection)
        if len(si) != len(ti):
            raise ValueError(f"Coupling '{self.name}' has different source and target {kind} dimensions; "
                             "provide explicit source_idx and target_idx")
        sm = _continuity_channel_metadata(source.problem, kind)
        tm = _continuity_channel_metadata(target.problem, kind)
        for a, b in zip(si, ti):
            left, right = sm[a], tm[b]
            if not explicit and (left is None or right is None or left[:2] != right[:2]):
                raise ValueError(f"Coupling '{self.name}' connects different {kind} channels; "
                                 "provide an explicit channel mapping")
            if left is not None and right is not None:
                self._validate_units(left[2], right[2], kind)
                if left[3] and right[3] and left[3] != right[3]:
                    raise ValueError(f"Coupling '{self.name}' has incompatible reference frames")
        container = "z" if kind == "state" else "nu"
        return (source.index_map.indices[container][kind][si],
                target.index_map.indices[container][kind][ti],
                np.asarray(getattr(source.nondim, f"{kind}_scales"))[si],
                np.asarray(getattr(target.nondim, f"{kind}_scales"))[ti])

    def _validate_units(self, left, right, kind):
        conversion = self._config.get("conversion")
        if hasattr(conversion, "get"):
            conversion = conversion.get(kind)
        if left and right and left != right and conversion is None:
            raise ValueError(f"Coupling '{self.name}' has incompatible units {left!r}/{right!r}; "
                             f"provide an explicit {kind} conversion")

    def _conversion(self, kind):
        conversion = self._config.get("conversion", 1.)
        return conversion.get(kind, 1.) if hasattr(conversion, "get") else conversion

    def _coefficients(self, source, target, kind):
        si, ti, source_scale, target_scale = self._layout(source, target, kind)
        scale = self._config.get("scale", target_scale)
        conversion = self._conversion(kind)
        if hasattr(scale, "get"):
            scale = scale.get(kind, target_scale)
        try:
            scale = np.broadcast_to(np.asarray(scale, dtype=float), target_scale.shape)
            conversion = np.broadcast_to(np.asarray(conversion, dtype=float), source_scale.shape)
        except ValueError as exc:
            raise ValueError(f"Coupling '{self.name}' scale/conversion must match its {kind} channel count") from exc
        if not np.all(np.isfinite(scale)) or np.any(scale <= 0):
            raise ValueError(f"Coupling '{self.name}' residual scales must be finite and positive")
        if not np.all(np.isfinite(conversion)):
            raise ValueError(f"Coupling '{self.name}' conversions must be finite")
        return si, ti, source_scale * conversion / scale, target_scale / scale

    def residual(self, other_scp_phase, this_scp_phase):
        pieces = []
        for kind in self.groups:
            si, ti, a, b = self._coefficients(other_scp_phase, this_scp_phase, kind)
            if kind == "control":
                left = other_scp_phase.cp_params.nu_ref[-1, si] + other_scp_phase.dnu[-1, si]
                right = this_scp_phase.cp_params.nu_ref[0, ti] + this_scp_phase.dnu[0, ti]
            else:
                left = other_scp_phase.cp_params.z_ref[-1, si] + other_scp_phase.dz[-1, si]
                right = this_scp_phase.cp_params.z_ref[0, ti] + this_scp_phase.dz[0, ti]
            # Preserve the existing expression for equal-scale configurations.
            pieces.append((left if np.all(a == 1.) else cp.multiply(a, left)) -
                          (right if np.all(b == 1.) else cp.multiply(b, right)))
        return pieces[0] if len(pieces) == 1 else cp.hstack(pieces)

    def accepted_residual(self, source, target):
        pieces = []
        for kind in self.groups:
            si, ti, a, b = self._coefficients(source, target, kind)
            field = "nu_opt" if kind == "control" else "z_opt"
            pieces.append(a * source.current_iter_data[field][-1, si] -
                          b * target.current_iter_data[field][0, ti])
        return np.concatenate(pieces)


class full_continuity(_continuity):
    """Mapped state, control, and time match in physical units."""
    type = "full_continuity"
    groups = ("state", "control", "time")


class state_continuity(_continuity):
    """Mapped physical state matches at the segment boundary."""
    type = "state_continuity"
    groups = ("state",)


class control_continuity(_continuity):
    """Mapped physical control matches at the segment boundary."""
    type = "control_continuity"
    groups = ("control",)


class time_continuity(_continuity):
    """Physical time matches at the segment boundary."""
    type = "time_continuity"
    groups = ("time",)


class dynamics(Constraint):
    def __init__(self, cnstr_config: dict, phase) -> None:
        index_map = phase.index_map
        nondim = phase.nondim
        fcns = phase.fcns

        self.type       = "dynamics"
        self.name       = cnstr_config.name
        self.index_map  = index_map
        self.dimension  = index_map.n.z

        self.fcn_string  = cnstr_config.fcn
        self.fcn_txu_dim = tools.resolve_function_from_string(self.fcn_string, fcns)

        self.M_out_d2nd   = jnp.asarray(nondim.M.state.d2nd * nondim.time_scale)
        self.M_state_nd2d = jnp.asarray(nondim.M.state.nd2d)
        self.M_ctrl_nd2d  = jnp.asarray(nondim.M.control.nd2d)

        self.time_scale     = nondim.time_scale

        self.ctcs_constraints = ()
        self.running_costs = ()

    def fcn_txu_nd(self, x, u, t, params):
        return self.M_out_d2nd @ self.fcn_txu_dim(self.M_state_nd2d @ x, self.M_ctrl_nd2d @ u, self.time_scale * t, params)

    def fcn_znu(self, z, nu, params):
        x, t, _, u, s = self.index_map.unpack_znu(z, nu)

        dx_dt = self.fcn_txu_nd(x, u, t, params)
        dt_dt = jnp.asarray([1.0], dtype=z.dtype)

        n_ctcs = len(self.index_map.indices.z.ctcs)
        if self.ctcs_constraints:
            ctcs_values = jnp.concatenate(
                [jnp.atleast_1d(c.fcn_znu(z, nu, params)) for c in self.ctcs_constraints],
            )
            dbeta_dt = jnp.maximum(ctcs_values, 0.0)**2
        else:
            dbeta_dt = jnp.zeros(n_ctcs, dtype=z.dtype)

        n_rc = len(self.index_map.indices.z.running_cost)
        if self.running_costs:
            dgamma_dt = jnp.concatenate(
                [jnp.atleast_1d(c.fcn_znu(z, nu, params)) for c in self.running_costs],
            )
        else:
            dgamma_dt = jnp.zeros(n_rc, dtype=z.dtype)

        return s * jnp.concatenate([dx_dt, dt_dt, dbeta_dt, dgamma_dt])
