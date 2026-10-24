"""Penalty-method building blocks: weight autotuning, penalty cost, cvxpy param plumbing."""

import numpy as np
import cvxpy as cp
import jax.numpy as jnp

from trajopt.utils.tools import AttrDict


# =============================================================================
# PENALTY ALLOCATION -- a constraint's penalty state: weights named by hyperparams.penalties,
# vb buffers, cvxpy plumbing. Shared shape used by every variation below.
# =============================================================================

def _param_name(name, suffix=""):
    return f"{name}{suffix}_sqrt_param" if name == "W" else f"{name}{suffix}_param"


def _is_split(vb_type):
    """Return whether the buffer uses separate positive and negative parts."""
    return vb_type in ("split", "sparse")


def _suffixes(penalty):
    """Return ``('_p', '_m')`` for split or sparse buffers, or ``('',)`` otherwise."""
    return ("_p", "_m") if _is_split(penalty.vb_type) else ("",)


def alloc_penalty(shape, vb_type: str = "standard", norm: str = "l2",
                   cfg=None, nonnegative_dual: bool = False) -> AttrDict:
    """A constraint's penalty state: weights named by hyperparams.penalties, vb buffers, cvxpy plumbing."""
    p = AttrDict()
    p.shape, p.vb_type, p.norm = shape, vb_type, norm
    p.cfg, p.nonnegative_dual = cfg, nonnegative_dual
    p.eps = np.atleast_1d(1e-4)

    p.names = [name for name, spec in (cfg or {}).items()
               if hasattr(spec, 'items') and 'init' in spec]

    for name in p.names:
        p[name] = np.zeros(shape)
        p[_param_name(name)] = None
        if _is_split(vb_type):
            p[f"{name}_p"] = np.zeros(shape)
            p[f"{name}_m"] = np.zeros(shape)
            p[_param_name(name, "_p")] = None
            p[_param_name(name, "_m")] = None

    p.vb, p.vb_var = np.zeros(shape), None
    if _is_split(vb_type):
        p.vb_p, p.vb_p_var = np.zeros(shape), None
        p.vb_m, p.vb_m_var = np.zeros(shape), None
    return p


def reset_penalty_values(p) -> None:
    """Reset penalties and buffers, preserving CP references."""
    parts = p.get('parts')
    if parts is not None:
        for part in parts.values():
            reset_penalty_values(part)
        p.vb = np.zeros(p.shape)
        return
    for name in ['vb'] + p.names:
        p[name] = np.zeros(p.shape)
        if _is_split(p.vb_type):
            p[f'{name}_p'] = np.zeros(p.shape)
            p[f'{name}_m'] = np.zeros(p.shape)
    init_penalty_values(p)
    if p.get('v_penalties') is not None:
        reset_penalty_values(p.v_penalties)


def capture_penalty_state(p):
    """Copy numerical state, including nested penalty parts and sparse aggregates."""
    state = {key: value.copy() for key, value in p.items() if isinstance(value, np.ndarray)}
    if p.get('parts') is not None:
        state['parts'] = {name: capture_penalty_state(part) for name, part in p.parts.items()}
    if p.get('v_penalties') is not None:
        state['v_penalties'] = capture_penalty_state(p.v_penalties)
    return state


def install_penalty_state(p, state):
    """Restore numerical values without replacing CVXPY objects or configuration."""
    for key, value in state.items():
        if key == 'parts':
            for name, part in value.items():
                install_penalty_state(p.parts[name], part)
        elif key == 'v_penalties':
            install_penalty_state(p.v_penalties, value)
        else:
            p[key] = np.array(value, copy=True)


def penalty_descriptor(p):
    """Copy the metadata needed to interpret penalty buffer histories."""
    data = dict(eps=np.array(p.eps, copy=True), vb_type=p.vb_type)
    if p.get('parts') is not None:
        data['parts'] = {name: penalty_descriptor(part) for name, part in p.parts.items()}
        data['part_indices'] = {name: np.array(idx, copy=True) for name, idx in p.part_indices.items()}
    if p.get('v_penalties') is not None:
        data['v_penalties'] = penalty_descriptor(p.v_penalties)
    return data


def init_penalty_values(p) -> None:
    if not (p.cfg and 'W' in p.names and p.cfg.W.penalty):
        return
    for name in p.names:
        init = float(p.cfg[name].init)
        p[name] = np.full(p.shape, init)
        if _is_split(p.vb_type):
            p[f"{name}_p"] = np.full(p.shape, init)
            p[f"{name}_m"] = np.full(p.shape, init)


# =============================================================================
# PENALTY PARTS -- one constraint's buffer split across named z-component index ranges
# (hyperparams.penalties.<type>.indices), each part its own plain penalty with its own vb_type.
# =============================================================================

def _resolve_part_indices(index_map, indices_map, n_cols):
    part_indices, vb_type_by_part = {}, {}
    covered = set()
    for name, spec in indices_map.items():
        if name == "default":
            continue
        idx = np.atleast_1d(getattr(index_map.indices.z, name)).astype(int)
        part_indices[name] = idx
        vb_type_by_part[name] = spec.vb
        covered.update(int(i) for i in idx)

    if "default" in indices_map:
        part_indices["default"] = np.array([i for i in range(n_cols) if i not in covered], dtype=int)
        vb_type_by_part["default"] = indices_map["default"].vb

    return part_indices, vb_type_by_part


def alloc_penalty_parts(scp_subproblem, shape, indices_map, norm, cfg, nonnegative_dual, *, eps) -> AttrDict:
    """Split a constraint buffer into named parts by component indices in ``z``.

    Each part has a penalty dictionary and a ``vb_type``. The combined
    ``vb_var`` uses the original column order.
    """
    N, n_cols = shape
    part_indices, vb_type_by_part = _resolve_part_indices(scp_subproblem.index_map, indices_map, n_cols)

    gp = AttrDict()
    gp.shape, gp.nonnegative_dual = shape, nonnegative_dual
    gp.vb_type = "parts"
    gp.vb = np.zeros(shape)
    gp.vb_var = None
    gp.vb_p_var = None
    gp.parts = AttrDict()
    gp.part_indices = part_indices
    gp.eps = np.broadcast_to(eps, (n_cols,)).astype(float).copy()
    for name, idx in part_indices.items():
        p = alloc_penalty(shape=(N, len(idx)), vb_type=vb_type_by_part[name], norm=norm,
                           cfg=cfg, nonnegative_dual=nonnegative_dual)
        p.eps = gp.eps[idx].copy()
        init_penalty_values(p)
        gp.parts[name] = p
    return gp


def _place_columns(parts, part_indices, shape):
    """Place each part's columns in a combined ``(N, n_cols)`` expression."""
    N, n_cols = shape
    result = cp.Constant(np.zeros((N, n_cols)))
    for name, idx in part_indices.items():
        part = parts.get(name)
        if part is None:
            continue
        S = np.zeros((n_cols, len(idx)))
        for j, col in enumerate(idx):
            S[int(col), j] = 1.0
        result = result + part @ S.T
    return result


def create_penalty_part_variables(gp, scp_subproblem, cname) -> None:
    parts = {}
    for name, p in gp.parts.items():
        pcname = f"{cname}_{name}"
        if p.vb_type == "none":
            p.vb_var = None
        elif _is_split(p.vb_type):
            p.vb_p_var = cp.Variable(p.shape, nonneg=True, name=f"vb_p_{pcname}_{scp_subproblem.name}")
            p.vb_m_var = cp.Variable(p.shape, nonneg=True, name=f"vb_m_{pcname}_{scp_subproblem.name}")
            p.vb_var   = p.vb_p_var - p.vb_m_var
        else:
            p.vb_var = cp.Variable(p.shape, name=f"vb_{pcname}_{scp_subproblem.name}")
        parts[name] = p.vb_var

    gp.vb_var = _place_columns(parts, gp.part_indices, gp.shape)


# =============================================================================
# PENALTY FUNCTIONS -- l1/l2/linear cost terms, in cvxpy form (for the subproblem) and raw-array
# form (for merit eval and reporting). norm_cp/norm_jax dispatch by name instead of branching.
# =============================================================================

def l1_norm(W_param, vb_var):
    """l1 term: W * |vb| (l1 params carry W directly, not sqrt(W))."""
    return cp.sum(cp.multiply(W_param, cp.abs(vb_var)))


def l2_norm(W_sqrt_param, vb_var):
    """l2 term: 0.5 * sum((sqrt(W) * vb)^2)."""
    return 0.5 * cp.sum_squares(cp.multiply(W_sqrt_param, vb_var))


def linear_term(param, vb_var):
    """Return the linear penalty term ``sum(param * vb_var)``."""
    return cp.sum(cp.multiply(param, vb_var))


def l1_norm_jax(W, viol):
    return jnp.sum(W * jnp.abs(viol))


def l2_norm_jax(W, viol):
    return 0.5 * jnp.sum(W * viol ** 2)


def linear_term_jax(param, viol):
    return jnp.sum(param * viol)


norm_cp  = {"l1": l1_norm,     "l2": l2_norm}
norm_jax = {"l1": l1_norm_jax, "l2": l2_norm_jax}


def l1_reg_jax(w, traj):
    """Return the regularization cost ``w * sum(abs(traj))``."""
    return w * jnp.sum(jnp.abs(traj))


def l2_reg_jax(w, traj):
    """Return the regularization cost ``w * sum(traj ** 2)``."""
    return w * jnp.sum(traj ** 2)


reg_norm_jax = {"l1": l1_reg_jax, "l2": l2_reg_jax}


def _cost_term(param, vb_var, is_w, norm):
    if param is None:
        return 0
    return norm_cp[norm](param, vb_var) if is_w else linear_term(param, vb_var)


def _cost_value_term(weight, vb, is_w, norm):
    return float(norm_jax[norm](weight, vb)) if is_w else float(linear_term_jax(weight, vb))


# =============================================================================
# PENALTY READBACK -- generic, used by every flavor below: pulling solved vb values back into
# numpy, and the numpy-evaluated penalty cost used for iteration reporting.
# =============================================================================

def pull_vb(penalty) -> None:
    parts = penalty.get('parts')
    if parts is not None:
        for p in parts.values():
            pull_vb(p)
        composite = np.zeros(penalty.shape)
        for name, idx in penalty.part_indices.items():
            composite[:, idx] = parts[name].vb
        penalty.vb = composite
        return
    for g in _suffixes(penalty):
        var = penalty.get(f"vb{g}_var")
        if var is not None:
            penalty[f"vb{g}"] = np.array(var.value)
    if _is_split(penalty.vb_type):
        penalty.vb = penalty.vb_p - penalty.vb_m
    v = penalty.get('v_penalties')
    if v is not None:
        pull_vb(v)


def penalty_cost_value(penalty) -> float:
    """Numpy-evaluated penalty cost, off the solved values, for reporting."""
    parts = penalty.get('parts')
    if parts is not None:
        return sum(penalty_cost_value(p) for p in parts.values())

    if penalty.get('w_param') is not None:
        # prox-linear: static w, no dual, see penalty_cost_proxlin
        return float(np.sum(penalty.cfg.w * np.abs(penalty.vb))) if penalty.cfg else 0.0

    total = 0.0
    for name in penalty.names:
        is_w = name == "W"
        for g in _suffixes(penalty):
            total += _cost_value_term(penalty[f"{name}{g}"], penalty[f"vb{g}"], is_w, penalty.norm)
    v = penalty.get('v_penalties')
    if v is not None:
        total += penalty_cost_value(v)
    return total


# =============================================================================
# AUTOSCVX -- the autotuned quadratic-weight / dual augmented-Lagrangian scheme.
# =============================================================================

def autotune_quad(penalty) -> None:
    """If ``W.autotune`` is set, update the quadratic penalty weight."""
    cfg = penalty.cfg
    if not (cfg and cfg.W.autotune):
        return
    for g in _suffixes(penalty):
        vb = penalty[f"vb{g}"]
        W  = penalty[f"W{g}"]
        eps_target = np.maximum(cfg.fac_target * penalty.eps, cfg.fac_eps * np.abs(vb))
        with np.errstate(invalid='ignore', divide='ignore'):
            Wh = np.nan_to_num(W * vb / (eps_target * np.sign(vb)), nan=0.0, posinf=0.0, neginf=0.0)
        penalty[f"W{g}"] = np.maximum(Wh, cfg.eps_floor)


def autotune_dual(penalty) -> None:
    """If ``dual.autotune`` is set, update the dual weight.

    ``tr_step`` sets a fixed step size. If ``tr_step`` is ``'al'``, use the
    current quadratic weight as the step size.
    """
    cfg = penalty.cfg
    if not (cfg and cfg.dual.autotune):
        return
    tr_step = cfg.dual.tr_step
    for g in _suffixes(penalty):
        vb   = penalty[f"vb{g}"]
        step = penalty[f"W{g}"] if tr_step == 'al' else tr_step
        dual_new = step * vb + penalty[f"dual{g}"]
        penalty[f"dual{g}"] = np.maximum(0, dual_new) if penalty.nonnegative_dual else dual_new


def autotune_std(constraint, scp_subproblem) -> None:
    autotune_quad(constraint.penalties)
    autotune_dual(constraint.penalties)


def autotune_quad_only(constraint, scp_subproblem) -> None:
    """Update the quadratic weight without changing the dual weight."""
    autotune_quad(constraint.penalties)


def autotune_dual_only(constraint, scp_subproblem) -> None:
    """Update the dual weight without changing the quadratic weight."""
    autotune_dual(constraint.penalties)


def _accumulate_penalty_cost(penalty, scp_subproblem) -> None:
    for name in penalty.names:
        is_w = name == "W"
        for g in _suffixes(penalty):
            param  = penalty.get(_param_name(name, g))
            vb_var = penalty.get(f"vb{g}_var")
            scp_subproblem.cp_cost += _cost_term(param, vb_var, is_w, penalty.norm)


def penalty_cost_std(constraint, scp_subproblem) -> None:
    _accumulate_penalty_cost(constraint.penalties, scp_subproblem)


def _create_penalty_params(penalty, cname) -> None:
    if penalty.vb_type == "none":
        return
    shape = penalty.shape
    for name in penalty.names:
        is_w = name == "W"
        for g in _suffixes(penalty):
            cp_name = f"{name}{g}_{cname}" + ("_sqrt" if is_w else "")
            penalty[_param_name(name, g)] = cp.Parameter(shape, nonneg=is_w, name=cp_name, value=np.zeros(shape))


def create_params_std(constraint, scp_subproblem) -> None:
    if constraint.shape is None:
        return
    _create_penalty_params(constraint.penalties, constraint.name)


def push_penalty_values(penalty) -> None:
    for name in penalty.names:
        is_w = name == "W"
        for g in _suffixes(penalty):
            param = penalty.get(_param_name(name, g))
            if param is None:
                continue
            raw = penalty[f"{name}{g}"]
            param.value = raw if (not is_w or penalty.norm == "l1") else np.sqrt(raw)


def update_params_std(constraint, scp_subproblem) -> None:
    push_penalty_values(constraint.penalties)


# =============================================================================
# SPARSE-AUTOSCVX -- each part (or, with no parts, the constraint's whole buffer) reuses the
# plain AUTOSCVX machinery above, plus its own v_plus/v_minus aggregate.
# =============================================================================

def _parts_or_self(constraint):
    """Return the named penalty parts, or the complete penalty as a single item."""
    parts = constraint.penalties.get('parts')
    return parts.values() if parts is not None else [constraint.penalties]


def autotune_sparse(constraint, scp_subproblem) -> None:
    for p in _parts_or_self(constraint):
        autotune_quad(p)
        autotune_dual(p)
        v = p.get('v_penalties')
        if v is not None:
            autotune_quad(v)
            autotune_dual(v)


def penalty_cost_sparse(constraint, scp_subproblem) -> None:
    for p in _parts_or_self(constraint):
        _accumulate_penalty_cost(p, scp_subproblem)
        v = p.get('v_penalties')
        if v is not None:
            _accumulate_penalty_cost(v, scp_subproblem)


def create_params_sparse(constraint, scp_subproblem) -> None:
    """Create penalty parameters for each buffer part.

    ``constraints_sparse`` creates the aggregate buffers and their parameters.
    """
    if constraint.shape is None:
        return
    parts = constraint.penalties.get('parts')
    if parts is None:
        _create_penalty_params(constraint.penalties, constraint.name)
        return
    for name, p in parts.items():
        _create_penalty_params(p, f"{constraint.name}_{name}")


def update_params_sparse(constraint, scp_subproblem) -> None:
    for p in _parts_or_self(constraint):
        push_penalty_values(p)
        v = p.get('v_penalties')
        if v is not None:
            push_penalty_values(v)


def constraints_sparse(constraint, scp_subproblem) -> None:
    """Sum each sparse part's positive and negative buffers across columns.

    Each sum has one value per node and its own quadratic and dual weights.
    """
    parts = constraint.penalties.get('parts')
    if parts is not None:
        for name, p in parts.items():
            _create_sparse_aggregate(p, scp_subproblem, cname=f"v_{constraint.name}_{name}")
        return
    _create_sparse_aggregate(constraint.penalties, scp_subproblem, cname=f"v_{constraint.name}")


def _create_sparse_aggregate(p, scp_subproblem, cname) -> None:
    if p.vb_type != "sparse" or p.vb_p_var is None:
        return

    N = p.vb_p_var.shape[0]

    v = alloc_penalty(shape=(N,), vb_type="sparse", norm=p.norm, cfg=p.cfg, nonnegative_dual=True)
    v.eps = np.atleast_1d(np.sum(np.broadcast_to(p.eps, (p.shape[1],))))
    init_penalty_values(v)

    v.vb_p_var = cp.Variable(N, nonneg=True, name=f"vb_p_{cname}_{scp_subproblem.name}")
    v.vb_m_var = cp.Variable(N, nonneg=True, name=f"vb_m_{cname}_{scp_subproblem.name}")
    v.vb_var   = v.vb_p_var - v.vb_m_var

    scp_subproblem.cp_constraints.append(cp.sum(p.vb_p_var, axis=1) == v.vb_p_var)
    scp_subproblem.cp_constraints.append(cp.sum(p.vb_m_var, axis=1) == v.vb_m_var)

    _create_penalty_params(v, cname)
    p.v_penalties = v


def constraints_sparse_scalar(constraint, scp_subproblem) -> None:
    """Sum each sparse part's positive and negative buffers across columns and nodes.

    Each part has one pair of scalar sums with separate penalty weights.
    """
    parts = constraint.penalties.get('parts')
    if parts is not None:
        for name, p in parts.items():
            _create_sparse_aggregate_scalar(p, scp_subproblem, cname=f"v_{constraint.name}_{name}")
        return
    _create_sparse_aggregate_scalar(constraint.penalties, scp_subproblem, cname=f"v_{constraint.name}")


def _create_sparse_aggregate_scalar(p, scp_subproblem, cname) -> None:
    if p.vb_type != "sparse" or p.vb_p_var is None:
        return

    v = alloc_penalty(shape=(1,), vb_type="sparse", norm=p.norm, cfg=p.cfg, nonnegative_dual=True)
    v.eps = np.atleast_1d(p.shape[0] * np.sum(np.broadcast_to(p.eps, (p.shape[1],))))
    init_penalty_values(v)

    v.vb_p_var = cp.Variable(1, nonneg=True, name=f"vb_p_{cname}_{scp_subproblem.name}")
    v.vb_m_var = cp.Variable(1, nonneg=True, name=f"vb_m_{cname}_{scp_subproblem.name}")
    v.vb_var   = v.vb_p_var - v.vb_m_var

    scp_subproblem.cp_constraints.append(cp.reshape(cp.sum(p.vb_p_var), (1,)) == v.vb_p_var)
    scp_subproblem.cp_constraints.append(cp.reshape(cp.sum(p.vb_m_var), (1,)) == v.vb_m_var)

    _create_penalty_params(v, cname)
    p.v_penalties = v


# =============================================================================
# PROX-LINEAR -- a static l1 weight w, never autotuned, no dual at all.
# =============================================================================

def penalty_cost_proxlin(constraint, scp_subproblem) -> None:
    p = constraint.penalties
    if p.get('w_param') is None or p.vb_var is None:
        return
    scp_subproblem.cp_cost += l1_norm(p.w_param, p.vb_var)


def create_params_proxlin(constraint, scp_subproblem) -> None:
    if constraint.shape is None:
        return
    p = constraint.penalties
    p.w_param = cp.Parameter(p.shape, nonneg=True, name=f"w_{constraint.name}", value=np.zeros(p.shape))


def update_params_proxlin(constraint, scp_subproblem) -> None:
    p = constraint.penalties
    if p.get('w_param') is None:
        return
    p.w_param.value = np.full(p.shape, float(p.cfg.w))


# =============================================================================
# EXPERIMENTAL AUTOTUNING EXTENSIONS -- constants come from hyperparams.penalties.<type>.experimental.
# =============================================================================

def _rho_b(penalty, exp, iter_num, settled) -> float:
    """Return the scheduled weight for the experimental quadratic update.

    The weight decreases from one to zero over ``exp.freeze_iters`` iterations.
    After that, use ``exp.rho_stall`` if the iterate is settled but infeasible.
    """
    rho = max(0.0, 1.0 - iter_num / exp.freeze_iters)
    is_feasible = bool(np.all(np.abs(penalty.vb) <= penalty.eps))
    if rho == 0.0 and settled and not is_feasible:
        rho = exp.rho_stall
    return rho


def _autotune_W_b_split(W, vb, eps, rho, exp):
    """Move the split buffer's weight toward its ratio target by ``rho``.

    Limit the result to ``exp.split.clip``.
    """
    Wh = W * vb / (exp.split.eps_scale * eps)
    return np.clip(W + rho * (Wh - W), exp.split.clip[0], exp.split.clip[1])


def _autotune_W_b_standard(W, vb, eps, rho, exp):
    """Scale the standard buffer's weight by the residual ratio raised to a damped exponent.

    ``rho`` scales the exponent. Limit the result to ``exp.standard.clip``.
    """
    damp = exp.standard.damp_scale * rho
    ratio = np.abs(vb) / (exp.standard.eps_scale * eps)
    Wh = W * np.power(ratio, damp)
    return np.clip(Wh, exp.standard.clip[0], exp.standard.clip[1])


def autotune_W_b(penalty, iter_num, settled=False) -> None:
    """If ``W.autotune`` is set, apply the experimental quadratic weight update.

    Select the update for the buffer type. Skip configurations without ``W``.
    """
    cfg = penalty.cfg
    if not (cfg and hasattr(cfg, 'W') and cfg.W.autotune):
        return
    exp = cfg.experimental
    rho = _rho_b(penalty, exp, iter_num, settled)
    if penalty.vb_type == "split":
        penalty.W_p = _autotune_W_b_split(penalty.W_p, penalty.vb_p, penalty.eps, rho, exp)
        penalty.W_m = _autotune_W_b_split(penalty.W_m, penalty.vb_m, penalty.eps, rho, exp)
    else:
        penalty.W = _autotune_W_b_standard(penalty.W, penalty.vb, penalty.eps, rho, exp)


def autotune_dual_b(penalty, lagrangian_dual=None) -> None:
    """If ``dual.autotune`` is set, apply the experimental dual update.

    Split buffers use the fixed ``dual_step``. Other buffers use the
    constraint's accumulated Lagrangian dual, if supplied.
    """
    cfg = penalty.cfg
    if not (cfg and hasattr(cfg, 'dual') and cfg.dual.autotune):
        return
    if penalty.vb_type == "split":
        dual_step = cfg.experimental.dual_step
        penalty.dual_p = penalty.dual_p + dual_step * penalty.vb_p
        penalty.dual_m = penalty.dual_m + dual_step * penalty.vb_m
    elif lagrangian_dual is not None:
        penalty.dual = np.maximum(0.0, lagrangian_dual) if penalty.nonnegative_dual else lagrangian_dual.copy()


def sqp_autotune(constraint, scp_subproblem) -> None:
    """Apply the experimental quadratic and dual weight updates for SQP."""
    iter_num = scp_subproblem.current_iter_data.iter_num
    chk      = scp_subproblem.current_iter_data.get("chk", None)
    settled  = chk is not None and float(chk.dz) < 1.0
    autotune_W_b(constraint.penalties, iter_num, settled)
    autotune_dual_b(constraint.penalties, getattr(constraint, 'lagrangian_dual', None))


def penalty_cost_sqp(constraint, scp_subproblem) -> None:
    """Add the prox-linear penalty and accumulated dual term to the SQP cost.

    These terms match ``compile_merit_proxlin``. The dual term remains zero
    unless ``dual.autotune`` is set. ``penalty_cost_proxlin`` omits the dual term.
    """
    p = constraint.penalties
    if p.get('w_param') is None or p.vb_var is None:
        return
    scp_subproblem.cp_cost += l1_norm(p.w_param, p.vb_var) + linear_term(p.dual_param, p.vb_var)


def create_params_sqp(constraint, scp_subproblem) -> None:
    if constraint.shape is None:
        return
    p = constraint.penalties
    p.w_param    = cp.Parameter(p.shape, nonneg=True, name=f"w_{constraint.name}", value=np.zeros(p.shape))
    p.dual_param = cp.Parameter(p.shape, name=f"dual_{constraint.name}", value=np.zeros(p.shape))


def update_params_sqp(constraint, scp_subproblem) -> None:
    p = constraint.penalties
    if p.get('w_param') is None:
        return
    p.w_param.value    = np.full(p.shape, float(p.cfg.w))
    p.dual_param.value = p.dual
