"""Penalty-method building blocks: weight autotuning, penalty cost, cvxpy param plumbing."""

import numpy as np
import cvxpy as cp

from trajopt.utils.tools import AttrDict


def _param_name(name, suffix=""):
    return f"{name}{suffix}_sqrt_param" if name == "W" else f"{name}{suffix}_param"


def _groups(penalty):
    return ("_p", "_m") if penalty.vb_type == "split" else ("",)


_VB     = {"": "vb",     "_p": "vb_p",     "_m": "vb_m"}
_VB_VAR = {"": "vb_var", "_p": "vb_p_var", "_m": "vb_m_var"}


class Penalties(AttrDict):
    """A constraint's penalty state: weights named by hyperparams.penalties, vb buffers, cvxpy plumbing."""

    def __init__(self, shape, vb_type: str = "standard", norm: str = "l2",
                 cfg=None, nonnegative_dual: bool = False) -> None:
        super().__init__()
        self.shape, self.vb_type, self.norm = shape, vb_type, norm
        self.cfg, self.nonnegative_dual = cfg, nonnegative_dual
        self.eps = np.atleast_1d(1e-4)

        self.names = [name for name, spec in (cfg or {}).items() if hasattr(spec, 'items')]

        for name in self.names:
            self[name] = np.zeros(shape)
            self[_param_name(name)] = None
            if vb_type == "split":
                self[f"{name}_p"] = np.zeros(shape)
                self[f"{name}_m"] = np.zeros(shape)
                self[_param_name(name, "_p")] = None
                self[_param_name(name, "_m")] = None

        self.vb, self.vb_var = np.zeros(shape), None
        if vb_type == "split":
            self.vb_p, self.vb_p_var = np.zeros(shape), None
            self.vb_m, self.vb_m_var = np.zeros(shape), None

    def init_values(self) -> None:
        if not (self.cfg and 'W' in self.names and self.cfg.W.penalty):
            return
        for name in self.names:
            init = float(self.cfg[name].init)
            self[name] = np.full(self.shape, init)
            if self.vb_type == "split":
                self[f"{name}_p"] = np.full(self.shape, init)
                self[f"{name}_m"] = np.full(self.shape, init)


def noop(*args, **kwargs) -> None:
    pass


# =============================================================================
# BASELINE AUTOSCVX
# =============================================================================

def _autotune_W(W, vb, eps, cfg):
    eps_target = np.maximum(cfg.fac_target * eps, cfg.fac_eps * np.abs(vb))
    with np.errstate(invalid='ignore', divide='ignore'):
        Wh = np.nan_to_num(W * vb / (eps_target  * np.sign(vb)), nan=0.0, posinf=0.0, neginf=0.0)
    return np.maximum(Wh, cfg.eps_floor)


def autotune_W(penalty: Penalties) -> None:
    for g in _groups(penalty):
        penalty[f"W{g}"] = _autotune_W(penalty[f"W{g}"], penalty[_VB[g]], penalty.eps, penalty.cfg)


def _autotune_dual(dual, vb, cfg, nonnegative, W):
    # style 'al' steps by the current W; else a fixed cfg.dual.beta
    step = W if getattr(cfg.dual, 'style', 'beta') == 'al' else cfg.dual.beta
    dual_new = step * vb + dual
    return np.maximum(0, dual_new) if nonnegative else dual_new


def autotune_dual(penalty: Penalties) -> None:
    for g in _groups(penalty):
        penalty[f"dual{g}"] = _autotune_dual(penalty[f"dual{g}"], penalty[_VB[g]], penalty.cfg,
                                              penalty.nonnegative_dual, penalty[f"W{g}"])


def autotune(constraint, scp_subproblem) -> None:
    cfg = constraint.penalty
    if cfg is None or not hasattr(cfg, 'W'):
        return
    if cfg.W.autotune:
        autotune_W(constraint.penalties)
    if cfg.dual.autotune:
        autotune_dual(constraint.penalties)


def l1_norm(W_param, vb_var):
    """l1 penalty term: W (linear -- l1 params carry W itself, not sqrt(W)) times |vb|."""
    return cp.sum(cp.multiply(W_param, cp.abs(vb_var)))


def l2_norm(W_sqrt_param, vb_var):
    """l2 penalty term: 0.5 * sum((sqrt(W) * vb)^2) == 0.5 * vb^T diag(W) vb."""
    return 0.5 * cp.sum_squares(cp.multiply(W_sqrt_param, vb_var))


def _cost_term(param, vb_var, is_w, norm):
    if param is None:
        return 0
    if is_w:
        return l1_norm(param, vb_var) if norm == "l1" else l2_norm(param, vb_var)
    return cp.sum(cp.multiply(param, vb_var))


def penalty_cost(constraint, scp_subproblem) -> None:
    penalty = constraint.penalties
    for name in penalty.names:
        is_w = name == "W"
        for g in _groups(penalty):
            param  = penalty.get(_param_name(name, g))
            vb_var = penalty.get(_VB_VAR[g])
            scp_subproblem.cp_cost += _cost_term(param, vb_var, is_w, penalty.norm)


def _cost_value_term(weight, vb, is_w, norm):
    if is_w:
        return float(np.sum(weight * np.abs(vb))) if norm == "l1" else float(0.5 * np.sum(weight * vb ** 2))
    return float(np.sum(weight * vb))


def penalty_cost_value(penalty: Penalties) -> float:
    """Numpy-evaluated penalty cost, off the solved values, for reporting."""
    total = 0.0
    for name in penalty.names:
        is_w = name == "W"
        for g in _groups(penalty):
            total += _cost_value_term(penalty[f"{name}{g}"], penalty[_VB[g]], is_w, penalty.norm)
    return total


def create_params(constraint, scp_subproblem) -> None:
    if constraint.shape is None:
        return
    penalty = constraint.penalties
    if penalty.vb_type == "none":
        return
    shape, cname = penalty.shape, constraint.name
    for name in penalty.names:
        is_w = name == "W"
        for g in _groups(penalty):
            cp_name = f"{name}{g}_{cname}" + ("_sqrt" if is_w else "")
            penalty[_param_name(name, g)] = cp.Parameter(shape, nonneg=is_w, name=cp_name, value=np.zeros(shape))


def push_penalty_values(penalty: Penalties) -> None:
    for name in penalty.names:
        is_w = name == "W"
        for g in _groups(penalty):
            param = penalty.get(_param_name(name, g))
            if param is None:
                continue
            raw = penalty[f"{name}{g}"]
            param.value = raw if (not is_w or penalty.norm == "l1") else np.sqrt(raw)


def update_params(constraint, scp_subproblem) -> None:
    push_penalty_values(constraint.penalties)


def pull_vb(penalty: Penalties) -> None:
    for g in _groups(penalty):
        var = penalty.get(_VB_VAR[g])
        if var is not None:
            penalty[_VB[g]] = np.array(var.value)
    if penalty.vb_type == "split":
        penalty.vb = penalty.vb_p - penalty.vb_m


# =============================================================================
# EXPERIMENTAL AUTOTUNING EXTENSIONS
# =============================================================================

def _rho_b(obj, iter_num, settled, freeze_iters):
    rho = max(0.0, 1.0 - iter_num / freeze_iters)
    is_feasible = bool(np.all(np.abs(obj.vb) <= obj.eps))
    if rho == 0.0 and settled and not is_feasible:
        rho = 0.05
    return rho


def _autotune_W_b_split(W, vb, eps, rho):
    Wh = W * vb / (0.9 * eps)
    return np.clip(W + rho * (Wh - W), 0.0001, 1e8)


def _autotune_W_b_standard(W, vb, eps, rho):
    damp = 0.9 * rho
    ratio = np.abs(vb) / (0.01 * eps)
    Wh = W * np.power(ratio, damp)
    return np.clip(Wh, 0.00001, 1e7)


def autotune_W_b(obj, iter_num, settled=False, freeze_iters=100.0):
    rho = _rho_b(obj, iter_num, settled, freeze_iters)
    if obj.vb_type == "split":
        obj.W_p = _autotune_W_b_split(obj.W_p, obj.vb_p, obj.eps, rho)
        obj.W_m = _autotune_W_b_split(obj.W_m, obj.vb_m, obj.eps, rho)
    else:
        obj.W = _autotune_W_b_standard(obj.W, obj.vb, obj.eps, rho)


def autotune_dual_b(obj, lagrangian_dual=None):
    if obj.vb_type == "split":
        obj.dual_p = obj.dual_p + 0.1 * obj.vb_p
        obj.dual_m = obj.dual_m + 0.1 * obj.vb_m
    elif lagrangian_dual is not None:
        if obj.nonnegative_dual:
            obj.dual = np.maximum(0.0, lagrangian_dual)
        else:
            obj.dual = lagrangian_dual.copy()


def sqp_autotune(constraint, scp_subproblem) -> None:
    cfg = constraint.penalty
    if cfg is None or not hasattr(cfg, 'W'):
        return
    iter_num = scp_subproblem.current_iter_data.iter_num
    chk      = scp_subproblem.current_iter_data.get("chk", None)
    settled  = chk is not None and float(chk.dz) < 1.0
    if cfg.W.autotune:
        autotune_W_b(constraint.penalties, iter_num, settled)
    if cfg.dual.autotune:
        autotune_dual_b(constraint.penalties, getattr(constraint, 'lagrangian_dual', None))
