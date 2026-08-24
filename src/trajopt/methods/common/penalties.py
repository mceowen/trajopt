"""Penalty-method building blocks for dev/scvx's constraint penalty machinery.

Groups a constraint's W/dual/vb state into one `Penalty` object -- a hybrid
dataclass/dict (attribute *and* key access, like this codebase's other
AttrDict-based config objects) -- so it can be passed around as a single
argument, and factors the W/dual autotuning and penalty-cost math out of the
constraint classes so it's reusable. Each function below takes the whole
`Penalty` and extracts what it needs, including the standard-vs-split branch.
"""

import numpy as np
import cvxpy as cp

from trajopt.utils.tools import AttrDict


class Penalty(AttrDict):
    """A constraint's penalty state: W/dual weights, vb buffers, and their cvxpy plumbing."""

    def __init__(self, shape, vb_type: str = "standard", norm: str = "l2",
                 cfg=None, nonnegative_dual: bool = False) -> None:
        super().__init__()
        self.shape            = shape
        self.vb_type          = vb_type
        self.norm             = norm
        self.cfg              = cfg
        self.nonnegative_dual = nonnegative_dual

        self.eps = np.atleast_1d(1e-4)

        self.W      = np.zeros(shape)
        self.dual   = np.zeros(shape)
        self.W_p    = np.zeros(shape)
        self.W_m    = np.zeros(shape)
        self.dual_p = np.zeros(shape)
        self.dual_m = np.zeros(shape)

        self.vb   = np.zeros(shape)
        self.vb_p = np.zeros(shape)
        self.vb_m = np.zeros(shape)

        self.W_sqrt_param   = None
        self.dual_param     = None
        self.W_p_sqrt_param = None
        self.W_m_sqrt_param = None
        self.dual_p_param   = None
        self.dual_m_param   = None

        self.vb_var   = None
        self.vb_p_var = None
        self.vb_m_var = None

    def init_values(self) -> None:
        """Fill W/dual with their configured initial values."""
        if not (self.cfg and hasattr(self.cfg, 'W') and self.cfg.W.penalty):
            return
        self.W    = np.full(self.shape, float(self.cfg.W.init))
        self.dual = np.full(self.shape, float(self.cfg.dual.init))
        if self.vb_type == "split":
            self.W_p    = np.full(self.shape, float(self.cfg.W.init))
            self.W_m    = np.full(self.shape, float(self.cfg.W.init))
            self.dual_p = np.full(self.shape, float(self.cfg.dual.init))
            self.dual_m = np.full(self.shape, float(self.cfg.dual.init))


# =============================================================================
# BASELINE AUTOSCVX
# =============================================================================

def _autotune_W(W, vb, eps, cfg):
    eps_target = np.maximum(cfg.fac_target * eps * np.sign(vb), cfg.fac_eps * np.abs(vb))
    with np.errstate(invalid='ignore', divide='ignore'):
        Wh = np.nan_to_num(W * vb * np.sign(vb) / eps_target, nan=0.0, posinf=0.0, neginf=0.0)
    return np.maximum(Wh, cfg.eps_floor)


def _autotune_dual(dual, vb, cfg, nonnegative, W):
    # 'al' (Augmented-Lagrangian style): step size is the current W, element-wise.
    # otherwise: fixed step size cfg.dual.beta (the original autotune1-style update).
    step = W if getattr(cfg.dual, 'style', 'beta') == 'al' else cfg.dual.beta
    dual_new = step * vb + dual
    if nonnegative:
        dual_new = np.maximum(0, dual_new)
    return dual_new


def autotune_W(penalty: Penalty) -> None:
    """Penalty-weight update; mutates penalty.W (and W_p/W_m under split) in place."""
    if penalty.vb_type == "split":
        penalty.W_p = _autotune_W(penalty.W_p, penalty.vb_p, penalty.eps, penalty.cfg)
        penalty.W_m = _autotune_W(penalty.W_m, penalty.vb_m, penalty.eps, penalty.cfg)
    else:
        penalty.W = _autotune_W(penalty.W, penalty.vb, penalty.eps, penalty.cfg)


def autotune_dual(penalty: Penalty) -> None:
    """Dual-ascent update; mutates penalty.dual (and dual_p/dual_m under split) in place.

    cfg.dual.style selects the step size: unset/'beta' (default, fixed cfg.dual.beta,
    the original autotune1-style update) or 'al' (Augmented-Lagrangian style, where
    the step size is the current W, element-wise).
    """
    if penalty.vb_type == "split":
        penalty.dual_p = _autotune_dual(penalty.dual_p, penalty.vb_p, penalty.cfg, penalty.nonnegative_dual, penalty.W_p)
        penalty.dual_m = _autotune_dual(penalty.dual_m, penalty.vb_m, penalty.cfg, penalty.nonnegative_dual, penalty.W_m)
    else:
        penalty.dual = _autotune_dual(penalty.dual, penalty.vb, penalty.cfg, penalty.nonnegative_dual, penalty.W)


def l1_norm(W_param, vb_var):
    """l1 penalty term: W (linear -- l1 params carry W itself, not sqrt(W)) times |vb|."""
    return cp.sum(cp.multiply(W_param, cp.abs(vb_var)))


def l2_norm(W_sqrt_param, vb_var):
    """l2 penalty term: 0.5 * sum((sqrt(W) * vb)^2) == 0.5 * vb^T diag(W) vb.

    W is stored as a vector (one weight per vb component), which *is* the
    diagonal of the quadratic form's weight matrix -- squaring sqrt(W)*vb
    elementwise and summing is diag(W) applied to vb, without materializing
    the (mostly-zero) full matrix.
    """
    return 0.5 * cp.sum_squares(cp.multiply(W_sqrt_param, vb_var))


def _w_cost_buffer(W_param, vb_var, norm):
    if W_param is None:
        return 0
    return l1_norm(W_param, vb_var) if norm == "l1" else l2_norm(W_param, vb_var)


def _dual_cost_buffer(dual_param, vb_var):
    if dual_param is None:
        return 0
    return cp.sum(cp.multiply(dual_param, vb_var))


def w_penalty_cost(penalty: Penalty):
    """Penalty-weight cost term(s), l1 or l2, summed across vb buffer(s)."""
    if penalty.vb_type == "split":
        return (_w_cost_buffer(penalty.W_p_sqrt_param, penalty.vb_p_var, penalty.norm)
                + _w_cost_buffer(penalty.W_m_sqrt_param, penalty.vb_m_var, penalty.norm))
    return _w_cost_buffer(penalty.W_sqrt_param, penalty.vb_var, penalty.norm)


def dual_penalty_cost(penalty: Penalty):
    """Dual (linear) cost term(s), summed across vb buffer(s)."""
    if penalty.vb_type == "split":
        return (_dual_cost_buffer(penalty.dual_p_param, penalty.vb_p_var)
                + _dual_cost_buffer(penalty.dual_m_param, penalty.vb_m_var))
    return _dual_cost_buffer(penalty.dual_param, penalty.vb_var)


def _w_cost_value_buffer(W, vb, norm):
    return float(np.sum(W * np.abs(vb))) if norm == "l1" else float(0.5 * np.sum(W * vb ** 2))


def _dual_cost_value_buffer(dual, vb):
    return float(np.sum(dual * vb))


def penalty_cost_value(penalty: Penalty) -> float:
    """Numpy-evaluated total penalty cost (W-term + dual-term) on the current W/dual/vb --
    the same formula as w_penalty_cost/dual_penalty_cost, but off the solved numpy values
    rather than the cvxpy parameters, for reporting once a step has been taken."""
    if penalty.vb_type == "split":
        return (_w_cost_value_buffer(penalty.W_p, penalty.vb_p, penalty.norm)
                + _w_cost_value_buffer(penalty.W_m, penalty.vb_m, penalty.norm)
                + _dual_cost_value_buffer(penalty.dual_p, penalty.vb_p)
                + _dual_cost_value_buffer(penalty.dual_m, penalty.vb_m))
    return (_w_cost_value_buffer(penalty.W, penalty.vb, penalty.norm)
            + _dual_cost_value_buffer(penalty.dual, penalty.vb))


def _push_values_buffer(W_sqrt_param, dual_param, W, dual, norm):
    if W_sqrt_param is not None:
        W_sqrt_param.value = W if norm == "l1" else np.sqrt(W)
    if dual_param is not None:
        dual_param.value = dual


def push_penalty_values(penalty: Penalty) -> None:
    """Push numpy W/dual onto their cvxpy Parameters (sqrt for l2, raw for l1)."""
    if penalty.vb_type == "split":
        _push_values_buffer(penalty.W_p_sqrt_param, penalty.dual_p_param, penalty.W_p, penalty.dual_p, penalty.norm)
        _push_values_buffer(penalty.W_m_sqrt_param, penalty.dual_m_param, penalty.W_m, penalty.dual_m, penalty.norm)
    else:
        _push_values_buffer(penalty.W_sqrt_param, penalty.dual_param, penalty.W, penalty.dual, penalty.norm)


def pull_vb(penalty: Penalty) -> None:
    """Read the solved vb variable(s) back into penalty.vb (and vb_p/vb_m under split)."""
    if penalty.vb_var is None:
        return
    if penalty.vb_type == "split":
        penalty.vb_p = np.array(penalty.vb_p_var.value)
        penalty.vb_m = np.array(penalty.vb_m_var.value)
        penalty.vb   = penalty.vb_p - penalty.vb_m
    else:
        penalty.vb = np.array(penalty.vb_var.value)


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
