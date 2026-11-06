import numpy as np

from trajopt.methods.common import trust_region


def tighten_on_error(subproblems, exc=None) -> str:
    # tighten the trust region in response to a failed solve, returning a status message
    subproblems = list(subproblems)
    trust_region.tighten_scp_trust_region(subproblems)
    if exc is not None:
        return f"subproblem refused ({exc}), tightening trust region"
    return "tightening trust region"


def adjust_on_step(subproblems) -> tuple[bool, str | None]:
    # tighten the trust region when the step is unusable, else relax it toward target
    subproblems = list(subproblems)
    if not trust_region.step_is_usable(subproblems):
        trust_region.tighten_scp_trust_region(subproblems)
        return False, "step rejected, tightening trust region"
    trust_region.relax_scp_trust_region(subproblems)
    return True, None


def adapt_trust_region_on_step(subproblems) -> tuple[bool, str | None]:
    """Ratio test between the nonlinear and linearized decrease."""
    subproblems = list(subproblems)
    cfg = subproblems[0].hyperparams.trust_region.adaptive
    if not trust_region.step_is_usable(subproblems):
        trust_region.scale_s(subproblems, float(cfg.on_solve_fail))
        return False, "step unusable, shrinking trust region"

    dJ, dL = trust_region.predicted_and_actual_decrease(subproblems)
    if not (np.isfinite(dJ) and np.isfinite(dL)):
        trust_region.scale_s(subproblems, float(cfg.on_solve_fail))
        return False, "non-finite merit decrease, shrinking trust region"
    if dL <= float(cfg.pred_tol):
        return True, None

    ratio = dJ / dL
    band = next(b for b in cfg.bands if ratio < float(b.below))
    trust_region.scale_s(subproblems, float(band.s_factor))
    for sp in subproblems:
        sp.current_iter_data.tr_ratio = ratio
        sp.current_iter_data.s_mult   = sp.s_mult

    if not bool(band.accept):
        return False, f"step rejected (ratio {ratio:.3g}), s scaled by {float(band.s_factor):g}"
    return True, None


def tighten_lm_on_error(subproblems, exc=None) -> str:
    # tighten SQP's Hessian regularization in response to a failed solve
    for sp in subproblems:
        sp.lm_mu = min(max(sp.lm_mu, 1e-6) * 10.0, 1e4)
    if exc is not None:
        return f"subproblem refused ({exc}), tightening trust region"
    return "tightening trust region"


def adjust_lm_on_step(subproblems) -> tuple[bool, str | None]:
    # tighten SQP's Hessian regularization when the step is unusable
    subproblems = list(subproblems)
    if not trust_region.step_is_usable(subproblems):
        for sp in subproblems:
            sp.lm_mu = min(sp.lm_mu * 10.0, 1e4)
        return False, "step rejected, tightening trust region"
    return True, None
