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
