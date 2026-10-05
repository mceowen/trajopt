import numpy as np
import cvxpy as cp
import jax
import jax.numpy as jnp

from trajopt.methods.common import penalties


# refactor scvx

def resolve_tr_step(hyperparams, group: str, key: str) -> float:
    # tr_step.<group> is a scalar, or a mapping with a 'default' plus named overrides
    raw = hyperparams.trust_region.tr_step[group]
    if not hasattr(raw, 'items'):
        return float(raw)
    return float(raw.get(key, raw.default))


def resolve_tr_param(self, group: str, key: str) -> float:
    """Configured step size, rescaled by tr_scale (only != 1.0 after a tightening retry)."""
    return self.tr_scale / resolve_tr_step(self.hyperparams, group, key)


def create_tr_params_component(self) -> None:
    """One cvxpy Parameter per z/nu component group, from trust_region.tr_step."""
    self.cp_params.tr_x = cp.Parameter(nonneg=True, name="tr_x", value=resolve_tr_param(self, 'z', 'x'))
    self.cp_params.tr_t = cp.Parameter(nonneg=True, name="tr_t", value=resolve_tr_param(self, 'z', 't'))
    if self.index_map.n.ctcs > 0:
        self.cp_params.tr_ctcs = cp.Parameter(nonneg=True, name="tr_ctcs", value=resolve_tr_param(self, 'z', 'ctcs'))
    if self.index_map.n.running_cost > 0:
        self.cp_params.tr_gamma = cp.Parameter(nonneg=True, name="tr_gamma", value=resolve_tr_param(self, 'z', 'gamma'))
    self.cp_params.tr_u = cp.Parameter(nonneg=True, name="tr_u", value=resolve_tr_param(self, 'nu', 'u'))
    self.cp_params.tr_s = cp.Parameter(nonneg=True, name="tr_s", value=resolve_tr_param(self, 'nu', 's'))


def update_tr_params_component(self) -> None:
    self.cp_params.tr_x.value = resolve_tr_param(self, 'z', 'x')
    self.cp_params.tr_t.value = resolve_tr_param(self, 'z', 't')
    if self.index_map.n.ctcs > 0:
        self.cp_params.tr_ctcs.value = resolve_tr_param(self, 'z', 'ctcs')
    if self.index_map.n.running_cost > 0:
        self.cp_params.tr_gamma.value = resolve_tr_param(self, 'z', 'gamma')
    self.cp_params.tr_u.value = resolve_tr_param(self, 'nu', 'u')
    self.cp_params.tr_s.value = resolve_tr_param(self, 'nu', 's')


def create_tr_params_scalar(self) -> None:
    """One cvxpy Parameter each for z and nu, plus the LM state and compiled merit functions sqp's trust region needs."""
    self.cp_params.tr_z  = cp.Parameter(nonneg=True, name="tr_z",  value=resolve_tr_param(self, 'z', 'z'))
    self.cp_params.tr_nu = cp.Parameter(nonneg=True, name="tr_nu", value=resolve_tr_param(self, 'nu', 'nu'))

    self.lm_adapt = bool(int(getattr(self.hyperparams.trust_region, 'lm_adapt', 1)))
    self.lm_mu    = float(getattr(self.hyperparams.trust_region, 'lm_mu_init', 1e-8)) if self.lm_adapt else 0.0

    compile_merit(self)


def update_tr_params_scalar(self) -> None:
    self.cp_params.tr_z.value  = resolve_tr_param(self, 'z', 'z')
    self.cp_params.tr_nu.value = resolve_tr_param(self, 'nu', 'nu')
    update_lagrangian_hessian(self)


def create_cost_trust_region_component(self) -> None:
    # separate scalar weight per z/nu component group, from trust_region.tr_step
    if self.hyperparams.discretize.mode not in ("ms", "ps"):
        return
    idx_z  = self.index_map.indices.z
    idx_nu = self.index_map.indices.nu

    self.cp_cost += self.cp_params.tr_x * cp.sum_squares(self.dz[:, idx_z.state])
    self.cp_cost += self.cp_params.tr_t * cp.sum_squares(self.dz[:, idx_z.time])
    if self.index_map.n.ctcs > 0:
        self.cp_cost += self.cp_params.tr_ctcs * cp.sum_squares(self.dz[:, idx_z.ctcs])
    if self.index_map.n.running_cost > 0:
        self.cp_cost += self.cp_params.tr_gamma * cp.sum_squares(self.dz[:, idx_z.running_cost])

    self.cp_cost += self.cp_params.tr_u * cp.sum_squares(self.dnu[:, idx_nu.control])
    self.cp_cost += self.cp_params.tr_s * cp.sum_squares(self.dnu[:, idx_nu.dilation_factor])


def create_cost_trust_region_hessian(self) -> None:
    # Hessian-weighted quadratic penalty; cp_params.L is set each iteration by update_lagrangian_hessian
    if self.hyperparams.discretize.mode not in ("ms", "ps"):
        return
    for k in range(self.index_map.N.all):
        w_k = cp.hstack([self.dz[k], self.dnu[k]])
        self.cp_cost += 0.5 * cp.sum_squares(self.cp_params.L[k] @ w_k)


def create_cost_trust_region_scalar(self) -> None:
    """Penalize the step with a single scalar weight each on z and nu (tr_z/tr_nu)."""
    if self.hyperparams.discretize.mode not in ("ms", "ps"):
        return
    for k in range(self.index_map.N.all):
        self.cp_cost += 0.5 * self.cp_params.tr_z * cp.sum_squares(self.dz[k])
        self.cp_cost += 0.5 * self.cp_params.tr_nu * cp.sum_squares(self.dnu[k])


# SCP_METHOD

def tighten_scp_trust_region(subproblems) -> None:
    """Increase first-order retry weights, without changing configured step sizes."""
    for subproblem in subproblems:
        subproblem.tr_scale = min(subproblem.tr_scale * 10.0, 1.0e6)


def relax_scp_trust_region(subproblems) -> None:
    """Return first-order retry weights towards their configured values."""
    for subproblem in subproblems:
        if subproblem.tr_scale > 1.0:
            subproblem.tr_scale = max(subproblem.tr_scale * 0.7, 1.0)


def line_search(subproblems, alpha_min=1e-7, c1=1e-4, beta=0.5, max_iter=20):
    subproblems = list(subproblems)

    phi_0, dphi = 0.0, 0.0
    for seg in subproblems:
        v, g = merit_grad_at_zero(seg)
        phi_0 += v
        dphi += g

    slope = min(dphi, -abs(dphi) * 1e-10)

    alpha = 1.0
    for _ in range(max_iter):
        phi = sum(evaluate_merit_at_alpha(seg, alpha) for seg in subproblems)
        if np.isfinite(phi) and phi <= phi_0 + c1 * alpha * slope:
            return alpha
        alpha *= beta
        if alpha < alpha_min:
            return alpha_min

    return alpha_min


def step_is_usable(subproblems, max_step: float = 1e6) -> bool:
    """Whether the step is real, since a solver at its iteration limit can return nonsense."""
    for scp_phase in subproblems:
        for expr in (scp_phase.dz, scp_phase.dnu):
            value = expr.value
            if value is None:
                return False
            value = np.asarray(value)
            if not np.isfinite(value).all() or np.max(np.abs(value)) > max_step:
                return False
    return True


def update_scp_trust_region(subproblems) -> bool:
    # tighten and ask for a retry if the step is unusable, else relax back towards target
    subproblems = list(subproblems)
    if not step_is_usable(subproblems):
        tighten_scp_trust_region(subproblems)
        return False
    relax_scp_trust_region(subproblems)
    return True


# SCP_PHASE / SUBPROBLEM

def update_lagrangian_hessian(self) -> None:
    if self.hyperparams.discretize.mode not in ("ms", "ps"):
        return
    if not getattr(self.hyperparams.trust_region, 'second_order', True):
        return

    N    = self.index_map.N.all
    n_z  = self.index_map.n.z
    n_nu = self.index_map.n.nu

    iteration = len(self.iter_data_list)

    if iteration == 0:
        H = np.tile(np.eye(n_z + n_nu), (N, 1, 1))
    else:
        H = np.zeros((N, n_z + n_nu, n_z + n_nu))
        if self.lm_adapt:
            adapt_levenberg(self, iteration)
        H += self.lm_mu * np.eye(n_z + n_nu)[np.newaxis, :, :]

    for constraint in self.constraints.values():
        accumulate_hessian_constraint(constraint, self, H)

    for cost in self.costs.values():
        accumulate_hessian_cost(cost, self, H)

    self.cp_params.L.value = psd_sqrt(H, float(getattr(self.hyperparams.trust_region, 'min_eig_psd', 0.0)))


def adapt_levenberg(self, iteration: int) -> None:
    if iteration < 2:
        return
    prev = self.iter_data_list[-1]
    prev_prev = self.iter_data_list[-2]
    merit_curr = prev.cost + prev.penalty_cost
    merit_prev = prev_prev.cost + prev_prev.penalty_cost
    ratio = merit_curr / merit_prev if merit_prev != 0 else 1.0
    if ratio > 1.0:
        self.lm_mu = min(self.lm_mu * 3.0, 1e4)
    elif ratio > 0.99:
        self.lm_mu = min(self.lm_mu * 1.5, 1e4)
    else:
        self.lm_mu = max(self.lm_mu * 0.7, 1e-6)


def psd_sqrt(H_batch, min_eig_psd=0.0):
    eigvals, eigvecs = np.linalg.eigh(H_batch)
    eigvals_reg = np.maximum(eigvals, min_eig_psd)
    sqrt_eigvals = np.sqrt(eigvals_reg)
    return sqrt_eigvals[..., :, np.newaxis] * np.transpose(eigvecs, (0, 2, 1))


def compile_merit(self) -> None:
    for cost in self.costs.values():
        compile_merit_cost(cost, self)
    for constraint in self.constraints.values():
        compile_merit_penalty(constraint, self)


def evaluate_step_linearized(self) -> tuple[float, float]:
    """Trust the convex subproblem's own reported cost (scvx style)."""
    cost = self.cp_cost.value / self.w_cost
    penalty_cost = sum(penalties.penalty_cost_value(c.penalties) for c in self.constraints.values())
    return cost, penalty_cost


def evaluate_step_merit(self) -> tuple[float, float]:
    """Re-evaluate the true nonlinear cost/penalty at the stepped point (sqp style)."""
    z   = jnp.asarray(self.current_iter_data.z_opt)
    nu  = jnp.asarray(self.current_iter_data.nu_opt)
    cost = sum(evaluate_merit_cost(c, z, nu, self.params) for c in self.costs.values())
    penalty_cost = sum(evaluate_merit(c, z, nu, self.params) for c in self.constraints.values())
    return cost, penalty_cost


def evaluate_merit_at_alpha(self, alpha):
    z  = jnp.asarray(self.current_iter_data.z_opt)  + alpha * jnp.asarray(self._dz_new)
    nu = jnp.asarray(self.current_iter_data.nu_opt) + alpha * jnp.asarray(self._dnu_new)
    phi = sum(evaluate_merit_cost(c, z, nu, self.params) for c in self.costs.values())
    for c in self.constraints.values():
        phi += evaluate_merit(c, z, nu, self.params)
    return phi


def merit_grad_at_zero(self):
    z_ref  = jnp.asarray(self.current_iter_data.z_opt)
    nu_ref = jnp.asarray(self.current_iter_data.nu_opt)
    dz     = jnp.asarray(self._dz_new)
    dnu    = jnp.asarray(self._dnu_new)
    phi, dphi = 0.0, 0.0
    for c in self.costs.values():
        v, g = merit_cost_value_and_grad_alpha(c, 0.0, z_ref, dz, nu_ref, dnu, self.params)
        phi += v
        dphi += g
    for c in self.constraints.values():
        v, g = merit_value_and_grad_alpha(c, 0.0, z_ref, dz, nu_ref, dnu, self.params)
        phi += v
        dphi += g
    return phi, dphi


# SCP_CONSTRAINT

def compile_merit_penalty_from_violation(self, violation):
    l1 = self.penalties.norm == "l1"
    if self.penalties.vb_type == "split":
        if self.penalties.W_p.size == 0:
            return
        def merit_eval(z, nu, W_p, W_m, dual_p, dual_m, params):
            viol = violation(z, nu, params)
            viol_p = jnp.maximum(viol, 0.0)
            viol_m = jnp.maximum(-viol, 0.0)
            linear = jnp.sum(dual_p * viol_p) + jnp.sum(dual_m * viol_m)
            if l1:
                return linear + jnp.sum(W_p * viol_p) + jnp.sum(W_m * viol_m)
            return (linear
                    + 0.5 * jnp.sum(W_p * viol_p ** 2)
                    + 0.5 * jnp.sum(W_m * viol_m ** 2))
        def merit_line(alpha, z_ref, dz, nu_ref, dnu, W_p, W_m, dual_p, dual_m, params):
            return merit_eval(z_ref + alpha * dz, nu_ref + alpha * dnu, W_p, W_m, dual_p, dual_m, params)
        self._merit_eval = jax.jit(merit_eval)
        self._merit_vg   = jax.jit(jax.value_and_grad(merit_line, argnums=0))
    else:
        if self.penalties.W.size == 0:
            return
        def merit_eval(z, nu, W, dual, params):
            viol = violation(z, nu, params)
            if l1:
                return jnp.sum(dual * viol) + jnp.sum(W * jnp.abs(viol))
            return jnp.sum(dual * viol) + 0.5 * jnp.sum(W * viol ** 2)
        def merit_line(alpha, z_ref, dz, nu_ref, dnu, W, dual, params):
            return merit_eval(z_ref + alpha * dz, nu_ref + alpha * dnu, W, dual, params)
        self._merit_eval = jax.jit(merit_eval)
        self._merit_vg   = jax.jit(jax.value_and_grad(merit_line, argnums=0))


def evaluate_merit(self, z, nu, params):
    if not hasattr(self, '_merit_eval'):
        return 0.0
    if self.penalties.vb_type == "split":
        return float(self._merit_eval(
            z, nu,
            jnp.asarray(self.penalties.W_p), jnp.asarray(self.penalties.W_m),
            jnp.asarray(self.penalties.dual_p), jnp.asarray(self.penalties.dual_m),
            params))
    return float(self._merit_eval(z, nu, jnp.asarray(self.penalties.W), jnp.asarray(self.penalties.dual), params))


def merit_value_and_grad_alpha(self, alpha, z_ref, dz, nu_ref, dnu, params):
    if not hasattr(self, '_merit_vg'):
        return 0.0, 0.0
    if self.penalties.vb_type == "split":
        v, g = self._merit_vg(
            alpha, z_ref, dz, nu_ref, dnu,
            jnp.asarray(self.penalties.W_p), jnp.asarray(self.penalties.W_m),
            jnp.asarray(self.penalties.dual_p), jnp.asarray(self.penalties.dual_m),
            params)
    else:
        v, g = self._merit_vg(alpha, z_ref, dz, nu_ref, dnu, jnp.asarray(self.penalties.W), jnp.asarray(self.penalties.dual), params)
    return float(v), float(g)


# SCP_CONSTRAINT_TYPES

def accumulate_hessian_dynamics(self, scp_phase, H):
    n_z = scp_phase.index_map.n.z

    if scp_phase.hyperparams.discretize.mode == "ps":
        H[1:, :n_z, :n_z] += np.asarray(self.ps_H_z[0])
        H[1:, :n_z, n_z:] += np.asarray(self.ps_H_z[1])
        H[1:, n_z:, :n_z] += np.asarray(self.ps_H_nu[0])
        H[1:, n_z:, n_z:] += np.asarray(self.ps_H_nu[1])
        return

    H[:-1, :n_z, :n_z] += np.asarray(self.H_z_k[0])
    H[:-1, :n_z, n_z:] += np.asarray(self.H_z_k[1])
    H[:-1, n_z:, :n_z] += np.asarray(self.H_nu_k[0])
    H[:-1, n_z:, n_z:] += np.asarray(self.H_nu_k[1])
    H[1:,  n_z:, n_z:] += np.asarray(self.H_nu_kp[2])


def accumulate_hessian_nonconvex(self, scp_phase, H, constraints_attr):
    """Shared by scp_nonconvex_inequality (constraints_attr='cp_ineq_constraints')
    and scp_nonconvex_equality (constraints_attr='cp_eq_constraints')."""
    if not hasattr(self, constraints_attr):
        return
    n_z    = scp_phase.index_map.n.z
    z      = jnp.asarray(scp_phase.current_iter_data.z_opt)
    nu     = jnp.asarray(scp_phase.current_iter_data.nu_opt)
    params = scp_phase.params
    lam    = jnp.asarray(self.lagrangian_dual)

    H_z, H_nu = self.lagrangian_hessians(lam, z[self.nodes], nu[self.nodes], params)
    for i, k in enumerate(self.nodes):
        H[k, :n_z, :n_z] += np.asarray(H_z[0][i])
        H[k, :n_z, n_z:] += np.asarray(H_z[1][i])
        H[k, n_z:, :n_z] += np.asarray(H_nu[0][i])
        H[k, n_z:, n_z:] += np.asarray(H_nu[1][i])


def compile_merit_penalty_scp_dynamics(self, scp_phase):
    if self.penalties.W.size == 0:
        return
    if scp_phase.hyperparams.discretize.mode == "ps":
        D_jnp = jnp.asarray(self.ps_D)
        dyn_batched = self.dyn_fcn_batched
        H = self.ps_hp
        p = self.ps_p
        def violation(z, nu, params):
            f_col = dyn_batched(z[1:], nu[1:], params)
            defects = []
            for h in range(H):
                col_start = h * p
                z_h = jax.lax.dynamic_slice(z, (col_start, 0), (p + 1, z.shape[1]))
                defects.append(2.0 * D_jnp @ z_h)
            return jnp.concatenate(defects, axis=0) - f_col
        compile_merit_penalty_from_violation(self, violation)
        return
    propagate = self.propagate
    ks        = jnp.arange(scp_phase.index_map.N.all - 1)
    def violation(z, nu, params):
        return z[1:] - propagate(ks, z[:-1], nu[:-1], nu[1:], params)
    compile_merit_penalty_from_violation(self, violation)


def compile_merit_penalty_scp_final_state(self, scp_phase):
    if self.penalties.W.size == 0:
        return
    idx = jnp.asarray(self.constraint.idx)
    val = jnp.asarray(self.constraint.value)
    def violation(z, nu, params):
        return (z[-1, idx] - val).reshape(1, -1)
    compile_merit_penalty_from_violation(self, violation)


def compile_merit_penalty_scp_final_control(self, scp_phase):
    if self.penalties.W.size == 0:
        return
    idx = jnp.asarray(self.constraint.idx)
    val = jnp.asarray(self.constraint.value)
    def violation(z, nu, params):
        return (nu[-1, idx] - val).reshape(1, -1)
    compile_merit_penalty_from_violation(self, violation)


def compile_merit_penalty_scp_nonconvex_inequality(self, scp_phase):
    if self.penalties.W.size == 0:
        return
    fcn_b = self.fcn_batched
    nodes = jnp.asarray(self.nodes)
    dim   = self.constraint.dimension
    def violation(z, nu, params):
        return jnp.maximum(0.0, fcn_b(z[nodes], nu[nodes], params)).reshape(-1, dim)
    compile_merit_penalty_from_violation(self, violation)


def compile_merit_penalty_scp_nonconvex_equality(self, scp_phase):
    if self.penalties.W.size == 0:
        return
    fcn_b = self.fcn_batched
    nodes = jnp.asarray(self.nodes)
    dim   = self.constraint.dimension
    def violation(z, nu, params):
        return fcn_b(z[nodes], nu[nodes], params).reshape(-1, dim)
    compile_merit_penalty_from_violation(self, violation)


_INEQUALITY_TYPES = ('nonconvex_inequality', 'initial_nonconvex_inequality', 'final_nonconvex_inequality')
_EQUALITY_TYPES   = ('nonconvex_equality', 'initial_nonconvex_equality', 'final_nonconvex_equality')


def accumulate_hessian_constraint(constraint, scp_phase, H) -> None:
    """Used by sqp's second-order trust region: which constraint types contribute to H, and how."""
    if constraint.type == 'dynamics':
        accumulate_hessian_dynamics(constraint, scp_phase, H)
    elif constraint.type in _INEQUALITY_TYPES:
        accumulate_hessian_nonconvex(constraint, scp_phase, H, 'cp_ineq_constraints')
    elif constraint.type in _EQUALITY_TYPES:
        accumulate_hessian_nonconvex(constraint, scp_phase, H, 'cp_eq_constraints')


def compile_merit_penalty(constraint, scp_phase) -> None:
    """Used by sqp's line-search merit: which constraint types contribute, and how."""
    if constraint.type == 'dynamics':
        compile_merit_penalty_scp_dynamics(constraint, scp_phase)
    elif constraint.type == 'final_state':
        compile_merit_penalty_scp_final_state(constraint, scp_phase)
    elif constraint.type == 'final_control':
        compile_merit_penalty_scp_final_control(constraint, scp_phase)
    elif constraint.type in _INEQUALITY_TYPES:
        compile_merit_penalty_scp_nonconvex_inequality(constraint, scp_phase)
    elif constraint.type in _EQUALITY_TYPES:
        compile_merit_penalty_scp_nonconvex_equality(constraint, scp_phase)


# SCP_COST

def compile_merit_cost(self, scp_phase):
    build_fn = _MERIT_COST_FNS.get(self.type)
    if build_fn is None:
        return
    fn = build_fn(self, scp_phase)
    self._has_merit = True
    self._merit_eval = jax.jit(fn)
    def merit_line(alpha, z_ref, dz, nu_ref, dnu, params):
        return fn(z_ref + alpha * dz, nu_ref + alpha * dnu, params)
    self._merit_vg = jax.jit(jax.value_and_grad(merit_line, argnums=0))


def evaluate_merit_cost(self, z, nu, params):
    if not self._has_merit:
        return 0.0
    return float(self._merit_eval(z, nu, params))


def merit_cost_value_and_grad_alpha(self, alpha, z_ref, dz, nu_ref, dnu, params):
    if not self._has_merit:
        return 0.0, 0.0
    v, g = self._merit_vg(alpha, z_ref, dz, nu_ref, dnu, params)
    return float(v), float(g)


# SCP_COST_TYPES

def first_nonconvex_terminal_cost(scp_phase):
    for scp_cost in scp_phase.costs.values():
        if scp_cost.type == "nonconvex_terminal":
            return scp_cost.cost
    return None


def accumulate_hessian_nonconvex_terminal_cost(self, scp_phase, H, first_terminal_cost_fn):
    if first_terminal_cost_fn(scp_phase) is not self.cost:
        return
    z_opt  = scp_phase.current_iter_data.z_opt
    nu_opt = scp_phase.current_iter_data.nu_opt
    n_z    = scp_phase.index_map.n.z

    H_cost_z, H_cost_nu, H_cost_znu = compute_nonconvex_terminal_cost_hessians(z_opt, nu_opt, scp_phase.phase, scp_phase)
    H[:, :n_z, :n_z] += H_cost_z
    H[:, n_z:, n_z:] += H_cost_nu
    H[:, :n_z, n_z:] += H_cost_znu
    H[:, n_z:, :n_z] += np.transpose(H_cost_znu, (0, 2, 1))


def compute_nonconvex_terminal_cost_hessians(z, nu, phase, scp_phase):
    N    = phase.index_map.N.all
    n_z  = phase.index_map.n.z
    n_nu = phase.index_map.n.nu

    H_cost_z   = np.zeros((N, n_z, n_z))
    H_cost_nu  = np.zeros((N, n_nu, n_nu))
    H_cost_znu = np.zeros((N, n_z, n_nu))

    params = phase.params
    nonconvex_costs = [c for c in phase.costs.values() if c.type == "nonconvex"]
    terminal_costs  = [c for c in phase.costs.values() if c.type == "nonconvex_terminal"]

    if len(nonconvex_costs) + len(terminal_costs) == 0:
        return H_cost_z, H_cost_nu, H_cost_znu

    z_jax  = jnp.asarray(z)
    nu_jax = jnp.asarray(nu)

    for cost_fn in nonconvex_costs:
        w = getattr(cost_fn, 'w', 1.0)
        d2z   = cost_fn.d2fcn_dz2_batched(z_jax, nu_jax, params)
        d2nu  = cost_fn.d2fcn_dnu2_batched(z_jax, nu_jax, params)
        d2znu = cost_fn.d2fcn_dzdnu_batched(z_jax, nu_jax, params)
        H_cost_z   += w * np.asarray(d2z).reshape(N, n_z, n_z)
        H_cost_nu  += w * np.asarray(d2nu).reshape(N, n_nu, n_nu)
        H_cost_znu += w * np.asarray(d2znu).reshape(N, n_z, n_nu)

    for cost_fn in terminal_costs:
        nodes    = np.atleast_1d(cost_fn.nodes)
        z_nodes  = z_jax[nodes]
        nu_nodes = nu_jax[nodes]
        d2z   = cost_fn.d2fcn_dz2_batched(z_nodes, nu_nodes, params)
        d2nu  = cost_fn.d2fcn_dnu2_batched(z_nodes, nu_nodes, params)
        d2znu = cost_fn.d2fcn_dzdnu_batched(z_nodes, nu_nodes, params)
        for i, k in enumerate(nodes):
            H_cost_z[k]   += np.asarray(d2z[i]).reshape(n_z, n_z)
            H_cost_nu[k]  += np.asarray(d2nu[i]).reshape(n_nu, n_nu)
            H_cost_znu[k] += np.asarray(d2znu[i]).reshape(n_z, n_nu)

    return H_cost_z, H_cost_nu, H_cost_znu


def merit_cost_scp_nonconvex_running(self, scp_phase):
    w = self.cost.w
    fcn_batched = jax.jit(jax.vmap(jax.jit(self.cost.fcn_znu), in_axes=(0, 0, None)))
    def eval_fn(z, nu, params):
        return w * jnp.sum(fcn_batched(z, nu, params))
    return eval_fn


def merit_cost_scp_nonconvex_terminal(self, scp_phase):
    fcn_b = self.cost.fcn_batched
    nodes = jnp.asarray(self.cost.nodes)
    def eval_fn(z, nu, params):
        return jnp.sum(fcn_b(z[nodes], nu[nodes], params))
    return eval_fn


def merit_cost_scp_nonconvex_minimax(self, scp_phase):
    w     = self.cost.w
    g_b   = self.g_batched
    nodes = jnp.asarray(self.nodes)
    def eval_fn(z, nu, params):
        return w * jnp.sum(jnp.max(g_b(z[nodes], nu[nodes], params), axis=0))
    return eval_fn


def merit_cost_scp_min_time(self, scp_phase):
    dil_idx = jnp.array(scp_phase.index_map.indices.nu.dilation_factor)
    def eval_fn(z, nu, params):
        return jnp.sum(nu[:, dil_idx[0]])
    return eval_fn


def merit_cost_scp_final_state(self, scp_phase):
    w   = self.cost.w
    idx = jnp.array(self.cost.idx)
    def eval_fn(z, nu, params):
        return w * jnp.sum(z[-1, idx])
    return eval_fn


def merit_cost_scp_final_control(self, scp_phase):
    w   = self.cost.w
    idx = jnp.array(self.cost.idx)
    def eval_fn(z, nu, params):
        return w * jnp.sum(nu[-1, idx])
    return eval_fn


def merit_cost_scp_regularization(self, scp_phase):
    w         = self.cost.w
    norm_type = self.cost.norm_type
    is_nu     = self.cost.set == "control"
    def eval_fn(z, nu, params):
        traj = nu if is_nu else z
        if norm_type == "l2":
            return w * jnp.sum(traj ** 2)
        return w * jnp.sum(jnp.abs(traj))
    return eval_fn


def merit_cost_scp_rate_regularization(self, scp_phase):
    w         = self.cost.w
    norm_type = self.cost.norm_type
    is_nu     = self.cost.set == "control"
    idx       = jnp.asarray(self.cost.idx)
    def eval_fn(z, nu, params):
        traj  = nu if is_nu else z
        delta = traj[1:, idx] - traj[:-1, idx]
        if norm_type == "l2":
            return w * jnp.sum(delta ** 2)
        return w * jnp.sum(jnp.abs(delta))
    return eval_fn


_MERIT_COST_FNS = {
    'nonconvex_running':   merit_cost_scp_nonconvex_running,
    'nonconvex_terminal':  merit_cost_scp_nonconvex_terminal,
    'nonconvex_minimax':   merit_cost_scp_nonconvex_minimax,
    'min_time':            merit_cost_scp_min_time,
    'final_state':         merit_cost_scp_final_state,
    'final_control':       merit_cost_scp_final_control,
    'regularization':      merit_cost_scp_regularization,
    'rate_regularization': merit_cost_scp_rate_regularization,
}


def accumulate_hessian_cost(cost, scp_phase, H) -> None:
    """Used by sqp's second-order trust region: only the first nonconvex-terminal cost contributes."""
    if cost.type == 'nonconvex_terminal':
        accumulate_hessian_nonconvex_terminal_cost(cost, scp_phase, H, first_nonconvex_terminal_cost)
