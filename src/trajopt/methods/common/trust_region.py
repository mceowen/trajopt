import numpy as np
import cvxpy as cp
import jax
import jax.numpy as jnp


# SCP_METHOD

def line_search(self, c1=1e-4, beta=0.5, max_iter=20, alpha_min=None):
    segments = self.scp_trajectory.scp_segments

    if alpha_min is None:
        alpha_min = float(getattr(self.method_config.flags, 'alpha_min_ls', 1e-7))

    phi_0, dphi = 0.0, 0.0
    for seg in segments.values():
        v, g = merit_grad_at_zero(seg)
        phi_0 += v
        dphi += g

    slope = min(dphi, -abs(dphi) * 1e-10)

    alpha = 1.0
    for _ in range(max_iter):
        phi = sum(evaluate_merit_at_alpha(seg, alpha) for seg in segments.values())
        if np.isfinite(phi) and phi <= phi_0 + c1 * alpha * slope:
            return alpha
        alpha *= beta
        if alpha < alpha_min:
            return alpha_min

    return alpha_min


def step_is_usable(self, max_step: float = 1e6) -> bool:
    """Whether the step is real, since a solver at its iteration limit can return nonsense."""
    for scp_segment in self.scp_trajectory.scp_segments.values():
        for expr in (scp_segment.dz, scp_segment.dnu):
            value = expr.value
            if value is None:
                return False
            value = np.asarray(value)
            if not np.isfinite(value).all() or np.max(np.abs(value)) > max_step:
                return False
    return True


# SCP_SEGMENT

def create_cost_trust_region(self) -> None:
    if self.flags.discretize not in ("ms", "ps"):
        return
    if getattr(self.flags, 'second_order', True):
        for k in range(self.index_map.N.all):
            w_k = cp.hstack([self.dz[k], self.dnu[k]])
            self.cp_cost += 0.5 * cp.sum_squares(self.cp_params.L[k] @ w_k)
    else:
        for k in range(self.index_map.N.all):
            self.cp_cost += 0.5 * self.cp_params.tr_z * cp.sum_squares(self.dz[k])
            self.cp_cost += 0.5 * self.cp_params.tr_nu * cp.sum_squares(self.dnu[k])


def update_lagrangian_hessian(self) -> None:
    if self.flags.discretize not in ("ms", "ps"):
        return
    if not getattr(self.flags, 'second_order', True):
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
        constraint.accumulate_hessian(self, H)

    for cost in self.costs.values():
        cost.accumulate_hessian(self, H)

    self.cp_params.L.value = psd_sqrt(H, float(getattr(self.flags, 'min_eig_psd', 0.0)))


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
        constraint.compile_merit_penalty(self)


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

def compile_merit_penalty(self, scp_segment): pass


def compile_merit_penalty_from_violation(self, violation):
    l1 = self.penalty_state.norm == "l1"
    if self.penalty_state.vb_type == "split":
        if self.penalty_state.W_p.size == 0:
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
        if self.penalty_state.W.size == 0:
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
    if self.penalty_state.vb_type == "split":
        return float(self._merit_eval(
            z, nu,
            jnp.asarray(self.penalty_state.W_p), jnp.asarray(self.penalty_state.W_m),
            jnp.asarray(self.penalty_state.dual_p), jnp.asarray(self.penalty_state.dual_m),
            params))
    return float(self._merit_eval(z, nu, jnp.asarray(self.penalty_state.W), jnp.asarray(self.penalty_state.dual), params))


def merit_value_and_grad_alpha(self, alpha, z_ref, dz, nu_ref, dnu, params):
    if not hasattr(self, '_merit_vg'):
        return 0.0, 0.0
    if self.penalty_state.vb_type == "split":
        v, g = self._merit_vg(
            alpha, z_ref, dz, nu_ref, dnu,
            jnp.asarray(self.penalty_state.W_p), jnp.asarray(self.penalty_state.W_m),
            jnp.asarray(self.penalty_state.dual_p), jnp.asarray(self.penalty_state.dual_m),
            params)
    else:
        v, g = self._merit_vg(alpha, z_ref, dz, nu_ref, dnu, jnp.asarray(self.penalty_state.W), jnp.asarray(self.penalty_state.dual), params)
    return float(v), float(g)


# SCP_CONSTRAINT_TYPES

def accumulate_hessian_dynamics(self, scp_segment, H):
    n_z = scp_segment.index_map.n.z

    if scp_segment.flags.discretize == "ps":
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


def accumulate_hessian_nonconvex(self, scp_segment, H, constraints_attr):
    """Shared by scp_nonconvex_inequality (constraints_attr='cp_ineq_constraints')
    and scp_nonconvex_equality (constraints_attr='cp_eq_constraints')."""
    if not hasattr(self, constraints_attr):
        return
    n_z    = scp_segment.index_map.n.z
    z      = jnp.asarray(scp_segment.current_iter_data.z_opt)
    nu     = jnp.asarray(scp_segment.current_iter_data.nu_opt)
    params = scp_segment.params
    lam    = jnp.asarray(self.lagrangian_dual)

    H_z, H_nu = self.lagrangian_hessians(lam, z[self.nodes], nu[self.nodes], params)
    for i, k in enumerate(self.nodes):
        H[k, :n_z, :n_z] += np.asarray(H_z[0][i])
        H[k, :n_z, n_z:] += np.asarray(H_z[1][i])
        H[k, n_z:, :n_z] += np.asarray(H_nu[0][i])
        H[k, n_z:, n_z:] += np.asarray(H_nu[1][i])


def compile_merit_penalty_scp_dynamics(self, scp_segment):
    if self.penalty_state.W.size == 0:
        return
    if scp_segment.flags.discretize == "ps":
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
    ks        = jnp.arange(scp_segment.index_map.N.all - 1)
    def violation(z, nu, params):
        return z[1:] - propagate(ks, z[:-1], nu[:-1], nu[1:], params)
    compile_merit_penalty_from_violation(self, violation)


def compile_merit_penalty_scp_final_state(self, scp_segment):
    if self.penalty_state.W.size == 0:
        return
    idx = jnp.asarray(self.constraint.idx)
    val = jnp.asarray(self.constraint.value)
    def violation(z, nu, params):
        return (z[-1, idx] - val).reshape(1, -1)
    compile_merit_penalty_from_violation(self, violation)


def compile_merit_penalty_scp_final_control(self, scp_segment):
    if self.penalty_state.W.size == 0:
        return
    idx = jnp.asarray(self.constraint.idx)
    val = jnp.asarray(self.constraint.value)
    def violation(z, nu, params):
        return (nu[-1, idx] - val).reshape(1, -1)
    compile_merit_penalty_from_violation(self, violation)


def compile_merit_penalty_scp_nonconvex_inequality(self, scp_segment):
    if self.penalty_state.W.size == 0:
        return
    fcn_b = self.fcn_batched
    nodes = jnp.asarray(self.nodes)
    dim   = self.constraint.dimension
    def violation(z, nu, params):
        return jnp.maximum(0.0, fcn_b(z[nodes], nu[nodes], params)).reshape(-1, dim)
    compile_merit_penalty_from_violation(self, violation)


def compile_merit_penalty_scp_nonconvex_equality(self, scp_segment):
    if self.penalty_state.W.size == 0:
        return
    fcn_b = self.fcn_batched
    nodes = jnp.asarray(self.nodes)
    dim   = self.constraint.dimension
    def violation(z, nu, params):
        return fcn_b(z[nodes], nu[nodes], params).reshape(-1, dim)
    compile_merit_penalty_from_violation(self, violation)


# SCP_COST

def merit_cost(self, scp_segment): return None


def compile_merit_cost(self, scp_segment):
    fn = self.merit_cost(scp_segment)
    if fn is None:
        return
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

def accumulate_hessian_nonconvex_terminal_cost(self, scp_segment, H, first_terminal_cost_fn):
    if first_terminal_cost_fn(scp_segment) is not self.cost:
        return
    z_opt  = scp_segment.current_iter_data.z_opt
    nu_opt = scp_segment.current_iter_data.nu_opt
    n_z    = scp_segment.index_map.n.z

    H_cost_z, H_cost_nu, H_cost_znu = compute_nonconvex_terminal_cost_hessians(z_opt, nu_opt, scp_segment.segment, scp_segment)
    H[:, :n_z, :n_z] += H_cost_z
    H[:, n_z:, n_z:] += H_cost_nu
    H[:, :n_z, n_z:] += H_cost_znu
    H[:, n_z:, :n_z] += np.transpose(H_cost_znu, (0, 2, 1))


def compute_nonconvex_terminal_cost_hessians(z, nu, segment, scp_segment):
    N    = segment.index_map.N.all
    n_z  = segment.index_map.n.z
    n_nu = segment.index_map.n.nu

    H_cost_z   = np.zeros((N, n_z, n_z))
    H_cost_nu  = np.zeros((N, n_nu, n_nu))
    H_cost_znu = np.zeros((N, n_z, n_nu))

    params = segment.params
    nonconvex_costs = [c for c in segment.costs.values() if c.type == "nonconvex"]
    terminal_costs  = [c for c in segment.costs.values() if c.type == "nonconvex_terminal"]

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


def merit_cost_scp_nonconvex_running(self, scp_segment):
    w = self.cost.w
    fcn_batched = jax.jit(jax.vmap(jax.jit(self.cost.fcn_znu), in_axes=(0, 0, None)))
    def eval_fn(z, nu, params):
        return w * jnp.sum(fcn_batched(z, nu, params))
    return eval_fn


def merit_cost_scp_nonconvex_terminal(self, scp_segment):
    fcn_b = self.cost.fcn_batched
    nodes = jnp.asarray(self.cost.nodes)
    def eval_fn(z, nu, params):
        return jnp.sum(fcn_b(z[nodes], nu[nodes], params))
    return eval_fn


def merit_cost_scp_nonconvex_minimax(self, scp_segment):
    w     = self.cost.w
    g_b   = self.g_batched
    nodes = jnp.asarray(self.nodes)
    def eval_fn(z, nu, params):
        return w * jnp.sum(jnp.max(g_b(z[nodes], nu[nodes], params), axis=0))
    return eval_fn


def merit_cost_scp_min_time(self, scp_segment):
    dil_idx = jnp.array(scp_segment.index_map.indices.nu.dilation_factor)
    def eval_fn(z, nu, params):
        return jnp.sum(nu[:, dil_idx[0]])
    return eval_fn


def merit_cost_scp_final_state(self, scp_segment):
    w   = self.cost.w
    idx = jnp.array(self.cost.idx)
    def eval_fn(z, nu, params):
        return w * jnp.sum(z[-1, idx])
    return eval_fn


def merit_cost_scp_final_control(self, scp_segment):
    w   = self.cost.w
    idx = jnp.array(self.cost.idx)
    def eval_fn(z, nu, params):
        return w * jnp.sum(nu[-1, idx])
    return eval_fn


def merit_cost_scp_regularization(self, scp_segment):
    w         = self.cost.w
    norm_type = self.cost.norm_type
    is_nu     = self.cost.set == "control"
    def eval_fn(z, nu, params):
        traj = nu if is_nu else z
        if norm_type == "l2":
            return w * jnp.sum(traj ** 2)
        return w * jnp.sum(jnp.abs(traj))
    return eval_fn


def merit_cost_scp_rate_regularization(self, scp_segment):
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
