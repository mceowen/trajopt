import numpy as np
import jax
import jax.numpy as jnp


class SCPLinkConstraint:
    def __init__(self, constraint) -> None:
        self.constraint     = constraint
        self.type           = constraint.type
        self.cp_constraints = []
        self.cp_cost        = 0
        self.has_merit      = False
        self.converged      = True

        self.W      = None
        self.dual   = None
        self.vb     = None
        self.W_param    = None
        self.dual_param = None
        self.vb_var     = None

    def build(self, subproblems): pass

    def _compile_merit(self, residual_jax, has_W, has_dual):
        def merit_fn(z1, nu1, z2, nu2, W, dual):
            viol = residual_jax(z1, nu1, z2, nu2)
            al   = 0.0
            if has_dual:
                al = al + jnp.sum(dual[0] * viol)
            if has_W:
                al = al + 0.5 * jnp.sum(W[0] * viol ** 2)
            return al

        def merit_line(alpha, z1r, dz1, nu1r, dnu1, z2r, dz2, nu2r, dnu2, W, dual):
            return merit_fn(z1r + alpha * dz1, nu1r + alpha * dnu1, z2r + alpha * dz2, nu2r + alpha * dnu2, W, dual)

        self._merit_fn = jax.jit(merit_fn)
        self._merit_vg = jax.jit(jax.value_and_grad(merit_line, argnums=0))

    def _W_dual_jnp(self):
        W    = jnp.asarray(self.W) if self.W is not None else jnp.zeros((1, self.eps.size))
        dual = jnp.asarray(self.dual) if self.dual is not None else jnp.zeros((1, self.eps.size))
        return W, dual

    def update_cvxpy_parameters(self):
        if self.W_param is not None:
            self.W_param.value = np.sqrt(self.W)
        if self.dual_param is not None:
            self.dual_param.value = self.dual

    def apply_step(self, alpha):
        if self.vb_var is None:
            self.converged = True
            return
        self.vb        = np.array(self.vb_var.value)
        self.converged = bool(np.all(np.abs(self.vb) <= self.eps))

    def update_W_dual(self, alpha=1.0):
        if not self.has_merit:
            return

        eps = np.atleast_1d(self.eps)

        if self.dual is not None and self.penalty.dual.autotune:
            dual_new  = self.dual + (self.W if self.W is not None else 0.0) * self.vb
            self.dual = dual_new

        if self.W is not None and self.penalty.W.autotune:
            Wh     = np.abs(self.dual) / (0.9 * eps)
            self.W = np.clip(Wh, 1e-5, 1e5)

    def merit_grad_at_zero(self, scp_segments):
        if not self.has_merit:
            return 0.0, 0.0
        s1, s2 = scp_segments[self.seg1], scp_segments[self.seg2]
        W, dual = self._W_dual_jnp()
        v, g = self._merit_vg(
            0.0,
            jnp.asarray(s1.current_iter_data.z_opt),  jnp.asarray(s1._dz_new),
            jnp.asarray(s1.current_iter_data.nu_opt), jnp.asarray(s1._dnu_new),
            jnp.asarray(s2.current_iter_data.z_opt),  jnp.asarray(s2._dz_new),
            jnp.asarray(s2.current_iter_data.nu_opt), jnp.asarray(s2._dnu_new),
            W, dual,
        )
        return float(v), float(g)

    def evaluate_merit_at_alpha(self, scp_segments, alpha):
        if not self.has_merit:
            return 0.0
        s1, s2 = scp_segments[self.seg1], scp_segments[self.seg2]
        z1  = jnp.asarray(s1.current_iter_data.z_opt)  + alpha * jnp.asarray(s1._dz_new)
        nu1 = jnp.asarray(s1.current_iter_data.nu_opt) + alpha * jnp.asarray(s1._dnu_new)
        z2  = jnp.asarray(s2.current_iter_data.z_opt)  + alpha * jnp.asarray(s2._dz_new)
        nu2 = jnp.asarray(s2.current_iter_data.nu_opt) + alpha * jnp.asarray(s2._dnu_new)
        W, dual = self._W_dual_jnp()
        return float(self._merit_fn(z1, nu1, z2, nu2, W, dual))
