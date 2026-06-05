import jax
import jax.numpy as jnp

class SCPCost:
    def __init__(self, cost, scp_segment) -> None:
        self.cost = cost
        self.type = cost.type
        self.name = cost.name
        self._has_merit = False

    def create_cvxpy_cost(self, scp_segment): pass
    def update_cvxpy_parameters(self, scp_segment): pass
    def merit_cost(self, scp_segment): return None
    def accumulate_hessian(self, scp_segment, H): pass

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
