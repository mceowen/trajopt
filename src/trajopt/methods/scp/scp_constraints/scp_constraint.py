import numpy as np
import cvxpy as cp
import jax
import jax.numpy as jnp


class SCPConstraint():
    nonnegative_dual = False

    def __init__(self, constraint, scp_segment) -> None:
        self.constraint = constraint
        self.type       = constraint.type
        self.name       = constraint.name
        self.penalty    = None
        self.shape      = None
        self.eps        = np.atleast_1d(1e-4)
        self.vb         = np.zeros(0)
        self.vb_type    = "standard"
        self.W          = np.zeros(0)
        self.dual       = np.zeros(0)
        self.W_param    = None
        self.dual_param = None
        self.vb_var     = None
        self.vb_p_var   = None
        self.vb_m_var   = None

    def compile(self, scp_segment): pass

    def build_cross_segment(self, scp_segments): pass

    def create_cvxpy_parameters(self, scp_segment): pass
    def create_cvxpy_variables(self, scp_segment): pass
    def create_cvxpy_constraints(self, scp_segment): pass
    def update_cvxpy_parameters(self, scp_segment): pass
    def update_current_iter_data(self, scp_segment): pass

    def accumulate_hessian(self, scp_segment, H): pass

    def init_penalty(self, scp_segment): pass

    def compile_merit_penalty(self, scp_segment): pass

    def _alloc_penalty(self, scp_segment, shape):
        raw = getattr(self.constraint, 'eps', 1e-4)
        self.eps = np.broadcast_to(np.atleast_1d(raw), (shape[-1],)).copy()

        self.penalty = scp_segment.penalty_config.get(self.type, scp_segment.penalty_config.get('default'))
        self.shape   = shape
        self.vb      = np.zeros(shape)
        self.vb_type = getattr(self.penalty, 'vb', 'standard') if self.penalty else 'standard'
        if self.penalty and hasattr(self.penalty, 'W') and self.penalty.W.penalty:
            self.W    = np.full(shape, float(self.penalty.W.init))
            self.dual = np.full(shape, float(self.penalty.dual.init))

    def create_penalty_parameters(self, scp_segment):
        if self.shape is None:
            return
        if self.vb_type == "none":
            return
        self.W_param    = cp.Parameter(self.shape, nonneg=True, name=f"W_{self.name}_sqrt", value=np.zeros(self.shape))
        self.dual_param = cp.Parameter(self.shape, name=f"dual_{self.name}", value=np.zeros(self.shape))

    def create_penalty_variables(self, scp_segment):
        if self.shape is None:
            return
        if self.vb_type == "none":
            return
        if self.vb_type == "split":
            self.vb_p_var = cp.Variable(self.shape, nonneg=True, name=f"vb_p_{self.name}_{scp_segment.name}")
            self.vb_m_var = cp.Variable(self.shape, nonneg=True, name=f"vb_m_{self.name}_{scp_segment.name}")
            self.vb_var   = self.vb_p_var - self.vb_m_var
        else:
            self.vb_var = cp.Variable(self.shape, name=f"vb_{self.name}_{scp_segment.name}")

    def add_penalty_cost(self, scp_segment):
        if self.W_param is None:
            return
        if self.vb_type == "split":
            scp_segment.cp_cost += 0.5 * cp.sum_squares(cp.multiply(self.W_param, self.vb_p_var))
            scp_segment.cp_cost += 0.5 * cp.sum_squares(cp.multiply(self.W_param, self.vb_m_var))
            scp_segment.cp_cost += cp.sum(cp.multiply(self.vb_p_var - self.vb_m_var, self.dual_param))
        else:
            scp_segment.cp_cost += 0.5 * cp.sum_squares(cp.multiply(self.W_param, self.vb_var))
            scp_segment.cp_cost += cp.sum(cp.multiply(self.vb_var, self.dual_param))

    def update_penalty_parameters(self, scp_segment):
        if self.W_param is not None:
            self.W_param.value = np.sqrt(self.W)
        if self.dual_param is not None:
            self.dual_param.value = self.dual

    def read_vb(self, scp_segment):
        if self.vb_var is not None:
            if self.vb_type == "split":
                self.vb = np.array(self.vb_p_var.value) - np.array(self.vb_m_var.value)
            else:
                self.vb = np.array(self.vb_var.value)

    def update_W_dual(self, scp_segment, alpha=1.0):
        if self.penalty is None:
            return
        if not hasattr(self.penalty, 'W'):
            return

        if self.penalty.W.autotune:
            Wh = self.W * np.abs(self.vb) / (0.01 * self.eps)
            self.W = np.clip(Wh, 0.0001, 1e8)

        if self.penalty.dual.autotune:
            dual_new = self.dual + 0.1 * self.vb

            if self.nonnegative_dual:
                self.dual = np.maximum(0.0, dual_new)
            else:
                self.dual = dual_new

    def _compile_merit_penalty(self, violation):
        if self.W.size == 0:
            return

        if self.vb_type == "split":
            def merit_eval(z, nu, W, dual, params):
                viol = violation(z, nu, params)
                viol_p = jnp.maximum(viol, 0.0)
                viol_m = jnp.maximum(-viol, 0.0)
                return (jnp.sum(dual * (viol_p - viol_m))
                        + 0.5 * jnp.sum(W * viol_p ** 2)
                        + 0.5 * jnp.sum(W * viol_m ** 2))
        else:
            def merit_eval(z, nu, W, dual, params):
                viol = violation(z, nu, params)
                return jnp.sum(dual * viol) + 0.5 * jnp.sum(W * viol ** 2)

        def merit_line(alpha, z_ref, dz, nu_ref, dnu, W, dual, params):
            return merit_eval(z_ref + alpha * dz, nu_ref + alpha * dnu, W, dual, params)

        self._merit_eval = jax.jit(merit_eval)
        self._merit_vg   = jax.jit(jax.value_and_grad(merit_line, argnums=0))

    def evaluate_merit(self, z, nu, params):
        if not hasattr(self, '_merit_eval'):
            return 0.0
        return float(self._merit_eval(z, nu, jnp.asarray(self.W), jnp.asarray(self.dual), params))

    def merit_value_and_grad_alpha(self, alpha, z_ref, dz, nu_ref, dnu, params):
        if not hasattr(self, '_merit_vg'):
            return 0.0, 0.0
        v, g = self._merit_vg(alpha, z_ref, dz, nu_ref, dnu, jnp.asarray(self.W), jnp.asarray(self.dual), params)
        
        return float(v), float(g)

    @property
    def vb_ratio(self):
        if self.vb.size == 0:
            return 0.0
        return float(np.max(np.abs(self.vb) / self.eps))

    @property
    def is_feasible(self):
        if self.vb.size == 0:
            return True
        return bool(np.all(np.abs(self.vb) <= self.eps))