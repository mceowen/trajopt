import cvxpy as cp
import numpy as np

from trajopt.methods.scp.scp_link_constraints.scp_link_constraint import SCPLinkConstraint


class scp_continuity(SCPLinkConstraint):
    def build(self, scp_segments):
        c     = self.constraint
        sub1  = scp_segments[c.segment1_name]
        sub2  = scp_segments[c.segment2_name]
        self.seg1 = c.segment1_name
        self.seg2 = c.segment2_name

        residual = c.residual(sub1, sub2)
        dim      = residual.shape[0]

        self.eps = np.full(dim, 1e-4)

        self.penalty   = sub1.penalty_config.get(c.type, sub1.penalty_config.get('default', c.penalty))
        penalize_W     = bool(self.penalty) and bool(self.penalty.W.penalty)
        penalize_dual  = bool(self.penalty) and bool(self.penalty.dual.penalty)
        self.has_merit = penalize_W or penalize_dual

        if not self.has_merit:
            self.cp_constraints.append(residual == 0)
            return

        self.vb_var = cp.Variable((1, dim), name=f"vb_link_{c.name}")
        self.vb     = np.zeros((1, dim))
        self.cp_constraints.append(residual - self.vb_var[0, :] == 0)

        if penalize_W:
            self.W       = np.full((1, dim), float(self.penalty.W.init))
            self.W_param = cp.Parameter((1, dim), nonneg=True, name=f"W_link_{c.name}_sqrt", value=np.zeros((1, dim)))
            self.cp_cost += 0.5 * cp.sum_squares(cp.multiply(self.W_param[0, :], self.vb_var[0, :]))

        if penalize_dual:
            self.dual       = np.full((1, dim), float(self.penalty.dual.init))
            self.dual_param = cp.Parameter((1, dim), name=f"dual_link_{c.name}", value=np.zeros((1, dim)))
            self.cp_cost += cp.sum(cp.multiply(self.vb_var, self.dual_param))

        self._compile_merit(c.residual_jax(sub1.index_map), penalize_W, penalize_dual)
