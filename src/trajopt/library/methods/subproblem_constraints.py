import copy
import numpy as np
from trajopt.core.constraints.constraints import Constraints
from trajopt.utils import tools

from trajopt.library.methods import hyperparameters as hp


class SubproblemConstraints(Constraints):
    """
    Constraints + SCP-specific per-constraint data (vb/W/dual/etc.).
    """

    def __init__(self, problem=None, method=None, cnstr_config_list=None, config=None):
        # Mode 1: replicate from Problem.constraints
        if problem is not None and method is not None:
            self.constraints_list = [copy.copy(c) for c in problem.constraints.constraints_list]
            
            # TODO(Skye): Fix this fundamental logic
            # Set dynamics constraint dimension from indexmap
            dyn_constraint = next((c for c in self.constraints_list if getattr(c, "type", None) == "dynamics"), None)
            if dyn_constraint is not None:
                dyn_constraint.dimension = int(method.index_map.n.dynamics)

        # Mode 2: rebuild like Constraints
        elif cnstr_config_list is not None and config is not None and method is not None:
            super().__init__(cnstr_config_list, config)

        else:
            raise ValueError("Provide either (problem, method) OR (cnstr_config_list, config, method).")

        # Allocate time-indexed SCP arrays using IndexMap.time_grid
        time_grid = method.index_map.N
        N_default = int(method.index_map.N['N'])

        for c in self.constraints_list:
            d = int(getattr(c, "dimension"))
            Nk = int(time_grid.get(getattr(c, "type", None), N_default))

            if Nk == 1:
                c.W    = np.zeros(d)
                c.dual = np.zeros(d)
                c.vb   = np.zeros(d)
            else:
                c.W    = np.zeros((Nk, d))
                c.dual = np.zeros((Nk, d))
                c.vb   = np.zeros((Nk, d))

        # Aggregate plus/minus buffers (not tied to a single constraint)
        idx = method.index_map


    def stack_W_and_dual(self, problem, method):
        """
        Build organized W and dual attribute dictionaries from per-constraint arrays.
        Returns (W_stack, dual_stack) where each is an AttrDict containing aggregate arrays.
        """
        idx = method.index_map

        W_stack = tools.AttrDict()
        dual_stack = tools.AttrDict()

        ineq_constraints = [
            c for c in self.constraints_list
            if getattr(c, "type", None) == "nonconvex_inequality"
        ]
        term_constraints = [
            c for c in self.constraints_list
            if getattr(c, "type", None) in {"equality_bc", "inequality_bc"}
            and getattr(c, "boundary", None) == "final"
            and getattr(c, "set", None) == "state"
        ]
        ctcs_constraints = [c for c in self.constraints_list if getattr(c, "ct", None) == 1]
        dyn_constraints = [c for c in self.constraints_list if getattr(c, "type", None) == "dynamics"]

        W_stack.nonconvex_inequality  = self._stack_constraint_attr(ineq_constraints, "W", (idx.N.N, idx.n.nonconvex_inequality))
        dual_stack.nonconvex_inequality = self._stack_constraint_attr(ineq_constraints, "dual", (idx.N.N, idx.n.nonconvex_inequality))

        W_stack.terminal              = self._stack_constraint_attr(term_constraints + ctcs_constraints, "W", (max(idx.n.term_total, 1),))
        dual_stack.terminal           = self._stack_constraint_attr(term_constraints + ctcs_constraints, "dual", (max(idx.n.term_total, 1),))

        W_stack.dynamics              = self._stack_constraint_attr(dyn_constraints, "W", (max(idx.N.N - 1, 1), max(idx.n.z, 1)))
        dual_stack.dynamics           = self._stack_constraint_attr(dyn_constraints, "dual", (max(idx.N.N - 1, 1), max(idx.n.z, 1)))

        # Initialize plus/minus buffers as zeros (will be populated by configure_penalty_weights)
        W_stack.plus_real     = np.zeros((idx.N.pm_real, idx.n.plus_real))
        W_stack.minus_real    = np.zeros((idx.N.pm_real, idx.n.minus_real))
        W_stack.plus_ctcs     = np.zeros((idx.N.pm_ctcs, idx.n.plus_ctcs))
        W_stack.minus_ctcs    = np.zeros((idx.N.pm_ctcs, idx.n.minus_ctcs))

        dual_stack.plus_real  = np.zeros((idx.N.pm_real, idx.n.plus_real))
        dual_stack.minus_real = np.zeros((idx.N.pm_real, idx.n.minus_real))
        dual_stack.plus_ctcs  = np.zeros((idx.N.pm_ctcs, idx.n.plus_ctcs))
        dual_stack.minus_ctcs = np.zeros((idx.N.pm_ctcs, idx.n.minus_ctcs))

        return W_stack, dual_stack

    def apply_stacked_W_and_dual(self, W_stack, dual_stack, method):
        """
        Initialize each constraint's W and dual from stacked arrays.
        """
        idx = method.index_map
        N = idx.N.N

        W_ineq      = tools.ensure_shape(W_stack.get("nonconvex_inequality", 0.0), (N, max(idx.n.nonconvex_inequality, 1))) if idx.n.nonconvex_inequality > 0 else np.zeros((N, 0))
        dual_ineq   = tools.ensure_shape(dual_stack.get("nonconvex_inequality", 0.0), (N, max(idx.n.nonconvex_inequality, 1))) if idx.n.nonconvex_inequality > 0 else np.zeros((N, 0))

        W_term      = tools.ensure_shape(W_stack.get("terminal", 0.0), (max(idx.n.term_total, 1),)) if idx.n.term_total > 0 else np.zeros((0,))
        dual_term   = tools.ensure_shape(dual_stack.get("terminal", 0.0), (max(idx.n.term_total, 1),)) if idx.n.term_total > 0 else np.zeros((0,))

        W_dyn       = tools.ensure_shape(W_stack.get("dynamics", 0.0), (max(N - 1, 1), max(idx.n.z, 1))) if idx.n.z > 0 else np.zeros((max(N - 1, 0), 0))
        dual_dyn    = tools.ensure_shape(dual_stack.get("dynamics", 0.0), (max(N - 1, 1), max(idx.n.z, 1))) if idx.n.z > 0 else np.zeros((max(N - 1, 0), 0))

        offsets = {
            "path": 0,
            "nfz": idx.n.path,
            "custom": idx.n.path + idx.n.nfz,
        }

        term_offset = 0

        for c in self.constraints_list:
            c_type = getattr(c, "type", None)
            c_group = getattr(c, "group", None)
            dim = int(getattr(c, "dimension", 0))

            if c_type == "nonconvex_inequality":
                start = offsets.get(c_group, offsets["custom"])
                end = start + dim
                c.W = W_ineq[:, start:end] if dim > 0 else np.zeros((N, 0))
                c.dual = dual_ineq[:, start:end] if dim > 0 else np.zeros((N, 0))
                offsets[c_group] = end

            elif c_type in {"equality_bc", "inequality_bc"} and getattr(c, "boundary", None) == "final" and getattr(c, "set", None) == "state":
                sl = slice(term_offset, term_offset + dim)
                c.W = W_term[sl] if dim > 0 else np.zeros((0,))
                c.dual = dual_term[sl] if dim > 0 else np.zeros((0,))
                term_offset += dim

            elif c_type == "dynamics":
                c.W = W_dyn[:, :dim] if dim > 0 else np.zeros((max(N - 1, 0), 0))
                c.dual = dual_dyn[:, :dim] if dim > 0 else np.zeros((max(N - 1, 0), 0))

        # Note: plus_real/minus_real/plus_ctcs/minus_ctcs are not per-constraint,
        # so they are not distributed here. They remain in the W_stack/dual_stack objects.

    def stack_W_and_dual_by(self, key):
        """
        Stack W and dual arrays by a constraint attribute (type, group, or name).
        Returns (W_stack, dual_stack) AttrDicts keyed by attribute value.
        """
        W_stack = tools.AttrDict()
        dual_stack = tools.AttrDict()

        groups = {}
        for c in self.constraints_list:
            val = getattr(c, key, None)
            if val is None:
                continue
            groups.setdefault(val, []).append(c)

        for val, lst in groups.items():
            W_stack[val] = self._stack_constraint_attr(lst, "W")
            dual_stack[val] = self._stack_constraint_attr(lst, "dual")

        return W_stack, dual_stack

    def _stack_constraint_attr(self, constraints, attr, fallback_shape=None):
        if not constraints:
            if fallback_shape is None:
                return np.zeros((1, 1))
            # If fallback_shape is 1D tuple, return 1D array
            if len(fallback_shape) == 1:
                return np.zeros(fallback_shape[0])
            return np.zeros(fallback_shape)

        raw_arrays = [np.asarray(getattr(c, attr)) for c in constraints]
        arrays = [a.reshape(1, -1) if a.ndim == 1 else a for a in raw_arrays]

        Nk = arrays[0].shape[0]
        total_dim = sum(a.shape[1] for a in arrays)
        stacked = np.zeros((Nk, total_dim))

        j = 0
        for a in arrays:
            if a.shape[0] != Nk:
                raise ValueError("Inconsistent time index rows when stacking constraints.")
            d = a.shape[1]
            stacked[:, j:j + d] = a
            j += d

        if all(a.ndim == 1 for a in raw_arrays) and Nk == 1:
            return stacked.reshape(-1)

        return stacked

    def stack_W(self, constraint_type: str) -> np.ndarray:
        """
        Stack W for all constraints of a given type.
        Returns array of shape (Nk, total_dimension_for_type).
        """

        lst = [c for c in self.constraints_list
               if getattr(c, "type", None) == constraint_type]

        if not lst:
            return np.zeros((1, 1))

        # infer time dimension from first constraint
        W0 = np.asarray(lst[0].W)
        if W0.ndim == 1:
            W0 = W0.reshape(1, -1)
        Nk = W0.shape[0]

        total_dim = sum(int(c.dimension) for c in lst)
        Wstk = np.zeros((Nk, total_dim))

        j = 0
        for c in lst:
            Wc = np.asarray(c.W)
            if Wc.ndim == 1:
                Wc = Wc.reshape(1, -1)
            d = int(c.dimension)

            if Wc.shape[0] != Nk:
                raise ValueError(
                    f"Inconsistent time index rows for type '{constraint_type}'."
                )

            Wstk[:, j:j+d] = Wc
            j += d

        return Wstk



def configure_penalty_weights(problem, method, subconstraints=None):
    """
    Wrapper to configure penalty weights using existing hyperparameters logic.
    This ensures subproblem constraint weights are populated consistently.
    """
    return hp.configure_penalty_weights(problem, method, subconstraints=subconstraints)