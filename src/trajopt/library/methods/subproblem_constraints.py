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
                dyn_constraint.dimension = int(method.index_map.n.dyn)

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

            c.W    = np.zeros((Nk, d))
            c.dual = np.zeros((Nk, d))   # leave for later
            c.vb   = np.zeros((Nk, d))   # leave for later


    def stack_W_and_dual(self, problem, method):
        """
        Build organized W and dual attribute dictionaries from method.weights.
        Returns (W, dual) where each is an AttrDict containing constraint type → array mappings.
        
        Dimensions come from method.index_map.n and method.index_map.N.
        """
        idx     = method.index_map
        
        # Initialize AttrDicts
        W      = tools.AttrDict()
        dual   = tools.AttrDict()
        
        # Initialize from method.weights, using index_map dimensions
        W.ineq          = method.weights.get("W_ineq", np.zeros((idx.N.N, idx.n.ineq)))
        W.term          = method.weights.get("W_term", np.zeros(idx.n.term_total))
        W.dyn           = method.weights.get("W_dyn", np.zeros((idx.N.N - 1, idx.n.z)))
        W.plus_real     = method.weights.get("W_plus_real", np.zeros((idx.N.pm_real, idx.n.plus_real)))
        W.minus_real    = method.weights.get("W_minus_real", np.zeros((idx.N.pm_real, idx.n.minus_real)))
        W.plus_ctcs     = method.weights.get("W_plus_ctcs", np.zeros((idx.N.pm_ctcs, idx.n.plus_ctcs)))
        W.minus_ctcs    = method.weights.get("W_minus_ctcs", np.zeros((idx.N.pm_ctcs, idx.n.minus_ctcs)))
        
        # Dual dictionary
        dual.ineq       = method.weights.get("dual_ineq", np.zeros((idx.N.N, idx.n.ineq)))
        dual.term       = method.weights.get("dual_term", np.zeros(idx.n.term_total))
        dual.dyn        = method.weights.get("dual_dyn", np.zeros((idx.N.N - 1, idx.n.z)))
        dual.plus_real  = method.weights.get("dual_plus_real", np.zeros((idx.N.pm_real, idx.n.plus_real)))
        dual.minus_real = method.weights.get("dual_minus_real", np.zeros((idx.N.pm_real, idx.n.minus_real)))
        dual.plus_ctcs  = method.weights.get("dual_plus_ctcs", np.zeros((idx.N.pm_ctcs, idx.n.plus_ctcs)))
        dual.minus_ctcs = method.weights.get("dual_minus_ctcs", np.zeros((idx.N.pm_ctcs, idx.n.minus_ctcs)))
        
        return W, dual

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



def configure_penalty_weights(problem, method):
    """
    Wrapper to configure penalty weights using existing hyperparameters logic.
    This ensures `method.weights` is populated consistently.
    """
    hp.configure_penalty_weights(problem, method)