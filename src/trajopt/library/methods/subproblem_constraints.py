from copy import deepcopy
import numpy as np

from trajopt.core.constraints.constraints import Constraints
from trajopt.utils import tools


class SubproblemConstraints(Constraints):
    """
    Subclass of Constraints that inherits all functionality and adds
    the ability to own W, dual, vb (numpy arrays or CVXPY variables) per constraint.
    
    Aggregation methods allow Subproblem to collect these constraint-owned objects
    into type-aligned matrices for building the optimization problem.
    """
    
    def __init__(self, constraint_config_list, params):
        """
        Initialize SubproblemConstraints by calling parent constructor.
        """
        super().__init__(constraint_config_list, params)
    
    def attach_constraint_data(self, constraint_idx, vb=None, W=None, dual=None):
        """
        Attach additional data to a specific constraint object.
        """
        if constraint_idx < 0 or constraint_idx >= len(self.constraints_list):
            raise IndexError(f"Constraint index {constraint_idx} out of bounds")
        
        constraint = self.constraints_list[constraint_idx]
        
        if vb is not None:
            constraint.vb = vb
        if W is not None:
            constraint.W = W
        if dual is not None:
            constraint.dual = dual
    
    def attach_data_by_name(self, constraint_name, vb=None, W=None, dual=None):
        """
        Attach additional data to constraint(s) by name.
        """
        if constraint_name not in self.constraint_ids['name']:
            raise KeyError(f"Constraint '{constraint_name}' not found")
        
        indices = self.constraint_ids['name'][constraint_name]
        for idx in indices:
            self.attach_constraint_data(idx, vb=vb, W=W, dual=dual)
    
    def attach_data_by_type(self, constraint_type, vb=None, W=None, dual=None):
        """
        Attach additional data to all constraints of a specific type.
        """
        if constraint_type not in self.constraint_ids['type']:
            raise KeyError(f"Constraint type '{constraint_type}' not found")
        
        indices = self.constraint_ids['type'][constraint_type]
        for idx in indices:
            self.attach_constraint_data(idx, vb=vb, W=W, dual=dual)
    
    def get_constraint_data(self, constraint_idx, data_name=None):
        """
        Retrieve additional data from a constraint object.
        """
        if constraint_idx < 0 or constraint_idx >= len(self.constraints_list):
            raise IndexError(f"Constraint index {constraint_idx} out of bounds")
        
        constraint = self.constraints_list[constraint_idx]
        
        if data_name is not None:
            return getattr(constraint, data_name, None)
        else:
            data_dict = {}
            for attr in ['vb', 'W', 'dual']:
                if hasattr(constraint, attr):
                    data_dict[attr] = getattr(constraint, attr)
            return data_dict
    
    def has_data(self, constraint_idx, data_name):
        """
        Check if a constraint has attached data.
        """
        if constraint_idx < 0 or constraint_idx >= len(self.constraints_list):
            raise IndexError(f"Constraint index {constraint_idx} out of bounds")
        
        return hasattr(self.constraints_list[constraint_idx], data_name)
    
    # ====================================================================
    # AGGREGATION METHODS: Collect W, dual, vb from constraints by type
    # ====================================================================
    
    def aggregate_W_by_type(self, constraint_type):
        """
        Aggregate W matrices from all constraints of a given type by stacking columns.
        
        For nodal constraints (ineq, term), returns shape (N, total_dim) or (total_dim,).
        For ct/dynamics constraints, returns shape (N-1, total_dim) or (total_dim,).
        
        Args:
            constraint_type: str, e.g., "nonconvex_inequality", "equality_bc", "ct", etc.
        
        Returns:
            numpy array or None if no constraints of that type exist.
        """
        indices = self.constraint_ids['type'].get(constraint_type, [])
        if not indices:
            return None
        
        W_list = []
        for idx in indices:
            constraint = self.constraints_list[idx]
            if hasattr(constraint, 'W') and constraint.W is not None:
                W_list.append(constraint.W)
        
        if not W_list:
            return None
        
        # Stack along columns (axis=1) to build the aggregated matrix
        return np.hstack(W_list) if len(W_list) > 0 else None
    
    def aggregate_dual_by_type(self, constraint_type):
        """
        Aggregate dual (Lagrange multiplier) arrays from all constraints of a given type.
        
        Args:
            constraint_type: str, e.g., "nonconvex_inequality", "equality_bc", "ct", etc.
        
        Returns:
            numpy array or None if no constraints of that type exist.
        """
        indices = self.constraint_ids['type'].get(constraint_type, [])
        if not indices:
            return None
        
        dual_list = []
        for idx in indices:
            constraint = self.constraints_list[idx]
            if hasattr(constraint, 'dual') and constraint.dual is not None:
                dual_list.append(constraint.dual)
        
        if not dual_list:
            return None
        
        # Stack along columns (axis=1)
        return np.hstack(dual_list) if len(dual_list) > 0 else None
    
    def aggregate_vb_by_type(self, constraint_type):
        """
        Aggregate virtual buffers (vb) from all constraints of a given type.
        May be numpy arrays or CVXPY variables.
        
        Args:
            constraint_type: str, e.g., "nonconvex_inequality", "equality_bc", "ct", etc.
        
        Returns:
            list of vb arrays/variables, or None if no constraints of that type exist.
        """
        indices = self.constraint_ids['type'].get(constraint_type, [])
        if not indices:
            return None
        
        vb_list = []
        for idx in indices:
            constraint = self.constraints_list[idx]
            if hasattr(constraint, 'vb') and constraint.vb is not None:
                vb_list.append(constraint.vb)
        
        return vb_list if vb_list else None
    
    def get_all_W_matrices(self):
        """
        Return dict of all aggregated W matrices keyed by constraint type.
        """
        W_dict = {}
        for ctype in self.constraint_ids['type'].keys():
            W = self.aggregate_W_by_type(ctype)
            if W is not None:
                W_dict[ctype] = W
        return W_dict
    
    def get_all_dual_matrices(self):
        """
        Return dict of all aggregated dual matrices keyed by constraint type.
        """
        dual_dict = {}
        for ctype in self.constraint_ids['type'].keys():
            dual = self.aggregate_dual_by_type(ctype)
            if dual is not None:
                dual_dict[ctype] = dual
        return dual_dict
    
    def get_all_vb_lists(self):
        """
        Return dict of all vb lists keyed by constraint type.
        """
        vb_dict = {}
        for ctype in self.constraint_ids['type'].keys():
            vb_list = self.aggregate_vb_by_type(ctype)
            if vb_list is not None:
                vb_dict[ctype] = vb_list
        return vb_dict

    def update_from_weights(self, W: dict, method, problem) -> None:
        """
        Update each constraint's `W`, `dual`, and `vb` from a global
        weights dictionary (e.g., `method.weights`). This centralizes the
        mapping logic so callers (like `Subproblem`) don't need to
        duplicate assignment code.

        Args:
            W: weights dictionary
            method: Method object (for sizes like Npm)
            problem: Problem object (for sizes like nz, n_ineq)
        """
        # sizes
        N = int(method.N)
        n_ineq = getattr(problem, "n_ineq", 0)
        n_term_total = getattr(problem, "n_term_total", 0)
        nz = getattr(problem, "nz", getattr(problem, "n", 0))

        Npm_real        = int(getattr(method, "Npm_real", 0))
        n_plus_real     = int(getattr(method, "n_plus_real", 0))
        n_minus_real    = int(getattr(method, "n_minus_real", 0))
        Npm_ctcs        = int(getattr(method, "Npm_ctcs", 0))
        n_plus_ctcs     = int(getattr(method, "n_plus_ctcs", 0))
        n_minus_ctcs    = int(getattr(method, "n_minus_ctcs", 0))

        # ensure W is a dict-like
        W = W if isinstance(W, dict) else {}

        # Extract global arrays from weights
        W_ineq      = tools.ensure_shape(W.get("W_ineq", 0.0), (N, max(n_ineq, 1))) if n_ineq > 0 else np.zeros((N, 0))
        dual_ineq   = tools.ensure_shape(W.get("dual_ineq", 0.0), (N, max(n_ineq, 1))) if n_ineq > 0 else np.zeros((N, 0))

        W_term      = tools.ensure_shape(W.get("W_term", 0.0), (max(n_term_total, 1),)) if n_term_total > 0 else np.zeros((0,))
        dual_term   = tools.ensure_shape(W.get("dual_term", 0.0), (max(n_term_total, 1),)) if n_term_total > 0 else np.zeros((0,))

        W_dyn       = tools.ensure_shape(W.get("W_dyn", 0.0), (max(N - 1, 1), max(nz, 1))) if nz > 0 else np.zeros((max(N - 1, 0), 0))
        dual_dyn    = tools.ensure_shape(W.get("dual_dyn", 0.0), (max(N - 1, 1), max(nz, 1))) if nz > 0 else np.zeros((max(N - 1, 0), 0))

        W_plus_real = tools.ensure_shape(W.get("W_plus_real", 0.0), (max(Npm_real, 1), max(n_plus_real, 1))) if n_plus_real > 0 else np.zeros((0, 0))
        W_minus_real = tools.ensure_shape(W.get("W_minus_real", 0.0), (max(Npm_real, 1), max(n_minus_real, 1))) if n_minus_real > 0 else np.zeros((0, 0))
        W_plus_ctcs = tools.ensure_shape(W.get("W_plus_ctcs", 0.0), (max(Npm_ctcs, 1), max(n_plus_ctcs, 1))) if n_plus_ctcs > 0 else np.zeros((0, 0))
        W_minus_ctcs = tools.ensure_shape(W.get("W_minus_ctcs", 0.0), (max(Npm_ctcs, 1), max(n_minus_ctcs, 1))) if n_minus_ctcs > 0 else np.zeros((0, 0))

        dual_plus_real = tools.ensure_shape(W.get("dual_plus_real", 0.0), (max(Npm_real, 1), max(n_plus_real, 1))) if n_plus_real > 0 else np.zeros((0, 0))
        dual_minus_real = tools.ensure_shape(W.get("dual_minus_real", 0.0), (max(Npm_real, 1), max(n_minus_real, 1))) if n_minus_real > 0 else np.zeros((0, 0))
        dual_plus_ctcs = tools.ensure_shape(W.get("dual_plus_ctcs", 0.0), (max(Npm_ctcs, 1), max(n_plus_ctcs, 1))) if n_plus_ctcs > 0 else np.zeros((0, 0))
        dual_minus_ctcs = tools.ensure_shape(W.get("dual_minus_ctcs", 0.0), (max(Npm_ctcs, 1), max(n_minus_ctcs, 1))) if n_minus_ctcs > 0 else np.zeros((0, 0))

        # Populate nodal nonconvex inequalities with column slices
        if problem.constraints.has("nodal", "nonconvex_inequality"):
            offset = 0
            for constraint in self.get("nodal", "nonconvex_inequality"):
                dim = int(getattr(constraint, "dimension", 1))
                constraint.vb = np.zeros((N, dim))
                if dim > 0:
                    sl = slice(offset, offset + dim)
                    constraint.W = W_ineq[:, sl] if W_ineq.size > 0 else np.zeros((N, dim))
                    constraint.dual = dual_ineq[:, sl] if dual_ineq.size > 0 else np.zeros((N, dim))
                else:
                    constraint.W = np.zeros((N, 0))
                    constraint.dual = np.zeros((N, 0))
                offset += dim

        # Populate terminal constraints
        term_offset = 0
        for ctype in ("equality_bc", "inequality_bc"):
            if problem.constraints.has("nodal", ctype):
                for constraint in self.get("nodal", ctype):
                    dim = int(getattr(constraint, "dimension", 1))
                    constraint.vb = np.zeros((dim,))
                    if dim > 0:
                        sl = slice(term_offset, term_offset + dim)
                        constraint.W = W_term[sl] if W_term.size > 0 else np.zeros((dim,))
                        constraint.dual = dual_term[sl] if dual_term.size > 0 else np.zeros((dim,))
                    else:
                        constraint.W = np.zeros((0,))
                        constraint.dual = np.zeros((0,))
                    term_offset += dim

        # Populate ct / dynamics constraints
        if problem.constraints.has("ct"):
            for constraint in self.get("ct", "all"):
                dim = int(getattr(constraint, "dimension", nz)) if getattr(constraint, "dimension", None) is not None else nz
                constraint.vb = np.zeros((max(N - 1, 0), dim))
                if dim > 0:
                    constraint.W = W_dyn[:, :dim] if W_dyn.size > 0 else np.zeros((max(N - 1, 0), dim))
                    constraint.dual = dual_dyn[:, :dim] if dual_dyn.size > 0 else np.zeros((max(N - 1, 0), dim))
                else:
                    constraint.W = np.zeros((max(N - 1, 0), 0))
                    constraint.dual = np.zeros((max(N - 1, 0), 0))

        # Attach plus/minus aggregated buffers for constraints with explicit flags
        for constraint in self.constraints_list:
            if getattr(constraint, "use_plus_real", False):
                constraint.W_plus = W_plus_real
                constraint.dual_plus = dual_plus_real
            if getattr(constraint, "use_minus_real", False):
                constraint.W_minus = W_minus_real
                constraint.dual_minus = dual_minus_real
            if getattr(constraint, "use_plus_ctcs", False):
                constraint.W_plus = W_plus_ctcs
                constraint.dual_plus = dual_plus_ctcs
            if getattr(constraint, "use_minus_ctcs", False):
                constraint.W_minus = W_minus_ctcs
                constraint.dual_minus = dual_minus_ctcs


def build_subproblem_constraints(problem, method):
    """
    Construct a SubproblemConstraints wrapper and populate W, dual, vb
    on each individual constraint based on its type and dimension.
    
    W and dual are numpy arrays (fixed values) pulled from method.weights.
    vb is either zeros (as numpy array) or will be replaced with CVXPY variables
    by Subproblem when needed.
    
    Each constraint owns its W, dual, vb. Subproblem can then aggregate
    these to build the optimization problem.

    Args:
        problem: An instantiated Problem object (with `constraints` built).
        method: The Method object (provides `N`, `weights`, and other sizes).

    Returns:
        SubproblemConstraints: wrapper with W/dual/vb attached to each constraint.
    """
    subp_c          = SubproblemConstraints.__new__(SubproblemConstraints)

    # copy constraints list & ids (shallow copy of objects)
    subp_c.constraints_list = list(problem.constraints.constraints_list)
    subp_c.constraint_ids   = deepcopy(problem.constraints.constraint_ids)

    # sizes
    N               = int(method.N)
    n_ineq          = getattr(problem, "n_ineq", 0)
    n_term_total    = getattr(problem, "n_term_total", 0)
    nz              = getattr(problem, "nz", getattr(problem, "n", 0))

    Npm_real        = int(getattr(method, "Npm_real", 0))
    n_plus_real     = int(getattr(method, "n_plus_real", 0))
    n_minus_real    = int(getattr(method, "n_minus_real", 0))
    Npm_ctcs        = int(getattr(method, "Npm_ctcs", 0))
    n_plus_ctcs     = int(getattr(method, "n_plus_ctcs", 0))
    n_minus_ctcs    = int(getattr(method, "n_minus_ctcs", 0))

    W               = method.weights if hasattr(method, "weights") else {}

    # Fetch global weight/dual arrays from method.weights
    W_ineq          = tools.ensure_shape(W.get("W_ineq", 0.0), (N, max(n_ineq, 1))) if n_ineq > 0 else np.zeros((N, 0))
    dual_ineq       = tools.ensure_shape(W.get("dual_ineq", 0.0), (N, max(n_ineq, 1))) if n_ineq > 0 else np.zeros((N, 0))

    W_term          = tools.ensure_shape(W.get("W_term", 0.0), (max(n_term_total, 1),)) if n_term_total > 0 else np.zeros((0,))
    dual_term       = tools.ensure_shape(W.get("dual_term", 0.0), (max(n_term_total, 1),)) if n_term_total > 0 else np.zeros((0,))

    W_dyn           = tools.ensure_shape(W.get("W_dyn", 0.0), (max(N - 1, 1), max(nz, 1))) if nz > 0 else np.zeros((max(N - 1, 0), 0))
    dual_dyn        = tools.ensure_shape(W.get("dual_dyn", 0.0), (max(N - 1, 1), max(nz, 1))) if nz > 0 else np.zeros((max(N - 1, 0), 0))

    W_plus_real     = tools.ensure_shape(W.get("W_plus_real", 0.0), (max(Npm_real, 1), max(n_plus_real, 1))) if n_plus_real > 0 else np.zeros((0, 0))
    W_minus_real    = tools.ensure_shape(W.get("W_minus_real", 0.0), (max(Npm_real, 1), max(n_minus_real, 1))) if n_minus_real > 0 else np.zeros((0, 0))
    W_plus_ctcs     = tools.ensure_shape(W.get("W_plus_ctcs", 0.0), (max(Npm_ctcs, 1), max(n_plus_ctcs, 1))) if n_plus_ctcs > 0 else np.zeros((0, 0))
    W_minus_ctcs    = tools.ensure_shape(W.get("W_minus_ctcs", 0.0), (max(Npm_ctcs, 1), max(n_minus_ctcs, 1))) if n_minus_ctcs > 0 else np.zeros((0, 0))

    dual_plus_real  = tools.ensure_shape(W.get("dual_plus_real", 0.0), (max(Npm_real, 1), max(n_plus_real, 1))) if n_plus_real > 0 else np.zeros((0, 0))
    dual_minus_real = tools.ensure_shape(W.get("dual_minus_real", 0.0), (max(Npm_real, 1), max(n_minus_real, 1))) if n_minus_real > 0 else np.zeros((0, 0))
    dual_plus_ctcs  = tools.ensure_shape(W.get("dual_plus_ctcs", 0.0), (max(Npm_ctcs, 1), max(n_plus_ctcs, 1))) if n_plus_ctcs > 0 else np.zeros((0, 0))
    dual_minus_ctcs = tools.ensure_shape(W.get("dual_minus_ctcs", 0.0), (max(Npm_ctcs, 1), max(n_minus_ctcs, 1))) if n_minus_ctcs > 0 else np.zeros((0, 0))

    # ====================================================================
    # Attach W, dual, vb to each constraint individually (by type)
    # ====================================================================
    
    # Populate nodal nonconvex inequalities with column slices
    if problem.constraints.has("nodal", "nonconvex_inequality"):
        offset = 0
        for constraint in subp_c.get("nodal", "nonconvex_inequality"):
            dim = int(getattr(constraint, "dimension", 1))
            # vb: initialize to zeros (may become CVXPY variables later)
            constraint.vb = np.zeros((N, dim))
            if dim > 0:
                sl = slice(offset, offset + dim)
                constraint.W = W_ineq[:, sl] if W_ineq.size > 0 else np.zeros((N, dim))
                constraint.dual = dual_ineq[:, sl] if dual_ineq.size > 0 else np.zeros((N, dim))
            else:
                constraint.W = np.zeros((N, 0))
                constraint.dual = np.zeros((N, 0))
            offset += dim

    # Populate terminal constraints (equality_bc and inequality_bc)
    term_offset = 0
    for ctype in ("equality_bc", "inequality_bc"):
        if problem.constraints.has("nodal", ctype):
            for constraint in subp_c.get("nodal", ctype):
                dim = int(getattr(constraint, "dimension", 1))
                # vb: initialize to zeros
                constraint.vb = np.zeros((dim,))
                if dim > 0:
                    sl = slice(term_offset, term_offset + dim)
                    constraint.W = W_term[sl] if W_term.size > 0 else np.zeros((dim,))
                    constraint.dual = dual_term[sl] if dual_term.size > 0 else np.zeros((dim,))
                else:
                    constraint.W = np.zeros((0,))
                    constraint.dual = np.zeros((0,))
                term_offset += dim

    # Populate ct / dynamics constraints
    if problem.constraints.has("ct"):
        for constraint in subp_c.get("ct", "all"):
            dim = int(getattr(constraint, "dimension", nz)) if getattr(constraint, "dimension", None) is not None else nz
            # vb: initialize to zeros
            constraint.vb = np.zeros((max(N - 1, 0), dim))
            if dim > 0:
                constraint.W = W_dyn[:, :dim] if W_dyn.size > 0 else np.zeros((max(N - 1, 0), dim))
                constraint.dual = dual_dyn[:, :dim] if dual_dyn.size > 0 else np.zeros((max(N - 1, 0), dim))
            else:
                constraint.W = np.zeros((max(N - 1, 0), 0))
                constraint.dual = np.zeros((max(N - 1, 0), 0))

    # Attach plus/minus aggregated buffers for constraints with explicit flags
    for constraint in subp_c.constraints_list:
        if getattr(constraint, "use_plus_real", False):
            constraint.W_plus = W_plus_real
            constraint.dual_plus = dual_plus_real
        if getattr(constraint, "use_minus_real", False):
            constraint.W_minus = W_minus_real
            constraint.dual_minus = dual_minus_real
        if getattr(constraint, "use_plus_ctcs", False):
            constraint.W_plus = W_plus_ctcs
            constraint.dual_plus = dual_plus_ctcs
        if getattr(constraint, "use_minus_ctcs", False):
            constraint.W_minus = W_minus_ctcs
            constraint.dual_minus = dual_minus_ctcs

    return subp_c


def initialize_subproblem_constraints(subp_c, problem, method):
    """
    Initialize all vb, W, and dual arrays in an existing SubproblemConstraints
    to zeros.
    
    W and dual remain zero-valued numpy arrays.
    vb is zero-initialized (may be replaced with CVXPY variables later).
    
    Use this function when you want to reset all constraint data to zero state
    before any Subproblem-specific modifications.

    Args:
        subp_c: A SubproblemConstraints instance (from build_subproblem_constraints).
        problem: The instantiated Problem object.
        method: The Method object (provides `N` and other sizes).

    Returns:
        subp_c: The modified SubproblemConstraints with all arrays zero-initialized.
    """
    # sizes
    N = int(method.N)
    nz = getattr(problem, "nz", getattr(problem, "n", 0))

    # Initialize all constraints with zero-valued vb, W, dual
    
    # Iteration 1: nodal nonconvex inequalities
    if problem.constraints.has("nodal", "nonconvex_inequality"):
        for constraint in subp_c.get("nodal", "nonconvex_inequality"):
            dim = int(getattr(constraint, "dimension", 1))
            constraint.vb = np.zeros((N, dim))
            constraint.W = np.zeros((N, dim))
            constraint.dual = np.zeros((N, dim))

    # Iteration 2: terminal constraints (equality_bc and inequality_bc)
    for ctype in ("equality_bc", "inequality_bc"):
        if problem.constraints.has("nodal", ctype):
            for constraint in subp_c.get("nodal", ctype):
                dim = int(getattr(constraint, "dimension", 1))
                constraint.vb = np.zeros((dim,))
                constraint.W = np.zeros((dim,))
                constraint.dual = np.zeros((dim,))

    # Iteration 3: ct / dynamics constraints
    if problem.constraints.has("ct"):
        for constraint in subp_c.get("ct", "all"):
            dim = int(getattr(constraint, "dimension", nz)) if getattr(constraint, "dimension", None) is not None else nz
            constraint.vb = np.zeros((max(N - 1, 0), dim))
            constraint.W = np.zeros((max(N - 1, 0), dim))
            constraint.dual = np.zeros((max(N - 1, 0), dim))

    return subp_c
