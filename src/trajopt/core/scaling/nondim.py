import numpy as np

# example usage: set_nondim_params(["d", "d", "d", "v", "v", "v"], ["f", "f"], [("d", 10), ("v", 10), ("m", 1)], params)


class Nondim:

    def __init__(self, problem):

        """
        Initializes all nondimensional parameters
        """

        n = problem.n
        m = problem.m

        # this solves the following linear system to backout base scales for
        # distance, time, and mass:
        # A @ ln([d, t, m]^T) = ln([anchor0, anchor1, anchor2]^T)
        # then ([d, t, m]^T) = exp(log([d, t, m]^T))

        exponents = {
            "d": np.array([1,  0,  0]),
            "t": np.array([0,  1,  0]),
            "m": np.array([0,  0,  1]),
            "v": np.array([1, -1,  0]),
            "a": np.array([1, -2,  0]),
            "f": np.array([1, -2,  1]),
        }

        A = np.vstack([exponents[key] for key in problem.params['model']['nondim']['anchor_types']])
        b = np.log(np.array([val for val in problem.params['model']['nondim']['anchor_scales']]))

        log_base_scales = np.linalg.solve(A, b)
        base_scales = np.exp(log_base_scales)

        # retrieve remaining scales from base scales
        nd = base_scales[0]
        nt = base_scales[1]
        nm = base_scales[2]

        self.scales = {
            "d"    : nd,
            "t"    : nt,
            "m"    : nm,
            "v"    : nd / nt,
            "a"    : nd / (nt**2),
            "f"    : nm * nd / (nt**2),
            "ang"  : 180 / np.pi,
            "angv" : (180 / np.pi) / nt,
            "none": 1.0 
        }

        d_lbl = "m"
        t_lbl = "s"
        m_lbl = "kg"

        self.scale_labels = {
            "d"    : d_lbl,
            "t"    : t_lbl,
            "m"    : m_lbl,
            "v"    : f"{d_lbl} / {t_lbl}" ,
            "a"    : f"{d_lbl} / ({t_lbl}^2)",
            "f"    : f"{m_lbl} * {d_lbl} / ({t_lbl}^2)",
            "ang"  : "deg",
            "angv" : f"deg / {t_lbl}",
            "none": ""
        }

        print("scales: ")
        print(", ".join(f"{k}: {v:.4f}" for k, v in self.scales.items()))

        self.z_types = problem.params['model']['nondim']['z_types']
        self.u_types = problem.params['model']['nondim']['u_types']

        self.nd_state = np.array([self.scales[self.z_types[i]] for i in range(n)])
        self.nd_ctrl  = np.array([self.scales[self.u_types[i]] for i in range(m)])

        self.M          = {}
        self.M["state"] = {}
        self.M["ctrl"]  = {}
        self.M["ineq_nodal"] = {}
        self.M["ineq_ct"] = {}
        self.M["term_total"] = {}

        self.labels     = {}

        self.M["state"]["d2nd"] = np.diag(1 / self.nd_state).copy()
        self.M["state"]["nd2d"] = np.diag(self.nd_state).copy()
        self.M["ctrl"]["d2nd"] = np.diag(1 / self.nd_ctrl).copy()
        self.M["ctrl"]["nd2d"] = np.diag(self.nd_ctrl).copy()

        # add scalar nondim variables to nondim substruct
        self.nd = self.scales["d"]
        self.na = self.scales["a"]
        self.nt = self.scales["t"]
        self.nt_inv = 1 / self.nt
        self.nv = self.scales["v"]
        self.nm = self.scales["m"]
        self.nm_dot = self.nm / self.nt
        self.nf = self.scales["f"]
        self.nang = self.scales["ang"]
        self.nangv = self.nang / self.nt
        self.labels["state"] = [self.scale_labels[self.z_types[i]] for i in range(n)]
        self.labels["ctrl"]  = [self.scale_labels[self.u_types[i]] for i in range(m)]

        # set nondim for cost and constraints
        
        # TODO(carlos): setting to one rn, need to think about how to properly specifiy this
        # when there are multiple summed cost functions

        nd_cost = 1.0
        self.nd_cost = nd_cost

        # nodal inequality nondim
        self.nd_ineq = np.ones(problem.n_ineq)
        
        idx = 0
        # TODO: change 'nonconvex_inequality' to 'inequality' once we add general buffering
        for constraint in problem.constraints.get('nodal', 'nonconvex_inequality'):
            if getattr(constraint, 'units', None) is not None:
                for unit in constraint.units:
                    scale_list = [self.scales[unit_type]**exponent for unit_type, exponent in unit.items()]
                    scale = np.prod(scale_list)
                    self.nd_ineq[idx:idx + constraint.dimension] = scale
            idx += constraint.dimension

        self.M["ineq_nodal"]["d2nd"] = np.diag(1 / self.nd_ineq).copy()
        self.M["ineq_nodal"]["nd2d"] = np.diag(self.nd_ineq).copy()

        # ct inequality nondim
        self.nd_ctcs = np.ones(problem.n_ctcs)
        idx = 0
        # TODO: change 'nonconvex_inequality' to 'inequality' once we add general buffering
        for constraint in problem.constraints.get('ct', 'nonconvex_inequality'):
            if getattr(constraint, 'units', None) is not None:
                for unit in constraint.units:
                    scale_list = [self.scales[unit_type]**exponent for unit_type, exponent in unit.items()]
                    scale = np.prod(scale_list)
                    self.nd_ctcs[idx:idx + constraint.dimension] = scale
            idx += constraint.dimension

        self.M["ineq_ct"]["d2nd"] = np.diag(1 / self.nd_ctcs).copy()
        self.M["ineq_ct"]["nd2d"] = np.diag(self.nd_ctcs).copy()

        # TODO (CARLOS): fix this pls, set to all ones so i can run it
        # the problem is vb_term mixes different constraint types 
        # together so indexing into is a bit tricky

        terminal_constraint = problem.constraints.get('name', 'final_state')[0]
        x_idx = terminal_constraint.x_idx

        self.nd_term_total = np.hstack([self.nd_state[x_idx], np.ones(problem.n_term_ineq), np.ones(problem.n_ctcs)])


        self.M["term_total"]["d2nd"] = np.diag(1 / self.nd_term_total).copy()
        self.M["term_total"]["nd2d"] = np.diag(self.nd_term_total).copy()

    def build_nondim_matrix(self, units_list):
        """
        Build a diagonal nondimensionalization matrix from a list of unit dictionaries.
        
        Args:
            units_list: List of dictionaries specifying units as {type: exponent}.
                        e.g., [{'d': 1, 't': -1}, {'f': 1}] for [velocity, force]
                        Supported types: 'd', 't', 'm', 'v', 'a', 'f', 'ang', 'angv', 'none'
            method: Method object containing nondim scales
        
        Returns:
            M_d2nd: Diagonal matrix (n x n) for dim-to-nondim conversion
            M_nd2d: Diagonal matrix (n x n) for nondim-to-dim conversion
        
        Example:
            units_list = [{'d': 1, 't': -1}, {'f': 1}]  # [velocity, force]
            M_d2nd, M_nd2d = nondim.build_nondim_matrix(units_list, method)
            # M_d2nd @ [v_dim, f_dim] = [v_nondim, f_nondim]
        """
        scales = self.scales
        
        # Compute scale for each entry
        nd_scales = np.zeros(len(units_list))
        
        for i, unit_dict in enumerate(units_list):
            if unit_dict is None or len(unit_dict) == 0:
                # No units specified, use scale of 1
                nd_scales[i] = 1.0
            else:
                # Multiply scales raised to their exponents
                scale = 1.0
                for unit_type, exponent in unit_dict.items():
                    if unit_type in scales:
                        scale *= scales[unit_type] ** exponent
                    else:
                        raise ValueError(f"Unknown unit type: {unit_type}. "
                                         f"Valid types: {list(scales.keys())}")
                nd_scales[i] = scale
        
        M_d2nd = np.diag(1.0 / nd_scales)
        M_nd2d = np.diag(nd_scales)
        
        return M_d2nd, M_nd2d

    def nondim_function(self, fcn, M_state_nd2d, M_ctrl_nd2d, M_out_d2nd):
        
        def wrapped_fcn(t, z, nu, *args, **kwargs):
            return M_out_d2nd @ fcn(t, M_state_nd2d @ z, M_ctrl_nd2d @ nu, *args, **kwargs)
        
        return wrapped_fcn