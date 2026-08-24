import numpy as np
import cvxpy as cp

from trajopt.utils.tools import deep_merge
from trajopt.methods.common import penalties


class SCPConstraint():
    nonnegative_dual = False

    STATE_INDEXED   = ("dynamics", "initial_state", "final_state", "state_continuity")
    CONTROL_INDEXED = ("initial_control", "final_control", "control_continuity")

    def __init__(self, constraint, scp_segment) -> None:
        self.constraint    = constraint
        self.type          = constraint.type
        self.name          = constraint.name
        self.penalty       = None
        self.shape         = None
        self.penalty_state = penalties.Penalty(shape=(0,), nonnegative_dual=self.nonnegative_dual)

    def compile(self, scp_segment): pass

    def build_cross_segment(self, scp_segments): pass

    def create_cvxpy_parameters(self, scp_segment): pass
    def create_cvxpy_variables(self, scp_segment): pass
    def create_cvxpy_constraints(self, scp_segment): pass
    def update_cvxpy_parameters(self, scp_segment): pass
    def update_current_iter_data(self, scp_segment): pass

    def init_penalty(self, scp_segment): pass

    def _component_labels(self, scp_segment, width):
        """Component name for each buffer slot, or None if the slots are unnamed."""
        index_map = scp_segment.index_map

        if self.type in self.STATE_INDEXED:
            group = index_map.components.state
        elif self.type in self.CONTROL_INDEXED:
            group = index_map.components.control
        else:
            return None

        position = {}
        for name, idx in group.items():
            for i in np.atleast_1d(idx):
                position[int(i)] = name

        # the dynamics buffer covers z = [x, t, beta, gamma]
        if self.type == "dynamics":
            labels  = [position.get(i) for i in range(index_map.n.state)]
            labels += ["t"] * index_map.n.time
            return labels + [None] * (width - len(labels))

        idx = getattr(self.constraint, "idx", None)
        if idx is not None:
            idx = np.atleast_1d(idx)
            if len(idx) == width:
                return [position.get(int(i)) for i in idx]
            return None

        if width == len(position):
            return [position.get(i) for i in range(width)]
        return None

    def _resolve_eps(self, scp_segment, shape):
        """Feasibility tolerance: constraint eps, else penalty eps, else 1e-4.

        A mapping sets the components it names, the rest take its 'default' key.
        """
        raw = getattr(self.constraint, 'eps', None)
        if raw is None:
            raw = getattr(self.penalty, 'eps', None) if self.penalty else None
        if raw is None:
            raw = 1e-4

        if not hasattr(raw, 'items'):
            return np.broadcast_to(np.atleast_1d(raw), (shape[-1],)).copy()

        entries = dict(raw)
        default = entries.pop('default', 1e-4)
        eps     = np.broadcast_to(np.atleast_1d(default), (shape[-1],)).astype(float).copy()

        labels = self._component_labels(scp_segment, shape[-1])
        if labels is None:
            raise ValueError(
                f"constraint '{self.name}' ({self.type}): eps names components "
                f"{sorted(entries)}, but this constraint's buffer has no named "
                "components. Give a single value or a full-length list instead."
            )

        for name, value in entries.items():
            slots = [i for i, label in enumerate(labels) if label == name]
            if not slots:
                known = sorted(x for x in set(labels) if x is not None)
                raise ValueError(
                    f"constraint '{self.name}' ({self.type}): eps names "
                    f"'{name}', which is not one of its components {known}"
                )
            eps[slots] = float(value)
        return eps

    def _alloc_penalty(self, scp_segment, shape):
        # a penalty.<type> block only names the keys it changes
        default  = scp_segment.penalty_config.get('default')
        override = scp_segment.penalty_config.get(self.type)
        if override is None:
            self.penalty = default
        elif default is None:
            self.penalty = override
        else:
            self.penalty = deep_merge(default, override)

        self.shape = shape
        vb_type    = getattr(self.penalty, 'vb', 'standard') if self.penalty else 'standard'
        # under l1 the parameters named W_sqrt carry W itself
        norm       = getattr(self.penalty, 'norm', 'l2') if self.penalty else 'l2'

        self.penalty_state = penalties.Penalty(
            shape=shape, vb_type=vb_type, norm=norm,
            cfg=self.penalty, nonnegative_dual=self.nonnegative_dual,
        )
        self.penalty_state.eps = self._resolve_eps(scp_segment, shape)
        self.penalty_state.init_values()

    def create_penalty_parameters(self, scp_segment):
        p = self.penalty_state
        if self.shape is None:
            return
        if p.vb_type == "none":
            return
        if p.vb_type == "split":
            p.W_p_sqrt_param = cp.Parameter(self.shape, nonneg=True, name=f"W_p_{self.name}_sqrt", value=np.zeros(self.shape))
            p.W_m_sqrt_param = cp.Parameter(self.shape, nonneg=True, name=f"W_m_{self.name}_sqrt", value=np.zeros(self.shape))
            p.dual_p_param   = cp.Parameter(self.shape, name=f"dual_p_{self.name}", value=np.zeros(self.shape))
            p.dual_m_param   = cp.Parameter(self.shape, name=f"dual_m_{self.name}", value=np.zeros(self.shape))
        else:
            p.W_sqrt_param   = cp.Parameter(self.shape, nonneg=True, name=f"W_{self.name}_sqrt", value=np.zeros(self.shape))
            p.dual_param     = cp.Parameter(self.shape, name=f"dual_{self.name}", value=np.zeros(self.shape))

    def create_penalty_variables(self, scp_segment):
        p = self.penalty_state
        if self.shape is None:
            return
        if p.vb_type == "none":
            return
        if p.vb_type == "split":
            p.vb_p_var = cp.Variable(self.shape, nonneg=True, name=f"vb_p_{self.name}_{scp_segment.name}")
            p.vb_m_var = cp.Variable(self.shape, nonneg=True, name=f"vb_m_{self.name}_{scp_segment.name}")
            p.vb_var   = p.vb_p_var - p.vb_m_var
        else:
            p.vb_var = cp.Variable(self.shape, name=f"vb_{self.name}_{scp_segment.name}")

    def add_penalty_cost(self, scp_segment):
        scp_segment.cp_cost += penalties.w_penalty_cost(self.penalty_state)
        scp_segment.cp_cost += penalties.dual_penalty_cost(self.penalty_state)

    def update_penalty_parameters(self, scp_segment):
        penalties.push_penalty_values(self.penalty_state)

    def read_vb(self, scp_segment):
        penalties.pull_vb(self.penalty_state)

    def update_W_dual(self, scp_segment, alpha=1.0):
        if self.penalty is None:
            return
        if not hasattr(self.penalty, 'W'):
            return

        # autotune W (penalty weight update)
        if self.penalty.W.autotune:
            penalties.autotune_W(self.penalty_state)

        # autotune dual (dual ascent update, autotune1 style)
        if self.penalty.dual.autotune:
            penalties.autotune_dual(self.penalty_state)

    @property
    def vb_ratio(self):
        if self.penalty_state.vb.size == 0:
            return 0.0
        return float(np.max(np.abs(self.penalty_state.vb) / self.penalty_state.eps))

    @property
    def is_feasible(self):
        if self.penalty_state.vb.size == 0:
            return True
        return bool(np.all(np.abs(self.penalty_state.vb) <= self.penalty_state.eps))
