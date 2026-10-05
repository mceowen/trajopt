import copy
import time

import numpy as np
import cvxpy as cp

from trajopt.methods.common import initial_guess
import trajopt.methods.common.scp.constraint_types as scp_constraint_type_module
import trajopt.methods.common.scp.cost_types as scp_cost_type_module
from trajopt.utils.tools import AttrDict, recursive_attrdict, resolve_function_from_string


def _resolve_fcns_tree(node):
    # resolves 'path.py:func' leaves to callables at any nesting depth, e.g. default/<phase_name>
    if isinstance(node, str):
        return resolve_function_from_string(node)
    if node is None:
        return None
    return AttrDict({name: _resolve_fcns_tree(value) for name, value in node.items()})


class Subproblem():
    """The convex subproblem for one formulation object (a Phase or a whole flat Trajectory).

    Holds everything formulation-agnostic: cvxpy variables/parameters/constraints/cost,
    the trust region, the constraint penalties, and the SCP step. A formulation that splits
    a trajectory into multiple phases (e.g. dev.phases) subclasses this to add whatever
    links one subproblem to another -- see SCPPhase's boundary/continuity check.

    ``constraint_type_module``/``cost_type_module`` resolve each constraint's/cost's
    ``scp_<type>`` class.
    """

    constraint_type_module = scp_constraint_type_module
    cost_type_module       = scp_cost_type_module

    def __init__(self, formulation_obj, method_config: AttrDict) -> None:

        self.name           = formulation_obj.name
        self.phase          = formulation_obj  # analysis.py reaches params/nondim/outputs through this
        self.method_config  = method_config
        self.index_map      = formulation_obj.index_map
        self.nondim         = formulation_obj.nondim
        self.params         = formulation_obj.params

        # hyperparams/fcns are {<module>: {<name>: <value>}}, mirroring a trajectory's fcns:/params:
        method_hyperparams = method_config.get('hyperparams', AttrDict())
        method_fcns        = method_config.get('fcns', AttrDict())

        self.hyperparams = AttrDict()
        self.fcns        = AttrDict()
        for module_name in set(method_hyperparams) | set(method_fcns):
            self.hyperparams[module_name] = method_hyperparams.get(module_name, AttrDict())
            self.fcns[module_name] = _resolve_fcns_tree(method_fcns.get(module_name, AttrDict()))

        self._trust_region_penalty_fcn = self.fcns.trust_region.penalty
        self._convergence_check_fcn    = self.fcns.convergence.check

        # troubleshooting is opt-in: unset unless a config wires fcns.troubleshoot.*
        troubleshoot_fcns = self.fcns.get('troubleshoot', AttrDict())
        self._troubleshoot_on_error_fcn = troubleshoot_fcns.get('on_error')
        self._troubleshoot_on_step_fcn  = troubleshoot_fcns.get('on_step')

        # Dictionary of constraints for this subproblem
        self.constraints = AttrDict()
        for cnstr_name, constraint in formulation_obj.constraints.items():
            scp_class_name = f"scp_{constraint.type}"
            constraintClass = getattr(self.constraint_type_module, scp_class_name)
            self.constraints[cnstr_name] = constraintClass(constraint, self)

        # Dictionary of cost types for this subproblem
        self.costs = AttrDict()
        for cost_name, cost in formulation_obj.costs.items():
            scp_class_name = f"scp_{cost.type}"
            costClass = getattr(self.cost_type_module, scp_class_name)
            self.costs[cost_name] = costClass(cost, self)

        self.free_final_time = self.derive_free_final_time()

        self.initialize()

        self.cp_params            = AttrDict()
        self.cp_vars              = AttrDict()
        self.cp_constraints       = []
        self.cp_cost              = 0
        self.cp_subproblem_status = None
        self.tr_scale             = 1.0

        self.create_cvxpy_parameters()
        self.create_cvxpy_variables()
        self.create_cvxpy_constraints()
        self.create_cvxpy_cost()

    def find_constraint(self, cnstr_type):
        """This subproblem's constraint of the given type, or None."""
        return next((c.constraint for c in self.constraints.values() if c.type == cnstr_type), None)

    def penalty_snapshot(self) -> AttrDict:
        """Per-constraint penalty state, keyed by field then constraint name.

        Field names come from each constraint's own hyperparams.penalties block
        (Penalties.names), not an assumed list -- a constraint penalized through
        some other named weight is picked up automatically.
        """
        snapshot = AttrDict()
        for constraint in self.constraints.values():
            if constraint.shape is None:
                continue
            p = constraint.penalties
            fields = ["vb"] + list(p.names)
            if p.vb_type == "split":
                fields += [f"{name}_p" for name in p.names] + [f"{name}_m" for name in p.names]
            for field in fields:
                snapshot.setdefault(field, AttrDict())[constraint.name] = getattr(p, field)
        return snapshot

    def derive_free_final_time(self) -> bool:
        """False only when initial_time and a fixed final_time pin both ends."""
        initial = self.find_constraint("initial_time")
        final   = self.find_constraint("final_time")
        self._validate_time_config(initial, final)
        return not (initial is not None and final is not None and final.is_fixed)

    def _validate_time_config(self, initial, final) -> None:
        """Hook for a subclass to reject invalid combinations of time constraints."""
        pass

    def _anchor_start_time(self) -> bool:
        """Whether this subproblem's own t=0 should be pinned via dt[0,0]==0.

        A subclass whose start epoch is inherited from elsewhere (e.g. a
        chained phase) overrides this to skip the anchor when its own
        local epoch floats freely instead.
        """
        return True

    def initialize(self) -> None:
        formulation_obj     = self.phase
        self.initial_guess  = AttrDict()

        # the guess supplies whichever end the constraints do not give
        initial_time_cnstr    = self.find_constraint("initial_time")
        final_time_cnstr      = self.find_constraint("final_time")

        t_start_nd            = (initial_time_cnstr.value if initial_time_cnstr is not None
                                 else initial_guess.guess_field(formulation_obj, self, 't_start', 0.0) / self.nondim.time_scale)

        if not self.free_final_time:
            t_stop_nd = final_time_cnstr.fixed_value
        else:
            t_stop_nd = initial_guess.guess_field(formulation_obj, self, 't_stop') / self.nondim.time_scale

        self.Ts_init          = t_stop_nd - t_start_nd
        t_init                = np.linspace(t_start_nd, t_stop_nd, self.index_map.N.all)
        dt_init               = np.diff(t_init)
        self.initial_guess.t  = t_init
        self.initial_guess.dt = dt_init

        for constraint in self.constraints.values():
            constraint.compile(self)
            constraint.init_penalty(self)

        dyn = next((c for c in self.constraints.values() if c.type == "dynamics"), None)
        self.eps_dyn   = dyn.penalties.eps.copy() if dyn is not None else np.full(self.index_map.n.z, 1e-4)
        self.eps_dyn[self.index_map.indices.z.running_cost] = np.inf

        self.fcns.convergence.init_tolerances(self)

        self.fcns.initial_guess.set(formulation_obj, self)

        self.iter_data_list = []

        self.current_iter_data = recursive_attrdict({
            "iter_num": 0,
            "z_opt": self.initial_guess.z,
            "nu_opt": self.initial_guess.nu,
            "t_start": float(self.initial_guess.t[0]) * self.nondim.time_scale,
            "t_final": float(self.initial_guess.t[-1]) * self.nondim.time_scale,
            "cost": 0.0,
            "penalty_cost": 0.0,
            **self.penalty_snapshot(),
        })

        self.iter_data_list.append(copy.deepcopy(self.current_iter_data))

    def create_cvxpy_parameters(self) -> None:
        N       = self.index_map.N.all
        n_z     = self.index_map.n.z
        n_nu    = self.index_map.n.nu

        self.cp_params.z_ref  = cp.Parameter((N, n_z),  name="z_ref")
        self.cp_params.nu_ref = cp.Parameter((N, n_nu), name="nu_ref")

        self.x_ref, self.t_ref, self.beta_ref, self.u_ref, self.s_ref = self.index_map.unpack_znu(self.cp_params.z_ref, self.cp_params.nu_ref)

        self.fcns.trust_region.create_params(self)

        self.cp_params.dcostdx = cp.Parameter((N, n_z),  name="dcostdx")
        self.cp_params.dcostdu = cp.Parameter((N, n_nu), name="dcostdu")
        self.cp_params.cost0   = cp.Parameter((N,),      name="cost0")

        self.fcns.discretize.create_time_params(self)

        for constraint in self.constraints.values():
            constraint.create_penalty_parameters(self)
            constraint.create_cvxpy_parameters(self)

    def create_cvxpy_variables(self) -> None:
        N, n_x, n_t, n_u = self.index_map.N.all, self.index_map.n.state, self.index_map.n.time, self.index_map.n.control
        n_ctcs = self.index_map.n.ctcs
        n_rc   = self.index_map.n.running_cost

        self.cp_vars.dx     = cp.Variable((N, n_x),    name="dx")
        self.cp_vars.dbeta  = cp.Variable((N, n_ctcs), name="dbeta")  if n_ctcs > 0 else None
        self.cp_vars.dgamma = cp.Variable((N, n_rc),   name="dgamma") if n_rc   > 0 else None
        self.cp_vars.du     = cp.Variable((N, n_u),    name="du")

        if self.free_final_time:
            self.cp_vars.dt = cp.Variable((N, n_t), name="dt")
            self.cp_vars.ds = cp.Variable((N, 1),   name="ds")
        else:
            self.cp_vars.dt = cp.Constant(np.zeros((N, n_t)))
            self.cp_vars.ds = cp.Constant(np.zeros((N, 1)))

        dz_components = [self.cp_vars.dx, self.cp_vars.dt]

        if n_ctcs > 0:
            dz_components.append(self.cp_vars.dbeta)
        if n_rc > 0:
            dz_components.append(self.cp_vars.dgamma)

        self.dz  = cp.hstack(dz_components)
        self.dnu = cp.hstack([self.cp_vars.du, self.cp_vars.ds])
        self.dt  = self.cp_vars.dt
        self.ds  = self.cp_vars.ds

        for constraint in self.constraints.values():
            constraint.create_penalty_variables(self)
            constraint.create_cvxpy_variables(self)

    def create_cvxpy_constraints(self) -> None:
        for constraint in self.constraints.values():
            constraint.create_cvxpy_constraints(self)

        if self.free_final_time:
            self.create_free_final_time_constraints()

    def create_cvxpy_cost(self) -> None:
        for cost in self.costs.values():
            cost.create_cvxpy_cost(self)

        self._trust_region_penalty_fcn(self)

        for constraint in self.constraints.values():
            constraint.add_penalty_cost(self)

    def create_free_final_time_constraints(self) -> None:
        if self._anchor_start_time():
            self.cp_constraints.append(self.dt[0, 0] == 0)

        self.fcns.discretize.create_time_constraints(self)

    def update_cvxpy_parameters(self) -> None:
        z_opt  = self.current_iter_data.z_opt
        nu_opt = self.current_iter_data.nu_opt

        self.cp_params.z_ref.value  = z_opt
        self.cp_params.nu_ref.value = nu_opt

        self.x_ref, self.t_ref, self.beta_ref, self.u_ref, self.s_ref = self.index_map.unpack_znu(z_opt, nu_opt)

        self.fcns.discretize.update_time_params(self)

        disc_start_time = time.perf_counter()

        for constraint in self.constraints.values():
            constraint.update_cvxpy_parameters(self)

        for cost in self.costs.values():
            cost.update_cvxpy_parameters(self)

        disc_end_time = time.perf_counter()

        self.current_iter_data.discretization_time = (disc_end_time - disc_start_time) * 1000

        self.fcns.trust_region.update_params(self)

        for constraint in self.constraints.values():
            constraint.update_penalty_parameters(self)

    def read_solution(self) -> None:
        self._dz_new  = self.dz.value
        self._dnu_new = self.dnu.value

    def apply_step(self, alpha: float = 1.0) -> None:
        dz_new  = self._dz_new
        dnu_new = self._dnu_new

        self.current_iter_data.dz    = alpha * dz_new
        self.current_iter_data.dnu   = alpha * dnu_new
        self.current_iter_data.alpha = alpha

        z_new  = self.current_iter_data.z_opt  + alpha * dz_new
        nu_new = self.current_iter_data.nu_opt + alpha * dnu_new

        self.current_iter_data.z_opt  = z_new
        self.current_iter_data.nu_opt = nu_new

        x_opt_new, t_opt_new, beta_opt_new, u_opt_new, s_opt_new = self.index_map.unpack_znu(z_new, nu_new)

        t_start_new = float(np.asarray(t_opt_new[0]).ravel()[0])
        t_final_new = float(np.asarray(t_opt_new[-1]).ravel()[0])

        self.current_iter_data.x_opt    = x_opt_new
        self.current_iter_data.t_opt    = t_opt_new
        self.current_iter_data.t_start  = t_start_new * self.nondim.time_scale
        self.current_iter_data.t_final  = t_final_new * self.nondim.time_scale
        self.current_iter_data.beta_opt = beta_opt_new
        self.current_iter_data.u_opt    = u_opt_new
        self.current_iter_data.s_opt    = s_opt_new

        for constraint in self.constraints.values():
            constraint.read_vb(self)

        self.current_iter_data.cost, self.current_iter_data.penalty_cost = self.fcns.trust_region.evaluate_step(self)

        for constraint in self.constraints.values():
            constraint.update_current_iter_data(self)

        self.current_iter_data.iter_num += 1

        self.current_iter_data.update(self.penalty_snapshot())

        self._convergence_check_fcn(self)

        self.update_hyperparams()

    def record_iter_data(self) -> None:
        self.iter_data_list.append(copy.deepcopy(self.current_iter_data))

    def update_hyperparams(self, alpha: float = 1.0) -> None:
        """Post-step autotuning: trust region first (if configured), then constraint penalties."""
        self.update_trust_region()
        self.update_constraint_penalties(alpha)

    def update_trust_region(self) -> None:
        autotune = self.fcns.trust_region.get('autotune')
        if autotune is not None:
            autotune(self)

    def update_constraint_penalties(self, alpha: float = 1.0) -> None:
        for constraint in self.constraints.values():
            constraint.update_constraint_penalties(self, alpha)

    def troubleshoot_step(self) -> tuple[bool, str | None]:
        if self._troubleshoot_on_step_fcn is None:
            return True, None
        return self._troubleshoot_on_step_fcn([self])

    def troubleshoot(self, exc=None) -> str:
        if self._troubleshoot_on_error_fcn is None:
            raise (exc if exc is not None else RuntimeError("solve failed with no troubleshooting configured"))
        return self._troubleshoot_on_error_fcn([self], exc)
