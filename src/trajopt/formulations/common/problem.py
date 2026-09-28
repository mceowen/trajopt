from trajopt.formulations.common.index_map import IndexMap
from trajopt.nondim import Nondim
import trajopt.formulations.common.constraint_types as constraint_type_module
import trajopt.formulations.common.cost_types as cost_type_module
import trajopt.outputs.output_types as output_type_module
from trajopt.utils.tools import AttrDict, resolve_function_from_string


class Problem:
    """One single-phase OCP building block: a set of constraints/costs over one node grid.

    This is what a plain trajectory *is* (dev.simple.trajectory.Trajectory subclasses it
    directly), and what each phase of a multi-phase trajectory *is* (dev.phases.phase.Phase
    subclasses it, one instance per phase). A wrapper that chains several of these together
    -- dev.phases.trajectory.Trajectory today, a future dev.agents wrapper over multiple
    vehicles -- holds a collection of Problems rather than reimplementing this setup.
    """

    #: console label used when announcing this problem's configuration; subclasses override
    _KIND = "problem"

    def __init__(self, problem_config: AttrDict, name: str | None = None) -> None:

        self.name      = name if name is not None else problem_config.get("name", "main")
        self.num_nodes = problem_config.num_nodes

        # set index map for augmentation and nondim object for ocp
        self.index_map = IndexMap(problem_config)
        self.nondim    = Nondim(problem_config, self.index_map)

        # load functions
        self.fcns = AttrDict()
        for fcn_name, path in problem_config.fcns.items():
            self.fcns[fcn_name] = resolve_function_from_string(path)

        # load parameters
        self.params = problem_config.params

        # load guess config
        self.guess = problem_config.get("guess", AttrDict())

        print(f"{self._KIND} '{self.name}' configuration:")
        print("------------------------------------------------------------")

        # create dictionary of constraints
        self.constraints = AttrDict()
        for cnstr_name, cnstr_config in problem_config.constraints.items():
            cnstr_config.name = cnstr_name
            cnstr_type = cnstr_config.type
            constraintClass = getattr(constraint_type_module, cnstr_type)
            cnstr = constraintClass(cnstr_config, self)
            if "eps" in cnstr_config:
                cnstr.eps = cnstr_config.eps
            self.constraints[cnstr_name] = cnstr

        self._wire_ctcs_constraints()

        # create dictionary of costs
        self.costs = AttrDict()
        for cost_name, cost_config in problem_config.get("costs", AttrDict()).items():
            cost_config.name = cost_name
            cost_type = cost_config.type
            costClass = getattr(cost_type_module, cost_type)
            self.costs[cost_name] = costClass(cost_config, self)

        self._wire_running_costs()

        # create dictionary of outputs
        self.outputs = AttrDict()
        for output_name, output_config in problem_config.get("outputs", AttrDict()).items():
            output_config.name = output_name
            output_type = output_config.type
            outputClass = getattr(output_type_module, output_type)
            self.outputs[output_name] = outputClass(output_config, self)

        print("------------------------------------------------------------")
        print("\n")

    def _wire_ctcs_constraints(self):
        """Attach any ctcs_nonconvex_inequality constraints to the dynamics and resize the augmented state."""
        ctcs = [c for c in self.constraints.values() if c.type == "ctcs_nonconvex_inequality"]
        if not ctcs:
            return
        dyn = next((c for c in self.constraints.values() if c.type == "dynamics"), None)
        if dyn is None:
            return
        dyn.ctcs_constraints = tuple(ctcs)
        n_ctcs = sum(c.dimension for c in ctcs)
        self.index_map.set_augmented_dims(n_ctcs=n_ctcs)
        dyn.dimension = self.index_map.n.z

    def _wire_running_costs(self):
        """Attach any nonconvex_running costs to the dynamics and resize the augmented state."""
        rc = [c for c in self.costs.values() if c.type == "nonconvex_running"]
        if not rc:
            return
        dyn = next((c for c in self.constraints.values() if c.type == "dynamics"), None)
        if dyn is None:
            return
        dyn.running_costs = tuple(rc)
        for i, cost in enumerate(rc):
            cost.gamma_idx = i
        n_ctcs = self.index_map.n.ctcs
        self.index_map.set_augmented_dims(n_ctcs=n_ctcs, n_running_cost=len(rc))
        dyn.dimension = self.index_map.n.z
