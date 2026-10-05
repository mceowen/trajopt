from trajopt.methods.common import initial_guess
from trajopt.methods.common.scp.phase import SCPPhase
from trajopt.utils.tools import AttrDict


class SCPTrajectory():
    def __init__(self, trajectory, method_config) -> None:

        # create dictionary of scp-specific phase types
        self.scp_phases = AttrDict()
        previous = None
        for name, phase in trajectory.phases.items():
            print("=" * 60)
            print(f"phase: {name}:")
            print("=" * 60)

            # no subproblem exists yet, so this can only see literal guess data, not fcns.<name>
            x_start = getattr(phase.guess, "x_start", None)
            if x_start is None:
                x_start = getattr(initial_guess._method_guess(method_config, name), "x_start", None)
            if isinstance(x_start, str) and x_start == initial_guess.CHAIN_FROM_PREVIOUS:
                if previous is None:
                    raise ValueError(
                        f"phase '{name}' sets guess.x_start to "
                        f"'{initial_guess.CHAIN_FROM_PREVIOUS}' but is the first phase"
                    )
                phase.guess.x_start = initial_guess.guess_endpoint(previous).tolist()

            self.scp_phases[name] = SCPPhase(phase, method_config)
            previous = self.scp_phases[name]

        # build inter-phase constraints
        for seg in self.scp_phases.values():
            for cnstr in seg.constraints.values():
                cnstr.build_cross_phase(self.scp_phases)

    def troubleshoot_step(self) -> tuple[bool, str | None]:
        on_step = next(iter(self.scp_phases.values()))._troubleshoot_on_step_fcn
        if on_step is None:
            return True, None
        return on_step(self.scp_phases.values())

    def troubleshoot(self, exc=None) -> str:
        on_error = next(iter(self.scp_phases.values()))._troubleshoot_on_error_fcn
        if on_error is None:
            raise (exc if exc is not None else RuntimeError("solve failed with no troubleshooting configured"))
        return on_error(self.scp_phases.values(), exc)
