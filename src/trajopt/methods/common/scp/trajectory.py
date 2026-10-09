from trajopt.methods.common import initial_guess
from trajopt.methods.common.scp.phase import SCPPhase
from trajopt.utils.tools import AttrDict


class SCPTrajectory():
    def __init__(self, trajectory, method_config, *, supplied_guess=None) -> None:

        # create dictionary of scp-specific phase types
        self.scp_phases = AttrDict()
        supplied = initial_guess.phase_guesses(supplied_guess, trajectory.phases)
        previous = None
        for name, phase in trajectory.phases.items():
            print("=" * 60)
            print(f"phase: {name}:")
            print("=" * 60)

            self.scp_phases[name] = SCPPhase(
                phase, method_config, initial_guess=supplied.get(name), previous_guess=previous)
            prepared = self.scp_phases[name].initial_guess
            previous = initial_guess.dimensional_samples(phase, prepared.z, prepared.nu)

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
