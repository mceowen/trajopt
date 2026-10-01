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

            # a written-out x_start is an array, so compare the type first
            x_start = getattr(phase.guess, "x_start", None)
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
