from trajopt.methods.common import initial_guess
from trajopt.methods.dev.scvx.scp_segment import SCPSegment
from trajopt.utils.tools import AttrDict


class SCPTrajectory():
    def __init__(self, trajectory, method_config) -> None:

        # create dictionary of scp-specific segment types
        self.scp_segments = AttrDict()
        previous = None
        for name, segment in trajectory.segments.items():
            print("=" * 60)
            print(f"segment: {name}:")
            print("=" * 60)

            # a written-out x_start is an array, so compare the type first
            x_start = getattr(segment.guess, "x_start", None)
            if isinstance(x_start, str) and x_start == initial_guess.CHAIN_FROM_PREVIOUS:
                if previous is None:
                    raise ValueError(
                        f"segment '{name}' sets guess.x_start to "
                        f"'{initial_guess.CHAIN_FROM_PREVIOUS}' but is the first segment"
                    )
                segment.guess.x_start = initial_guess.guess_endpoint(previous).tolist()

            self.scp_segments[name] = SCPSegment(segment, method_config)
            previous = self.scp_segments[name]

        # build inter-segment constraints
        for seg in self.scp_segments.values():
            for cnstr in seg.constraints.values():
                cnstr.build_cross_segment(self.scp_segments)
