from trajopt.methods.scp.scp_segment import SCPSegment
from trajopt.utils.tools import AttrDict


class SCPTrajectory():
    def __init__(self, trajectory, method_config) -> None:
        
        # create dictionary of scp-specific segment types
        self.scp_segments = AttrDict()
        for name, segment in trajectory.segments.items():
            print("=" * 60)
            print(f"segment: {name}:")
            print("=" * 60)
            self.scp_segments[name] = SCPSegment(segment, method_config)

        # build inter-segment constraints
        for seg in self.scp_segments.values():
            for cnstr in seg.constraints.values():
                cnstr.build_cross_segment(self.scp_segments)