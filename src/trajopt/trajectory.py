from trajopt.segment import Segment
from trajopt.utils.tools import AttrDict


class Trajectory:
    def __init__(self, trajectory_config):

        self.config = trajectory_config

        self.segments: AttrDict[str, Segment] = AttrDict()
        for name, segment_config in self.config.segments.items():
            self.segments[name] = Segment(name, segment_config)
