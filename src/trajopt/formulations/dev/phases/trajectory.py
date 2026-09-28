from trajopt.formulations.dev.phases.phase import Phase
from trajopt.utils.tools import AttrDict


class Trajectory:
    def __init__(self, trajectory_config):

        self.config = trajectory_config

        self.phases: AttrDict[str, Phase] = AttrDict()
        for name, phase_config in self.config.phases.items():
            self.phases[name] = Phase(name, phase_config)