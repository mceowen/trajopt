import numpy as np
from trajopt.core.constraints.constraints import Constraints
from trajopt.core.costs.costs import Costs
from trajopt.core.trajectories.trajectories import Trajectories
from trajopt.utils.config_loader import resolve_function_from_path
from trajopt.utils.tools import AttrDict, recursive_attrdict

class Problem:

    def __init__(self, config, index_map=None):

        # ------------------------------------------------------------
        # Config
        # ------------------------------------------------------------

        self.config     = config
        self.index_map  = index_map

        # ------------------------------------------------------------
        # Functions
        # ------------------------------------------------------------

        fcn_config      = config.problem.fcns
        self.fcns       = AttrDict()
        
        for name, path in fcn_config.items():
            self.fcns[name] = resolve_function_from_path(path)

        # ------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------

        self.params = recursive_attrdict(config.problem.params)

        # ------------------------------------------------------------
        # Phases (multiphase support)
        # ------------------------------------------------------------

        self.phases = self._resolve_phases(config, index_map)

        print("problem configuration: ")
        print("------------------------------------------------------------")
        # ------------------------------------------------------------
        # Constraints
        # ------------------------------------------------------------

        self.constraints = Constraints(self.config, index_map, fcns=self.fcns)

        # ------------------------------------------------------------
        # Cost
        # ------------------------------------------------------------

        self.costs = Costs(self.config, index_map, fcns=self.fcns)
        print("------------------------------------------------------------")

        # ------------------------------------------------------------
        # Bind fcns dict to constraint/cost functions that accept it
        # ------------------------------------------------------------

        self.constraints.resolve_functions(self.fcns)
        self.costs.resolve_functions(self.fcns)

        # ------------------------------------------------------------
        # Trajectories (similar to constraints but used for analysis)
        # ------------------------------------------------------------
        self.trajectories = Trajectories(self.config, index_map, fcns=self.fcns)
        self.trajectories.resolve_functions(self.fcns)

    def _resolve_phases(self, config, index_map):
        phases_config = config.problem.mission.get('phases', None)
        if phases_config is None:
            return None

        N = index_map.N.time_grid
        sorted_phases = sorted(phases_config.items(), key=lambda x: x[1]['start'])

        phases = []
        phase_schedule = {}

        for i, (name, cfg) in enumerate(sorted_phases):
            start = cfg['start']
            end = sorted_phases[i + 1][1]['start'] if i + 1 < len(sorted_phases) else N
            phase_params = {k: v for k, v in cfg.items() if k not in ('start',)}

            for key, val in phase_params.items():
                if key not in phase_schedule:
                    phase_schedule[key] = np.zeros(N)
                phase_schedule[key][start:end] = val

            if i == 0:
                for key, val in phase_params.items():
                    self.params[key] = val

            phases.append(AttrDict({
                'name': name,
                'start': start,
                'end': end,
                'params': phase_params
            }))

        self.params['_phase_schedule'] = phase_schedule

        print("phases:")
        for phase in phases:
            print(f"  {phase.name}: nodes [{phase.start}..{phase.end - 1}], params: {dict(phase.params)}")

        return phases