import trajopt.utils.config_loader as config_loader
import trajopt.analysis.analysis as analysis
import trajopt.analysis.plotting as plotting
from trajopt.segment import Segment
from trajopt.utils.tools import AttrDict
import trajopt.link_constraints.link_constraint_types as link_constraint_type_module
import trajopt.link_costs.link_cost_types as link_cost_type_module
from trajopt.methods.scp.scp_method import SCPMethod

class Trajectory:
    def __init__(self, config_path):

        self.config = config_loader.load_trajopt_config(config_path)

        self.segments: AttrDict[str, Segment] = AttrDict()
        for name, segment_config in self.config.segments.items():
            self.segments[name] = Segment(name, segment_config)

        self.links = AttrDict({"constraints": AttrDict(), "costs": AttrDict()})

        links_config = self.config.get("links", AttrDict())

        for link_constraint_name, link_constraint_config in links_config.get("constraints", AttrDict()).items():
            link_constraint_config.name = link_constraint_name
            link_constraint_type = link_constraint_config.type
            segment1 = self.segments[link_constraint_config.segment1]
            segment2 = self.segments[link_constraint_config.segment2]

            linkConstraintClass = getattr(link_constraint_type_module, link_constraint_type)
            self.links.constraints[link_constraint_name] = linkConstraintClass(link_constraint_config, segment1, segment2)

        for link_cost_name, link_cost_config in links_config.get("costs", AttrDict()).items():
            link_cost_config.name = link_cost_name
            link_cost_type = link_cost_config.type
            segment1 = self.segments[link_cost_config.segment1]
            segment2 = self.segments[link_cost_config.segment2]

            linkCostClass = getattr(link_cost_type_module, link_cost_type)
            self.links.costs[link_cost_name] = linkCostClass(link_cost_config, segment1, segment2)

        method_config = self.config.method
        self.method = SCPMethod(method_config, self.segments, self.links)

    def solve(self):
        self.method.solve()

    def analyze(self):
        analysis_cfg = self.config.get("analysis", {})
        analysis_type = analysis_cfg.get("type", "standalone")
        self.analysis_type = analysis_type

        if analysis_type == "standalone":
            self.results = analysis.run_standalone_analysis(self)

        elif analysis_type == "mc":
            self.results = analysis.run_mc_analysis(self)

        return self.results

    def plot(self, data):
        plotting.plot(self, data)
