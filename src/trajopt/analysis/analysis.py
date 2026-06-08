import copy

import jax
import numpy as np

from trajopt.methods.scp import integrators, pseudospectral
from trajopt.utils import tools
from trajopt.utils.tools import AttrDict, recursive_attrdict

jax.config.update("jax_enable_x64", True)

"""
outline of solution_data structure:
results = {
    "trajectories": 
        "segments":
            [{"iter_data_list": [...]}, {"iter_data_list": [...]}, ...]
}
"""

def perform_analysis(traj):
    """Propagate every segment's iterates and merge them into one mission trajectory."""
    segments = [analyze_segment(subprob, traj.config) for subprob in traj.method.scp_trajectory.scp_segments.values()]

    if len(segments) == 1:
        iter_data_list = segments[0]
    else:
        n_iters        = min(len(seg) for seg in segments)
        iter_data_list = [concat([seg[i] for seg in segments]) for i in range(n_iters)]

    return AttrDict({"iter_data_list": iter_data_list})


def analyze_segment(subprob, config):
    """Propagate each iterate's nonlinear trajectory and evaluate the segment's trajplots."""
    segment    = subprob.segment
    params     = segment.params
    nondim     = segment.nondim
    idx        = segment.index_map.indices
    time_scale = nondim.time_scale

    iters    = subprob.iter_data_list if config.analysis.compute_iters else [subprob.iter_data_list[-1]]
    dynamics = next(c for c in segment.constraints.values() if c.type == "dynamics").fcn_znu
    solver   = integrators.make_node_propagation_solver(dynamics, params, n_steps=50)

    z_init = subprob.initial_guess.z_dense
    nu_init = subprob.initial_guess.nu_dense

    analyzed = []
    for iter_data in iters:
        z_opt  = np.asarray(iter_data.z_opt)
        nu_opt = np.asarray(iter_data.nu_opt)

        N = z_opt.shape[0]
        if subprob.flags.get("discretize", "ms") == "ps":
            _, etau, _, _ = pseudospectral.flipped_radau_differential_operator(N - 1)
            tau_nodes = (etau + 1.0) / 2.0
        else:
            tau_nodes = np.linspace(0.0, 1.0, N)

        _, z_nl, nu_nl = integrators.propagate_from_nodes(
            z_opt, tau_nodes, nu_opt, dynamics, params, _solver=solver,
        )

        # general trajplot values (not necessarily constraints as specified in the problem)
        trajplot_data = AttrDict({})
        for trajplot in segment.trajplots.values():
            if not hasattr(trajplot, "compute_trajplot_values"):
                continue

            output = AttrDict({
                "name":         trajplot.name,
                "type":         trajplot.type,
                "opt_vals":     trajplot.compute_trajplot_values(z_opt,  nu_opt,  params),
                "nl_vals":      trajplot.compute_trajplot_values(z_nl,   nu_nl,   params),
                "init_nl_vals": trajplot.compute_trajplot_values(z_init, nu_init, params),
                "title":        getattr(trajplot, "title", None),
                "xlabel":       getattr(trajplot, "xlabel", None),
                "ylabel":       getattr(trajplot, "ylabel", None),
                "zlabel":       getattr(trajplot, "zlabel", None),
                "tick_nbins":   getattr(trajplot, "tick_nbins", None),
                "markers":      getattr(trajplot, "markers", None),
                "invert_x":     getattr(trajplot, "invert_x", False),
                "show_iters":   getattr(trajplot, "show_iters", None),
            })

            trajplot_data.setdefault(trajplot.group, AttrDict({}))[trajplot.name] = output

        # re-dimensionalize and store the data that the plots consume
        analyzed.append(AttrDict({
            "t_opt":         z_opt[:, idx.z.time].squeeze(-1) * time_scale,
            "z_opt":         z_opt[:, idx.z.state] @ nondim.M.state.nd2d,
            "nu_opt":        nu_opt[:, idx.nu.control] @ nondim.M.control.nd2d,
            "t_nl":          z_nl[:, idx.z.time].squeeze(-1) * time_scale,
            "z_nl":          z_nl[:, idx.z.state] @ nondim.M.state.nd2d,
            "nu_nl":         nu_nl[:, idx.nu.control] @ nondim.M.control.nd2d,
            "t_init_nl":     z_init[:, idx.z.time].squeeze(-1) * time_scale,
            "z_init_nl":     z_init[:, idx.z.state] @ nondim.M.state.nd2d,
            "nu_init_nl":    nu_init[:, idx.nu.control] @ nondim.M.control.nd2d,
            "trajplot_data": trajplot_data,
        }))

    return analyzed


def concat(items):
    """Concatenate matching analysis payloads node-wise across segments.

    Arrays that cannot be stacked (e.g. states of segments with different
    dimensions) keep the first segment's values, since only the time grids
    and trajplot values are plotted across the whole mission.
    """
    head = items[0]

    if isinstance(head, np.ndarray):
        if all(isinstance(x, np.ndarray) and x.shape[1:] == head.shape[1:] for x in items):
            return np.concatenate(items, axis=0)
        return head

    if isinstance(head, dict):
        return AttrDict({k: concat([x[k] for x in items]) for k in head if all(k in x for x in items)})

    if isinstance(head, list):
        if all(isinstance(x, list) and len(x) == len(head) for x in items):
            return [concat([x[i] for x in items]) for i in range(len(head))]
        return head

    return head


def run_standalone_analysis(traj):
    """Analyze a single solve and wrap it in the {method: {runs: [...]}} schema."""
    method_name = traj.config.method.get("name", "method1")
    return recursive_attrdict({method_name: {"runs": [perform_analysis(traj)]}})


def run_mc_analysis(traj):
    """Monte Carlo analysis driven entirely by the config (the single source of truth).

    Each run perturbs the values listed under ``config.variations.samples`` (each key
    is a dot-path into the config), updates the config, and re-solves. Run 0 is the
    nominal (unperturbed) case.

    Expected config schema::

        variations:
          seed: 42
          num:  10
          samples:
            segments.entry.params.vehicle.bc: {type: normal, mu: 0.0, sigma: 10.0}
            segments.entry.constraints.initial_state.value: {type: uniform, lb: [...], ub: [...]}
    """
    nominal_config = traj.config
    var_cfg        = nominal_config.variations
    method_name    = nominal_config.method.get("name", "method1")
    num            = int(var_cfg.get("num", 0))

    np.random.seed(var_cfg.get("seed", 0))

    runs = []
    for i in range(num + 1):
        config = copy.deepcopy(nominal_config)
        for path, spec in (var_cfg.get("samples", {}) if i > 0 else {}).items():
            if spec["type"] == "uniform":
                delta = np.random.uniform(spec["lb"], spec["ub"])
            elif spec["type"] == "normal":
                delta = np.random.normal(spec["mu"], spec["sigma"])
            else:
                raise ValueError(f"unknown variation type: {spec['type']!r}")
            tools.set_from_path(config, path, tools.get_from_path(config, path) + delta)

        traj.config = config
        traj.solve()
        runs.append(perform_analysis(traj))
        if i > 0:
            print(f"=== {method_name} | run {i} / {num} ===")

    traj.config = nominal_config
    return recursive_attrdict({method_name: {"runs": runs}})
