import copy

import jax
import numpy as np

from trajopt.analysis.results import Iterate, MissionResult, RunResult
from trajopt.utils import tools
from trajopt.utils.tools import AttrDict

jax.config.update("jax_enable_x64", True)

def perform_analysis(traj) -> RunResult:
    """Propagate every phase's iterates and merge them into one mission trajectory."""
    scp_trajectory = traj.method.scp_trajectory
    scp_phases = getattr(scp_trajectory, "scp_phases", None)
    if scp_phases is None:
        scp_phases = scp_trajectory.scp_subproblems
    phases = [analyze_phase(subprob, traj.config) for subprob in scp_phases.values()]

    if len(phases) == 1:
        iter_mappings = phases[0]
    else:
        n_iters       = min(len(seg) for seg in phases)
        iter_mappings = []
        for i in range(n_iters):
            per_phase = [seg[i] for seg in phases]
            pad_missing_outputs(per_phase)
            iter_mappings.append(concat(per_phase))

    solver_iters = {name: subprob.iter_data_list for name, subprob in scp_phases.items()}

    iter_data_list = [Iterate.from_mapping(m) for m in iter_mappings]
    return RunResult(iter_data_list=iter_data_list, solver_iters=solver_iters)


def analyze_phase(subprob, config):
    """Propagate each iterate and evaluate the phase's outputs."""
    phase    = subprob.phase
    params     = phase.params
    nondim     = phase.nondim
    idx        = phase.index_map.indices
    time_scale = nondim.time_scale

    iters    = subprob.iter_data_list if config.analysis.compute_iters else [subprob.iter_data_list[-1]]
    dynamics = next(c for c in phase.constraints.values() if c.type == "dynamics").fcn_znu

    discretize = subprob.hyperparams.discretize.get("mode", "ms")
    propagate_from_nodes_flag = config.analysis.get("propagate_from_nodes", False)

    H = int(getattr(subprob.hyperparams.discretize, 'hp_segments', 1))

    if propagate_from_nodes_flag:
        node_solver = subprob.fcns.discretize.make_node_propagation_solver(dynamics, params, n_steps=50)
    else:
        traj_solver = subprob.fcns.discretize.make_trajectory_solver(
            dynamics, params, n_steps=500, discretize=discretize, hp_segments=H,
        )

    z_init = subprob.initial_guess.z_dense
    nu_init = subprob.initial_guess.nu_dense

    analyzed = []
    for iter_data in iters:
        z_opt  = np.asarray(iter_data.z_opt)
        nu_opt = np.asarray(iter_data.nu_opt)

        N = z_opt.shape[0]
        tau_nodes = subprob.mesh.tau

        if propagate_from_nodes_flag:
            _, z_nl, nu_nl = subprob.fcns.discretize.propagate_from_nodes(
                z_opt, tau_nodes, nu_opt, dynamics, params, _solver=node_solver,
            )
        else:
            _, z_nl, nu_nl = subprob.fcns.discretize.propagate_trajectory(
                z_opt, tau_nodes, nu_opt, dynamics, params, _solver=traj_solver,
            )

        # an output from the config replaces the one built here with the same name
        outputs = auto_outputs(
            phase,
            opt        = (z_opt,  nu_opt),
            nl_prop    = (z_nl,   nu_nl),
            init_guess = (z_init, nu_init),
        )

        # outputs the config declares
        for output in phase.outputs.values():
            if not hasattr(output, "compute_values"):
                continue

            opt        = output.compute_values(z_opt,  nu_opt,  params)
            nl_prop    = output.compute_values(z_nl,   nu_nl,   params)
            init_guess = output.compute_values(z_init, nu_init, params)

            # limits come from the propagated values, quivers from the nodes
            outputs[output.name] = AttrDict({
                "opt":        opt["values"],
                "nl_prop":    nl_prop["values"],
                "init_guess": init_guess["values"],
                "limits":     nl_prop["limits"],
                "quivers":    opt["quivers"],
                "meta": AttrDict({
                    "name":         output.name,
                    "type":         output.type,
                    "group":        output.group,
                    "title":        getattr(output, "title", None),
                    "xlabel":       getattr(output, "xlabel", None),
                    "ylabel":       getattr(output, "ylabel", None),
                    "zlabel":       getattr(output, "zlabel", None),
                    "tick_nbins":   getattr(output, "tick_nbins", None),
                    "markers":      getattr(output, "markers", None),
                    "invert_x":     getattr(output, "invert_x", False),
                    "show_iters":   getattr(output, "show_iters", None),
                    "trigger_line": getattr(output, "trigger_line", None),
                    "units":        getattr(output, "units", None),
                }),
            })

        # re-dimensionalize and store the data that the plots consume
        analyzed.append(AttrDict({
            "iter_num":      int(iter_data.iter_num),
            "t_opt":         z_opt[:, idx.z.time].squeeze(-1) * time_scale,
            "x_opt":         z_opt[:, idx.z.state] @ nondim.M.state.nd2d,
            "u_opt":        nu_opt[:, idx.nu.control] @ nondim.M.control.nd2d,
            "t_nl":          z_nl[:, idx.z.time].squeeze(-1) * time_scale,
            "x_nl":          z_nl[:, idx.z.state] @ nondim.M.state.nd2d,
            "u_nl":         nu_nl[:, idx.nu.control] @ nondim.M.control.nd2d,
            "t_init_nl":     z_init[:, idx.z.time].squeeze(-1) * time_scale,
            "x_init_nl":     z_init[:, idx.z.state] @ nondim.M.state.nd2d,
            "u_init_nl":    nu_init[:, idx.nu.control] @ nondim.M.control.nd2d,
            "outputs":       outputs,
        }))

    return analyzed


def auto_outputs(phase, **representations):
    """Build one SI-unit output per state and control component, plus time and the augmented states."""
    index_map = phase.index_map
    nondim    = phase.nondim
    idx       = index_map.indices

    sliced = {}
    for rep, (z, nu) in representations.items():
        x = z[:, idx.z.state] @ nondim.M.state.nd2d
        u = nu[:, idx.nu.control] @ nondim.M.control.nd2d
        sliced[rep] = {
            **{f"state:{n}":   x[:, i]  for n, i in index_map.components.state.items()},
            **{f"control:{n}": u[:, i]  for n, i in index_map.components.control.items()},
            "time":            z[:, idx.z.time] * nondim.time_scale,
            "dilation_factor": nu[:, idx.nu.dilation_factor],
            "ctcs":            z[:, idx.z.ctcs],
            "running_cost":    z[:, idx.z.running_cost],
        }

    outputs = AttrDict({})
    for key in next(iter(sliced.values())):
        name = key.split(":", 1)[1] if ":" in key else key
        if any(sliced[rep][key].shape[1] == 0 for rep in sliced):
            continue  # augmented block this phase does not have
        outputs[name] = AttrDict({
            **{rep: sliced[rep][key] for rep in sliced},
            "limits":  None,
            "quivers": [],
            "meta": AttrDict({
                "name": name, "type": "time_series", "group": None,
                "title": None, "xlabel": None, "ylabel": None, "zlabel": None,
                "tick_nbins": None, "markers": None, "invert_x": False,
                "show_iters": None, "trigger_line": None, "units": None,
            }),
        })
    return outputs


#: each value array and the time grid whose length it matches
_REPRESENTATIONS = (("opt", "t_opt"), ("nl_prop", "t_nl"), ("init_guess", "t_init_nl"))


def pad_missing_outputs(per_phase):
    """Fill the outputs a phase is missing with NaN, in place."""
    template = {}
    for phase_data in per_phase:
        for name, output in phase_data.outputs.items():
            template.setdefault(name, output)

    for phase_data in per_phase:
        for name, output in template.items():
            if name in phase_data.outputs:
                continue
            phase_data.outputs[name] = AttrDict({
                key: np.full((len(phase_data[t_key]), output[key].shape[1]), np.nan)
                for key, t_key in _REPRESENTATIONS
            } | {
                "limits":  _nan_like(output.limits, len(phase_data.t_nl)),
                "quivers": [
                    {**q, "dirs": np.full_like(q["dirs"], np.nan),
                     "origins": None if q.get("origins") is None else np.full_like(q["origins"], np.nan)}
                    for q in (output.quivers or [])
                ],
                "meta": output.meta,
            })


def _nan_like(limits, n):
    """Replace any per-node limit array with NaN, keeping scalar limits as they are."""
    if not limits:
        return limits
    return {
        k: (np.full(n, np.nan) if isinstance(v, np.ndarray) else v)
        for k, v in limits.items()
    }


def concat(items):
    """Join each phase's values end to end. Arrays that cannot be stacked keep the first phase's."""
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


def run_standalone_analysis(traj) -> MissionResult:
    """Analyze a single solve and wrap it in the MissionResult schema."""
    method_name = traj.config.method.get("name", "method1")
    if not getattr(traj, "_solved", False):
        traj.solve()
    return MissionResult(runs_by_method={method_name: [perform_analysis(traj)]})


def run_method_variation(traj):
    """Compare different methods on the same mission.

    Expected config schema::

        analysis:
          type: method_variation
          methods:
            autoscvx-ms: {}
            autoscvx-ps-h10:
              hyperparams:
                discretize:
                  mode: ps
                  equal_dt: 0
                  hp_segments: 10
                trust_region:
                  line_search: 1
                penalties:
                  initial_state:
                    vb: 'none'
    """
    from trajopt.trajectory_analyzer import TrajectoryAnalyzer

    methods_cfg = traj.config.analysis.methods
    config_path = traj.config_path

    runs_by_method = {}
    for method_name, overrides in methods_cfg.items():
        print(f"\n{'='*60}")
        print(f"  Solving: {method_name}")
        print(f"{'='*60}\n")

        method_traj = TrajectoryAnalyzer(config_path, method_overrides=dict(overrides) if overrides else None)
        method_traj.solve()
        runs_by_method[method_name] = [perform_analysis(method_traj)]

    return MissionResult(runs_by_method=runs_by_method)


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
            phases.entry.params.vehicle.bc: {type: normal, mu: 0.0, sigma: 10.0}
            phases.entry.constraints.initial_state.value: {type: uniform, lb: [...], ub: [...]}
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
    return MissionResult(runs_by_method={method_name: runs})
