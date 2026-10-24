import jax
import numpy as np

from collections.abc import Mapping
from trajopt.utils.tools import AttrDict

jax.config.update("jax_enable_x64", True)

def analyze_segment(subprob, config):
    """Propagate each iterate and evaluate the phase's outputs."""
    phase    = subprob.problem
    params     = phase.params
    nondim     = phase.nondim
    idx        = phase.index_map.indices
    time_scale = nondim.time_scale

    analysis_config = config.get("analysis", {})
    if not subprob.iter_data_list:
        return []
    iters = subprob.iter_data_list if analysis_config.get("compute_iters", False) else [subprob.iter_data_list[-1]]
    dynamics = next(c for c in phase.constraints.values() if c.type == "dynamics").fcn_znu

    discretize = subprob.mesh.discretize
    propagate_from_nodes_flag = analysis_config.get("propagate_from_nodes", False)

    # ps only: controls between collocation nodes are reconstructed with "lagrange" or "foh"
    # ms always uses foh either way
    mesh_opts = AttrDict({
        "discretize":            discretize,
        "segments":              subprob.mesh.segments,
        "control_interpolation": analysis_config.get("control_interpolation", "lagrange"),
        "dilation_idx":          idx.nu.dilation_factor,
    })

    if propagate_from_nodes_flag:
        node_solver = subprob.fcns.discretize.make_node_propagation_solver(
            dynamics, params, n_steps=int(analysis_config.get("node_steps", 50)), **mesh_opts)
    else:
        traj_solver = subprob.fcns.discretize.make_trajectory_solver(
            dynamics, params, n_steps=int(analysis_config.get("trajectory_steps", 500)), **mesh_opts)

    z_init = subprob.initial_guess.z_dense
    nu_init = subprob.initial_guess.nu_dense

    analyzed = []
    for iter_data in iters:
        z_opt  = np.asarray(iter_data.z_opt)
        nu_opt = np.asarray(iter_data.nu_opt)

        tau_nodes = subprob.mesh.tau

        if propagate_from_nodes_flag:
            _, z_nl, nu_nl = subprob.fcns.discretize.propagate_from_nodes(
                z_opt, tau_nodes, nu_opt, dynamics, params, _solver=node_solver,
                discretize=mesh_opts.discretize, segments=mesh_opts.segments,
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


def concat(items):
    """Join compatible sequential segment values without discarding arrays."""
    if not items:
        raise ValueError("Select at least one segment value to concatenate")
    head = items[0]

    if isinstance(head, np.ndarray):
        if all(isinstance(x, np.ndarray) and x.shape[1:] == head.shape[1:] for x in items):
            return np.concatenate(items, axis=0)
        raise ValueError("Cannot merge segment arrays with different channel dimensions")

    if isinstance(head, Mapping):
        if any(not isinstance(x, Mapping) or x.keys() != head.keys() for x in items):
            raise ValueError("Segment values have different field identities")
        return AttrDict({k: concat([x[k] for x in items]) for k in head})

    if isinstance(head, (list, tuple)):
        if all(isinstance(x, (list, tuple)) and len(x) == len(head) for x in items):
            return [concat([x[i] for x in items]) for i in range(len(head))]
        raise ValueError("Segment values have different sequence lengths")

    if any(x != head for x in items):
        raise ValueError("Segment values have different non-array metadata")
    return head
