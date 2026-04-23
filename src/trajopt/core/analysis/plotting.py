import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from trajopt.core.analysis.trajplots import SCVXPLOTS
from trajopt.utils.tools import AttrDict, recursive_attrdict
import trajopt.utils.tools as tools

plt.rcParams["text.usetex"] = False
plt.rcParams.update({
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'lines.linewidth': 1.0,
})

plot_options = AttrDict({
    'figsize': (10, 2.8),
    'dpi': 300,
    'grid_gap_x': 0.06,
    'grid_gap_y': 0.18,
    'margins': [0.07, 0.02, 0.08, 0.12],
    'title_fontsize': 9,
    'title_pad': 6,
})

# pen: frgba, lrgba, lw, ls, msty, msz
pens = recursive_attrdict({
    'init':    {'frgba': [0,0,0,.1], 'lrgba': [0,0,0,1.],   'lw': 1, 'ls': '--', 'msty': '',  'msz': 3},
    'nl':      {'frgba': [0,0,0,.1], 'lrgba': [1,0,0,1.],   'lw': 2, 'ls': '-',  'msty': '',  'msz': 3},
    'opt':     {'frgba': [0,0,0,.1], 'lrgba': [0,0,1,1.],   'lw': 1, 'ls': '',   'msty': 'o', 'msz': 3},
    'itr_opt': {'frgba': [0,0,0,.1], 'lrgba': [.7,0,.3,.2], 'lw': 1, 'ls': '',   'msty': 'o', 'msz': 3},
    'itr_nl':  {'frgba': [0,0,0,.1], 'lrgba': [.7,0,.3,.4], 'lw': 1, 'ls': '-',  'msty': '',  'msz': 3},
    'wt_opt':  {'frgba': [0,0,0,.1], 'lrgba': [.7,0,.3,.8], 'lw': 1, 'ls': '-',  'msty': 'o', 'msz': 2},
    
    'opt1':    {'frgba': [0,0,0,.1], 'lrgba': [0,0,1,1.],   'lw': 1, 'ls': '',   'msty': 'o', 'msz': 2},
    'nl1':     {'frgba': [0,0,0,.1], 'lrgba': [0,0,1,1.],   'lw': 1, 'ls': '-',  'msty': '',  'msz': 3},
    'opt2':    {'frgba': [0,0,0,.1], 'lrgba': [1,0,1,1.],   'lw': 1, 'ls': '',   'msty': 'o', 'msz': 2},
    'nl2':     {'frgba': [0,0,0,.1], 'lrgba': [1,0,1,1.],   'lw': 1, 'ls': '-',  'msty': '',  'msz': 3},
    'opt3':    {'frgba': [0,0,0,.1], 'lrgba': [0,1,0,1.],   'lw': 1, 'ls': '',   'msty': 'o', 'msz': 2},
    'nl3':     {'frgba': [0,0,0,.1], 'lrgba': [0,1,0,1.],   'lw': 1, 'ls': '-',  'msty': '',  'msz': 3},
    'wt1':     {'frgba': [0,0,0,.1], 'lrgba': [0,0,1,1.],   'lw': 1, 'ls': '-',  'msty': 'o', 'msz': 2},
    'wt2':     {'frgba': [0,0,0,.1], 'lrgba': [1,0,1,1.],   'lw': 1, 'ls': '-',  'msty': 'o', 'msz': 2},
    'wt3':     {'frgba': [0,0,0,.1], 'lrgba': [0,1,0,1.],   'lw': 1, 'ls': '-',  'msty': 'o', 'msz': 2},
})

def _set_trajectory_limits(axs_trajectories, nominal_trajectory_data, nominal_iter_data, pad=0.05):
    t_opt = nominal_iter_data["t_opt"]
    t_nl  = nominal_iter_data["t_nl"]
    t_init_nl = nominal_iter_data.get("t_init_nl")
    for traj_group_name, traj_group_data in nominal_trajectory_data.items():
        for j, (traj_name, traj_data) in enumerate(traj_group_data.items()):
            ax = axs_trajectories[traj_group_name][j]
            ov = traj_data.opt_vals["values"]
            nv = traj_data.nl_vals["values"]
            parts = [ov, nv]
            iv = traj_data.get("init_nl_vals", {}).get("values") if traj_data.get("init_nl_vals") else None
            if iv is not None:
                parts.append(iv)
            if traj_data.type == "spatial":
                all_vals = np.concatenate(parts, axis=0)
                for dim_i, setter in enumerate([ax.set_xlim, ax.set_ylim] + ([ax.set_zlim] if hasattr(ax, 'set_zlim') and all_vals.shape[1] > 2 else [])):
                    lo, hi = all_vals[:, dim_i].min(), all_vals[:, dim_i].max()
                    margin = (hi - lo) * pad if hi > lo else 0.5
                    if dim_i == 0 and traj_data.get("invert_x", False):
                        setter(hi + margin, lo - margin)
                    else:
                        setter(lo - margin, hi + margin)
            elif traj_data.type == "time_series":
                t_parts = [t_opt, t_nl]
                y_parts = [ov.ravel(), nv.ravel()]
                if iv is not None and t_init_nl is not None:
                    t_parts.append(t_init_nl)
                    y_parts.append(iv.ravel())
                limits = traj_data.nl_vals.get("limits", {})
                if limits:
                    for val in [limits.get("upper"), limits.get("lower")]:
                        if val is not None:
                            if isinstance(val, np.ndarray):
                                y_parts.append(val.ravel())
                            else:
                                y_parts.append(np.array([val]))
                all_x = np.concatenate(t_parts)
                all_y = np.concatenate(y_parts)
                xlo, xhi = all_x.min(), all_x.max()
                ylo, yhi = all_y.min(), all_y.max()
                xm = (xhi - xlo) * pad if xhi > xlo else 0.5
                ym = (yhi - ylo) * pad if yhi > ylo else 0.5
                ax.set_xlim(xlo - xm, xhi + xm)
                ax.set_ylim(ylo - ym, yhi + ym)

def _mc_collect_batch(data, method_name, n_runs, paths_and_cols):
    all_segs = [[] for _ in paths_and_cols]
    nan = np.array([np.nan])
    for run in range(n_runs):
        iter_data = data[method_name]['runs'][run]['iters'][-1]
        arrays = []
        skip = False
        for path, col in paths_and_cols:
            arr = tools.get_from_path(iter_data, path)
            if arr.size == 0:
                skip = True
                break
            if arr.ndim > 1 and col is not None:
                arr = arr[:, col]
            elif arr.ndim > 1:
                arr = arr.ravel()
            arrays.append(arr)
        if skip:
            continue
        min_len = min(a.shape[0] for a in arrays)
        for k, arr in enumerate(arrays):
            all_segs[k].extend([arr[:min_len], nan])
    return [np.concatenate(s) if s else np.array([]) for s in all_segs]

def _mc_batch_plot(ax, data, method_name, n_runs, paths_and_cols, pen):
    arrays = _mc_collect_batch(data, method_name, n_runs, paths_and_cols)
    if arrays[0].size:
        ax.plot(*arrays, color=pen.lrgba[:3], alpha=pen.lrgba[3],
                linewidth=pen.lw, linestyle=pen.ls, marker=pen.msty, markersize=pen.msz)

def plot_default(trajopt_obj, data, analysis_type, show_iters=False, show_runs=None):
    PLTS = SCVXPLOTS(data)

    method_keys = list(data.keys())
    first_method = method_keys[0]

    nominal_iter_data = data[method_keys[0]]["runs"][-1]["iters"][-1]

    if show_iters:
        iters = slice(1, None)
    else:
        iters = [-1]

    problem = trajopt_obj.problem
    problem_config = problem.config.problem

    nominal_trajectory_data = nominal_iter_data["trajectory_data"]
    nominal_W_data    = nominal_iter_data["W"]
    nominal_dual_data = nominal_iter_data["dual"]

    pcfg = problem_config.get('plot_config', {})
    traj_group_pcfg = pcfg.get('trajectory_groups', {})

    figs_trajectories = AttrDict({})
    axs_trajectories  = AttrDict({})

    for trajectory_group_name, trajectory_group_data in nominal_trajectory_data.items():
        grp_cfg = traj_group_pcfg.get(trajectory_group_name, {})
        figs_trajectories[trajectory_group_name] = plt.figure(figsize=grp_cfg.get('figsize', plot_options.figsize), dpi=plot_options.dpi)

        plt_type = AttrDict({})

        for i, (trajectory_name, current_trajectory_data) in enumerate(trajectory_group_data.items()):
            if current_trajectory_data.type == "spatial":
                dim = current_trajectory_data.opt_vals["values"].shape[1]
            elif current_trajectory_data.type == "time_series":
                dim = 2
            
            plt_type[i] = "3D" if dim == 3 else "2D"

        num_trajectories_in_group = len(trajectory_group_data.keys())
        grid2_ins = {'plt_typs': plt_type}
        if 'pad_3d' in grp_cfg:
            grid2_ins['pad_3d'] = grp_cfg['pad_3d']
        axs_trajectories[trajectory_group_name] = PLTS.createGrid2(figs_trajectories[trajectory_group_name], grid=create_grid(num_trajectories_in_group, cfg=grp_cfg), ins=grid2_ins)

    # weights over time
    figs_weights_time = AttrDict({})
    axs_weights_time  = AttrDict({})
    t_steps = nominal_iter_data["t_opt"].shape[0]
    active_penalty_names_time = [key for key, val in nominal_W_data.items() if val.size > 0 and (nominal_W_data[key].shape[0] == t_steps or nominal_W_data[key].shape[0] == t_steps - 1)]
    active_dual_names_time   = [key for key, val in nominal_dual_data.items() if val.size > 0 and (nominal_dual_data[key].shape[0] == t_steps or nominal_dual_data[key].shape[0] == t_steps - 1)]
    weight_groups_time = AttrDict({})
    for name in active_penalty_names_time:
        if name not in weight_groups_time: weight_groups_time[name] = []
        weight_groups_time[name].append("W")
    for name in active_dual_names_time:
        if name not in weight_groups_time: weight_groups_time[name] = []
        weight_groups_time[name].append("dual")
    for wg_name, wg in weight_groups_time.items():
        figs_weights_time[wg_name] = plt.figure(figsize=plot_options.figsize, dpi=plot_options.dpi)
        axs_weights_time[wg_name]  = PLTS.createGrid(figs_weights_time[wg_name], grid=create_grid(len(wg)))

    # terminal weights (non-time-based)
    figs_weights_terminal = AttrDict({})
    axs_weights_terminal  = AttrDict({})
    active_penalty_names_terminal = [key for key, val in nominal_W_data.items() if val.size > 0 and key not in active_penalty_names_time]
    active_dual_names_terminal    = [key for key, val in nominal_dual_data.items() if val.size > 0 and key not in active_dual_names_time]
    weight_groups_terminal = AttrDict({})
    for name in active_penalty_names_terminal:
        if name not in weight_groups_terminal: weight_groups_terminal[name] = []
        weight_groups_terminal[name].append("W")
    for name in active_dual_names_terminal:
        if name not in weight_groups_terminal: weight_groups_terminal[name] = []
        weight_groups_terminal[name].append("dual")
    for wg_name, wg in weight_groups_terminal.items():
        figs_weights_terminal[wg_name] = plt.figure(figsize=plot_options.figsize, dpi=plot_options.dpi)
        axs_weights_terminal[wg_name]  = PLTS.createGrid(figs_weights_terminal[wg_name], grid=create_grid(len(wg)))

    # weights over iterations
    figs_weights_iters = AttrDict({})
    axs_weights_iters  = AttrDict({})
    active_penalty_names_iters = [key for key, val in nominal_W_data.items() if val.size > 0]
    active_dual_names_iters   = [key for key, val in nominal_dual_data.items() if val.size > 0]
    weight_groups_iters = AttrDict({})
    for name in active_penalty_names_iters:
        if name not in weight_groups_iters: weight_groups_iters[name] = []
        weight_groups_iters[name].append("W")
    for name in active_dual_names_iters:
        if name not in weight_groups_iters: weight_groups_iters[name] = []
        weight_groups_iters[name].append("dual")
    for wg_name, wg in weight_groups_iters.items():
        figs_weights_iters[wg_name] = plt.figure(figsize=plot_options.figsize, dpi=plot_options.dpi)
        axs_weights_iters[wg_name]  = PLTS.createGrid(figs_weights_iters[wg_name], grid=create_grid(len(wg)))

    if analysis_type == 'standalone':
        standalone_pens = AttrDict({"opt": pens.opt, "nl": pens.nl, "itr_opt": pens.itr_opt, "itr_nl": pens.itr_nl, "init": pens.init, "wt_opt": pens.wt_opt})
        plot_trajectories(PLTS, axs_trajectories, nominal_trajectory_data, method=first_method, run=0, iters=iters, pens=standalone_pens)
        plot_weights_time(PLTS, axs_weights_time, weight_groups_time, method=first_method, run=0, iters=iters, pens=standalone_pens)
        plot_weights_iters(PLTS, axs_weights_iters, weight_groups_iters, method=first_method, run=0, pens=standalone_pens)
        plot_weights_iters(PLTS, axs_weights_terminal, weight_groups_terminal, method=first_method, run=0, pens=standalone_pens)

    elif analysis_type == 'mc':
        for i, (method_name, method_data) in enumerate(data.items()):
            pen_nl  = pens[f"nl{i+1}"]
            pen_opt = pens[f"opt{i+1}"]
            n_runs = len(method_data['runs']) if show_runs is None else min(show_runs, len(method_data['runs']))

            for traj_group_name, traj_group_data in nominal_trajectory_data.items():
                for j, (traj_name, traj_data) in enumerate(traj_group_data.items()):
                    ax = axs_trajectories[traj_group_name][j]
                    if i == 0:
                        if traj_data.get("title"):
                            ax.set_title(traj_data["title"], fontsize=plot_options.title_fontsize, pad=plot_options.title_pad)
                        ax.grid(True, alpha=0.3)
                        if traj_data.get("xlabel"): ax.set_xlabel(traj_data["xlabel"])
                        if traj_data.get("ylabel"): ax.set_ylabel(traj_data["ylabel"])
                        if traj_data.get("zlabel") and hasattr(ax, 'set_zlabel'): ax.set_zlabel(traj_data["zlabel"])
                        if traj_data.get("tick_nbins"):
                            nbins = traj_data["tick_nbins"]
                            ax.xaxis.set_major_locator(MaxNLocator(nbins=nbins))
                            ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins))
                            if hasattr(ax, 'zaxis'):
                                ax.zaxis.set_major_locator(MaxNLocator(nbins=nbins))

                    nl_path  = f"trajectory_data.{traj_group_name}.{traj_name}.nl_vals.values"
                    opt_path = f"trajectory_data.{traj_group_name}.{traj_name}.opt_vals.values"

                    if traj_data.type == "spatial":
                        dim = traj_data.opt_vals["values"].shape[1]
                        _mc_batch_plot(ax, data, method_name, n_runs, [(nl_path, c) for c in range(dim)], pen_nl)
                        _mc_batch_plot(ax, data, method_name, n_runs, [(opt_path, c) for c in range(dim)], pen_opt)
                    elif traj_data.type == "time_series":
                        _mc_batch_plot(ax, data, method_name, n_runs, [('t_nl', None), (nl_path, None)], pen_nl)
                        _mc_batch_plot(ax, data, method_name, n_runs, [('t_opt', None), (opt_path, None)], pen_opt)
                        if i == 0:
                            limits = traj_data.nl_vals.get("limits", {})
                            if limits:
                                for val in [limits.get("upper"), limits.get("lower")]:
                                    if val is not None:
                                        if isinstance(val, np.ndarray):
                                            t_nl = data[method_name]['runs'][0]['iters'][-1]['t_nl']
                                            ax.plot(t_nl[:len(val)], val, color='k', ls='--', lw=1, alpha=0.5)
                                        else:
                                            ax.axhline(val, color='k', ls='--', lw=1, alpha=0.5)

                    if i == 0:
                        dim_m = traj_data.opt_vals["values"].shape[1] if traj_data.type == "spatial" else 2
                        plot_markers(ax, traj_data, dim_m)

            for wg_name, wg in weight_groups_time.items():
                for wi, weight_key in enumerate(wg):
                    ax = axs_weights_time[wg_name][wi]
                    if i == 0:
                        display_name = wg_name.replace('_', ' ').title()
                        ax.set_title(f"{display_name} — {WEIGHT_DISPLAY_NAMES[weight_key]}", fontsize=plot_options.title_fontsize, pad=plot_options.title_pad)
                        ax.set_xlabel("Time (s)")
                        ax.set_ylabel("Weight")
                        ax.grid(True, alpha=0.3)
                    w_path = f"{weight_key}.{wg_name}"
                    _mc_batch_plot(ax, data, method_name, n_runs, [('t_opt', None), (w_path, None)], pens[f"wt{i+1}"])

            for wg_name, wg in weight_groups_terminal.items():
                for wi, weight_key in enumerate(wg):
                    ax = axs_weights_terminal[wg_name][wi]
                    if i == 0:
                        display_name = wg_name.replace('_', ' ').title()
                        ax.set_title(f"{display_name} — {WEIGHT_DISPLAY_NAMES[weight_key]}", fontsize=plot_options.title_fontsize, pad=plot_options.title_pad)
                        ax.set_xlabel("Run")
                        ax.set_ylabel("Weight")
                        ax.grid(True, alpha=0.3)
                    w_path = f"{weight_key}.{wg_name}"
                    pen_wt = pens[f"wt{i+1}"]
                    vals = []
                    for run in range(n_runs):
                        arr = tools.get_from_path(data[method_name]['runs'][run]['iters'][-1], w_path)
                        if arr.size > 0:
                            vals.append(np.max(arr))
                    if vals:
                        ax.plot(np.arange(len(vals)), vals, color=pen_wt.lrgba[:3], alpha=pen_wt.lrgba[3],
                                linewidth=pen_wt.lw, linestyle=pen_wt.ls, marker=pen_wt.msty, markersize=pen_wt.msz)

    _set_trajectory_limits(axs_trajectories, nominal_trajectory_data, nominal_iter_data)

    all_figs = {}
    all_figs.update({f"trajectory_{k}": v for k, v in figs_trajectories.items()})
    all_figs.update({f"weights_time_{k}": v for k, v in figs_weights_time.items()})
    all_figs.update({f"weights_terminal_{k}": v for k, v in figs_weights_terminal.items()})
    all_figs.update({f"weights_iters_{k}": v for k, v in figs_weights_iters.items()})

    def _handle(pen, label):
        p = pens[pen]
        ls = p.ls if p.ls else 'None'
        return Line2D([], [], color=p.lrgba[:3], alpha=p.lrgba[3], lw=p.lw,
                       ls=ls, marker=p.msty or None, markersize=p.msz, label=label)

    if analysis_type == 'standalone':
        handles = [_handle('init', 'Initial Guess'), _handle('opt', 'Optimal'), _handle('nl', 'Nonlinear')]
        if show_iters:
            handles.append(_handle('itr_nl', 'Iterations'))
    elif analysis_type == 'mc':
        handles = [_handle(f'nl{i+1}', name) for i, name in enumerate(method_keys)]

    for fig in all_figs.values():
        first_ax = fig.axes[0]
        first_ax.legend(handles=handles, loc='best', fontsize=8, framealpha=0.8)

    save_dir = os.path.join("plots", analysis_type)
    os.makedirs(save_dir, exist_ok=True)

    for name, fig in all_figs.items():
        fig.savefig(os.path.join(save_dir, f"{name}.pdf"), bbox_inches="tight")

    print(f"Saved {len(all_figs)} figures to {save_dir}/")

    plt.show()

WEIGHT_DISPLAY_NAMES = {"W": "Penalty Weights", "dual": "Dual Weights"}

def plot_weights_time(PLTS, axs, weight_groups, method, run=0, iters=[-1],
                      pens=AttrDict({"opt": pens.opt, "nl": pens.nl, "itr_opt": pens.itr_opt, "itr_nl": pens.itr_nl})):
    for name, weight_key_list in weight_groups.items():
        for i, weight_key in enumerate(weight_key_list):
            ax = axs[name][i]
            w_path = f"{weight_key}.{name}"
            display_name = name.replace('_', ' ').title()
            ax.set_title(f"{display_name} — {WEIGHT_DISPLAY_NAMES[weight_key]}", fontsize=plot_options.title_fontsize, pad=plot_options.title_pad)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Weight")
            ax.grid(True, alpha=0.3)
            ins_opt = {'label': 'Optimal Solution', 'x': 't_opt', 'y': w_path, 'method': method, 'run': run, 'iters': iters, 'legend': "legend1"}
            PLTS.addPlot2D(ax, pen=pens.wt_opt, ins=ins_opt)

def plot_weights_iters(PLTS, axs, weight_groups, method, run=0,
                       pens=AttrDict({"opt": pens.opt, "nl": pens.nl, "itr_opt": pens.itr_opt, "itr_nl": pens.itr_nl})):
    for name, weight_key_list in weight_groups.items():
        for i, weight_key in enumerate(weight_key_list):
            ax = axs[name][i]
            w_path = f"{weight_key}.{name}"
            display_name = name.replace('_', ' ').title()
            ax.set_title(f"{display_name} — {WEIGHT_DISPLAY_NAMES[weight_key]}", fontsize=plot_options.title_fontsize, pad=plot_options.title_pad)
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Weight")
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ins_opt = {'label': 'Optimal Solution', 'x': 't_opt', 'y': w_path, 'method': method, 'run': run, 'legend': "legend1"}
            PLTS.addPlot2D_iters(ax, pen=pens.wt_opt, ins=ins_opt)

def plot_trajectories(PLTS, axs, nominal_trajectory_data, method, run=0, iters=[-1], 
                pens = AttrDict({"opt": pens.opt, "nl": pens.nl, "itr_opt": pens.itr_opt, "itr_nl": pens.itr_nl}),
                skip_init=False):

    for traj_group_name, traj_group_data in nominal_trajectory_data.items():
        
        for i, (traj_name, current_traj_data) in enumerate(traj_group_data.items()):
            ax = axs[traj_group_name][i]

            if current_traj_data.get("title"):
                ax.set_title(current_traj_data["title"], fontsize=plot_options.title_fontsize, pad=plot_options.title_pad)
            ax.grid(True, alpha=0.3)
            if current_traj_data.get("xlabel"):
                ax.set_xlabel(current_traj_data["xlabel"])
            if current_traj_data.get("ylabel"):
                ax.set_ylabel(current_traj_data["ylabel"])
            if current_traj_data.get("zlabel") and hasattr(ax, 'set_zlabel'):
                ax.set_zlabel(current_traj_data["zlabel"])
            if current_traj_data.get("tick_nbins"):
                nbins = current_traj_data["tick_nbins"]
                ax.xaxis.set_major_locator(MaxNLocator(nbins=nbins))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins))
                if hasattr(ax, 'zaxis'):
                    ax.zaxis.set_major_locator(MaxNLocator(nbins=nbins))

            opt_vals_path = f"trajectory_data.{traj_group_name}.{traj_name}.opt_vals.values"
            nl_vals_path = f"trajectory_data.{traj_group_name}.{traj_name}.nl_vals.values"

            traj_type = current_traj_data.type

            if traj_type == "spatial":
                plot_spatial_trajectories(PLTS, method, run, iters, current_traj_data, nl_vals_path, opt_vals_path, ax, pens, skip_init=skip_init)
            elif traj_type == "time_series":
                plot_time_series_trajectories(PLTS, method, run, iters, current_traj_data, nl_vals_path, opt_vals_path, ax, pens, skip_init=skip_init)

def plot_spatial_trajectories(PLTS, method, run, iters, current_traj_data, nl_vals_path, opt_vals_path, ax, pens, skip_init=False):
    dim = current_traj_data.opt_vals["values"].shape[1]

    if iters != [-1]:
        if dim == 3:
            ins_nl = {'label': 'Nonlinear Propagation', 'x': nl_vals_path, 'x_idx': 0, 'y': nl_vals_path, 'y_idx': 1, 'z': nl_vals_path, 'z_idx': 2, 'method': method, 'run': run, 'iters': iters, 'legend': "legend1"}
            PLTS.addPlot3D(ax, pen=pens.itr_nl, ins=ins_nl)

            ins_opt = {'label': 'Optimal Soltution', 'x': opt_vals_path, 'x_idx': 0, 'y': opt_vals_path, 'y_idx': 1, 'z': opt_vals_path, 'z_idx': 2, 'method': method, 'run': run, 'iters': iters, 'legend': "legend1"}
            PLTS.addPlot3D(ax, pen=pens.itr_opt, ins=ins_opt)
        else:
            ins_nl = {'label': 'Nonlinear Propagation', 'x': nl_vals_path, 'x_idx': 0, 'y': nl_vals_path, 'y_idx': 1, 'method': method, 'run': run, 'iters': iters, 'legend': "legend1"}
            PLTS.addPlot2D(ax, pen=pens.itr_nl, ins=ins_nl)

            ins_opt = {'label': 'Optimal Soltution', 'x': opt_vals_path, 'x_idx': 0, 'y': opt_vals_path, 'y_idx': 1, 'method': method, 'run': run, 'iters': iters, 'legend': "legend1"}
            PLTS.addPlot2D(ax, pen=pens.itr_opt, ins=ins_opt)

    if not skip_init:
        init_nl_vals_path = nl_vals_path.replace('nl_vals', 'init_nl_vals')

    if dim == 3:
        if not skip_init:
            ins_init = {'label': 'Initial Guess', 'x': init_nl_vals_path, 'x_idx': 0, 'y': init_nl_vals_path, 'y_idx': 1, 'z': init_nl_vals_path, 'z_idx': 2, 'method': method, 'run': run, 'iters': [-1], 'legend': "legend1"}
            PLTS.addPlot3D(ax, pen=pens.init, ins=ins_init)

        ins_nl = {'label': 'Nonlinear Propagation', 'x': nl_vals_path, 'x_idx': 0, 'y': nl_vals_path, 'y_idx': 1, 'z': nl_vals_path, 'z_idx': 2, 'method': method, 'run': run, 'iters': [-1], 'legend': "legend1"}
        PLTS.addPlot3D(ax, pen=pens.nl, ins=ins_nl)

        ins_opt = {'label': 'Optimal Soltution', 'x': opt_vals_path, 'x_idx': 0, 'y': opt_vals_path, 'y_idx': 1, 'z': opt_vals_path, 'z_idx': 2, 'method': method, 'run': run, 'iters': [-1], 'legend': "legend1"}
        PLTS.addPlot3D(ax, pen=pens.opt, ins=ins_opt)
    else:
        if not skip_init:
            ins_init = {'label': 'Initial Guess', 'x': init_nl_vals_path, 'x_idx': 0, 'y': init_nl_vals_path, 'y_idx': 1, 'method': method, 'run': run, 'iters': [-1], 'legend': "legend1"}
            PLTS.addPlot2D(ax, pen=pens.init, ins=ins_init)

        ins_nl = {'label': 'Nonlinear Propagation', 'x': nl_vals_path, 'x_idx': 0, 'y': nl_vals_path, 'y_idx': 1, 'method': method, 'run': run, 'iters': [-1], 'legend': "legend1"}
        PLTS.addPlot2D(ax, pen=pens.nl, ins=ins_nl)

        ins_opt = {'label': 'Optimal Soltution', 'x': opt_vals_path, 'x_idx': 0, 'y': opt_vals_path, 'y_idx': 1, 'method': method, 'run': run, 'iters': [-1], 'legend': "legend1"}
        PLTS.addPlot2D(ax, pen=pens.opt, ins=ins_opt)

    plot_markers(ax, current_traj_data, dim)
    plot_quivers(ax, current_traj_data, dim)

def plot_quivers(ax, traj_data, dim):
    quivers = traj_data.opt_vals.get("quivers", [])
    if not quivers:
        return
    origins = traj_data.opt_vals["values"]
    for qdata in quivers:
        cfg    = qdata["config"]
        dirs   = qdata["dirs"]
        stride = int(cfg.get("stride", 1))
        scale  = float(cfg.get("scale", 1.0))
        color  = cfg.get("color", [0.2, 0.2, 0.2])
        alpha  = float(cfg.get("alpha", 0.8))
        lw     = float(cfg.get("linewidth", 1.5))
        negate = bool(cfg.get("negate", False))

        idx = np.arange(0, len(origins), stride)
        o   = origins[idx]
        d   = dirs[idx] * scale * (-1.0 if negate else 1.0)

        # build NaN-separated segments so one ax.plot call draws all lines
        nan_col = np.full((len(idx), 1), np.nan)
        tips    = o + d
        # interleave: base, tip, NaN for each segment → shape (3*N, ndim)
        segs = np.concatenate(
            [np.stack([o, tips, np.full_like(o, np.nan)], axis=1).reshape(-1, o.shape[1])],
            axis=0,
        )

        if dim == 3:
            ax.plot(segs[:, 0], segs[:, 1], segs[:, 2],
                    color=color, alpha=alpha, linewidth=lw, linestyle='-')
        else:
            ax.plot(segs[:, 0], segs[:, 1],
                    color=color, alpha=alpha, linewidth=lw, linestyle='-')

MARKER_DEFAULTS = {
    'marker': '*',
    'color': [0.8, 0.0, 0.0],
    'size': 80,
    'edgecolor': 'k',
    'edgewidth': 0.4,
    'zorder': 10,
    'fontsize': 7,
    'text_offset': [0.0, 0.0],
}

def plot_markers(ax, traj_data, dim):
    markers = traj_data.get("markers", None)
    if not markers:
        return

    for m in markers:
        xy = m["xy"]
        label = m.get("label", None)

        mkr     = m.get("marker", MARKER_DEFAULTS['marker'])
        color   = m.get("color", MARKER_DEFAULTS['color'])
        sz      = m.get("size", MARKER_DEFAULTS['size'])
        ec      = m.get("edgecolor", MARKER_DEFAULTS['edgecolor'])
        ew      = m.get("edgewidth", MARKER_DEFAULTS['edgewidth'])
        zo      = m.get("zorder", MARKER_DEFAULTS['zorder'])
        fs      = m.get("fontsize", MARKER_DEFAULTS['fontsize'])
        t_off   = m.get("text_offset", MARKER_DEFAULTS['text_offset'])

        if dim == 3 and len(xy) >= 3:
            ax.scatter(xy[0], xy[1], xy[2], marker=mkr, s=sz, c=[color],
                       edgecolors=ec, linewidths=ew, zorder=zo)
            if label:
                ax.text(xy[0] + t_off[0], xy[1] + t_off[1], xy[2] + (t_off[2] if len(t_off) > 2 else 0),
                        label, fontsize=fs, ha='left', va='bottom')
        else:
            ax.scatter(xy[0], xy[1], marker=mkr, s=sz, c=[color],
                       edgecolors=ec, linewidths=ew, zorder=zo)
            if label:
                ax.annotate(label, (xy[0], xy[1]), textcoords="offset points",
                            xytext=(t_off[0] + 4, t_off[1] + 4), fontsize=fs,
                            ha='left', va='bottom')

def plot_time_series_trajectories(PLTS, method, run, iters, current_traj_data, nl_vals_path, opt_vals_path, ax, pens, skip_init=False):
    if iters != [-1]:
        ins_nl = {'label': 'Nonlinear Propagation', 'x': 't_nl', 'y': nl_vals_path, 'method': method, 'run': run, 'iters': iters, 'legend': "legend1"}
        PLTS.addPlot2D(ax, pen=pens.itr_nl, ins=ins_nl)

        ins_opt = {'label': 'Optimal Soltution', 'x': 't_opt', 'y': opt_vals_path, 'method': method, 'run': run, 'iters': iters, 'legend': "legend1"}
        PLTS.addPlot2D(ax, pen=pens.itr_opt, ins=ins_opt)

    if not skip_init:
        init_nl_vals_path = nl_vals_path.replace('nl_vals', 'init_nl_vals')
        ins_init = {'label': 'Initial Guess', 'x': 't_init_nl', 'y': init_nl_vals_path, 'method': method, 'run': run, 'iters': [-1], 'legend': "legend1"}
        PLTS.addPlot2D(ax, pen=pens.init, ins=ins_init)

    ins_nl = {'label': 'Nonlinear Propagation', 'x': 't_nl', 'y': nl_vals_path, 'method': method, 'run': run, 'iters': [-1], 'legend': "legend1"}
    PLTS.addPlot2D(ax, pen=pens.nl, ins=ins_nl)

    ins_opt = {'label': 'Optimal Soltution', 'x': 't_opt', 'y': opt_vals_path, 'method': method, 'run': run, 'iters': [-1], 'legend': "legend1"}
    PLTS.addPlot2D(ax, pen=pens.opt, ins=ins_opt)

    limits = current_traj_data.nl_vals.get("limits", {})
    if limits:
        for val in [limits.get("upper"), limits.get("lower")]:
            if val is not None:
                if isinstance(val, np.ndarray):
                    t_nl = np.array(PLTS.data[method]['runs'][run]['iters'])[-1]['t_nl']
                    ax.plot(t_nl[:len(val)], val, color='k', ls='--', lw=1, alpha=0.5)
                else:
                    ax.axhline(val, color='k', ls='--', lw=1, alpha=0.5)

def plot_animated(trajopt_obj, data, analysis_type):
    from matplotlib.animation import FuncAnimation
    from IPython.display import display, HTML

    method = list(data.keys())[0]
    all_iters = data[method]["runs"][0]["iters"]
    n_iters = len(all_iters)
    if n_iters < 2:
        return

    pcfg = trajopt_obj.problem.config.problem.get('plot_config', {}).get('trajectory_groups', {})
    last_traj = all_iters[-1]["trajectory_data"]
    t_init = all_iters[0].get("t_init_nl")

    save_dir = os.path.join("plots", analysis_type)
    os.makedirs(save_dir, exist_ok=True)

    for grp_name, grp_data in last_traj.items():
        grp_cfg = pcfg.get(grp_name, {})
        items = list(grp_data.items())
        n = len(items)
        ncols = 3 if n == 3 else int(np.ceil(np.sqrt(n)))
        nrows = int(np.ceil(n / ncols))

        fig = plt.figure(figsize=grp_cfg.get('figsize', plot_options.figsize), dpi=150)
        axes = []
        for i, (name, td) in enumerate(items):
            is3d = td.type == "spatial" and td.opt_vals["values"].shape[1] == 3
            axes.append(fig.add_subplot(nrows, ncols, i + 1, projection="3d" if is3d else None))
        fig.subplots_adjust(hspace=0.4, wspace=0.3)

        init_grp = all_iters[0]["trajectory_data"].get(grp_name, {})

        axis_limits = _compute_axis_limits(all_iters, grp_name, items, t_init, init_grp)

        def update(frame, axes=axes, items=items, init_grp=init_grp, grp_name=grp_name, axis_limits=axis_limits):
            it = all_iters[frame]
            it_grp = it["trajectory_data"].get(grp_name, {})
            for i, (name, _) in enumerate(items):
                ax, td = axes[i], it_grp[name]
                ax.cla()
                ax.set_title(td.get("title", name), fontsize=9)
                ax.grid(True, alpha=0.3)
                if td.get("xlabel"): ax.set_xlabel(td["xlabel"])
                if td.get("ylabel"): ax.set_ylabel(td["ylabel"])
                _draw_anim_frame(ax, td, it["t_opt"], it["t_nl"], t_init, init_grp.get(name))
                lims = axis_limits[i]
                ax.set_xlim(lims[0]); ax.set_ylim(lims[1])
                if len(lims) > 2 and hasattr(ax, 'set_zlim'):
                    ax.set_zlim(lims[2])
            fig.suptitle(f"Iteration {frame + 1} / {n_iters}", fontsize=10)

        anim = FuncAnimation(fig, update, frames=n_iters, interval=300)
        anim.save(os.path.join(save_dir, f"anim_{grp_name}.gif"), writer='pillow')
        display(HTML(anim.to_jshtml()))
        plt.close(fig)

    print(f"Saved {len(last_traj)} animation(s) to {save_dir}/")

def _compute_axis_limits(all_iters, grp_name, items, t_init, init_grp):
    PAD = 0.05
    limits = {}
    for i, (name, _) in enumerate(items):
        all_x, all_y, all_z = [], [], []
        for it in all_iters:
            td = it["trajectory_data"].get(grp_name, {}).get(name)
            if td is None:
                continue
            ov, nv = td.opt_vals["values"], td.nl_vals["values"]
            if td.type == "time_series":
                all_x.extend([it["t_opt"], it["t_nl"]])
                all_y.extend([ov, nv])
            elif td.type == "spatial":
                for v in [ov, nv]:
                    all_x.append(v[:, 0]); all_y.append(v[:, 1])
                    if v.shape[1] > 2:
                        all_z.append(v[:, 2])
        init_td = init_grp.get(name)
        if init_td is not None:
            iv = init_td.init_nl_vals["values"]
            td0 = all_iters[0]["trajectory_data"][grp_name][name]
            if td0.type == "time_series":
                all_x.append(t_init); all_y.append(iv)
            elif td0.type == "spatial":
                all_x.append(iv[:, 0]); all_y.append(iv[:, 1])
                if iv.shape[1] > 2:
                    all_z.append(iv[:, 2])

        xmin, xmax = np.min(np.concatenate(all_x)), np.max(np.concatenate(all_x))
        ymin, ymax = np.min(np.concatenate(all_y)), np.max(np.concatenate(all_y))
        dx, dy = (xmax - xmin) * PAD, (ymax - ymin) * PAD
        lims = [(xmin - dx, xmax + dx), (ymin - dy, ymax + dy)]
        if all_z:
            zmin, zmax = np.min(np.concatenate(all_z)), np.max(np.concatenate(all_z))
            dz = (zmax - zmin) * PAD
            lims.append((zmin - dz, zmax + dz))
        limits[i] = lims
    return limits

def _draw_anim_frame(ax, td, t_opt, t_nl, t_init, init_td):
    ov, nv = td.opt_vals["values"], td.nl_vals["values"]

    if td.type == "time_series":
        if init_td is not None:
            ax.plot(t_init, init_td.init_nl_vals["values"], 'k--', lw=0.8, alpha=0.4)
        ax.plot(t_nl, nv, 'r-', lw=1.5)
        ax.plot(t_opt, ov, 'bo', ms=2)
    elif td.type == "spatial":
        cols = lambda v: tuple(v[:, j] for j in range(v.shape[1]))
        if init_td is not None:
            ax.plot(*cols(init_td.init_nl_vals["values"]), 'k--', lw=0.8, alpha=0.4)
        ax.plot(*cols(nv), 'r-', lw=1.5)
        ax.plot(*cols(ov), 'bo', ms=2)

def create_grid(num_groups, cfg=None):
    if cfg is None:
        cfg = {}

    if num_groups == 3:
        num_columns = 3
        num_rows = 1
    else:
        num_columns = int(np.ceil(np.sqrt(num_groups)))
        num_rows = int(np.ceil(num_groups / num_columns))

    gap_x = cfg.get('grid_gap_x', plot_options.grid_gap_x)
    gap_y = cfg.get('grid_gap_y', plot_options.grid_gap_y)
    margins = cfg.get('margins', plot_options.margins)
    margin_left, margin_right, margin_top, margin_bottom = margins

    usable_w = 1.0 - margin_left - margin_right
    usable_h = 1.0 - margin_top - margin_bottom

    width_ratios = cfg.get('width_ratios', None)

    dy = (usable_h - (num_rows - 1) * gap_y) / num_rows

    grid = {}

    if width_ratios is not None and len(width_ratios) == num_columns:
        total_ratio = sum(width_ratios)
        total_gap = (num_columns - 1) * gap_x
        col_widths = [(r / total_ratio) * (usable_w - total_gap) for r in width_ratios]

        for i in range(num_rows):
            x_cursor = margin_left
            for j in range(num_columns):
                tag = i * num_columns + j
                grid[tag] = [x_cursor, (1.0 - margin_top) - (i + 1) * dy - i * gap_y, col_widths[j], dy]
                x_cursor += col_widths[j] + gap_x
    else:
        dx = (usable_w - (num_columns - 1) * gap_x) / num_columns
        for i in range(num_rows):
            for j in range(num_columns):
                tag = i * num_columns + j
                x = margin_left + j * (dx + gap_x)
                y = (1.0 - margin_top) - (i + 1) * dy - i * gap_y
                grid[tag] = [x, y, dx, dy]
    
    return grid
