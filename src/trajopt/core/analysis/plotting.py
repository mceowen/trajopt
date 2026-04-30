"""Default plotting routines for standalone and Monte Carlo SCP analysis results."""

import os
import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from trajopt.core.analysis.trajplots import SCVXPLOTS
from trajopt.core.problem import Problem
from trajopt.utils import tools
from trajopt.utils.tools import AttrDict, recursive_attrdict

plt.rcParams["text.usetex"] = True
plt.rcParams.update(
    {
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "lines.linewidth": 1.0,
        "axes.formatter.useoffset": False,
        "axes.formatter.limits": [-1, 3],
        "path.simplify": True,
        "path.simplify_threshold": 0.1,
    },
)

plot_options = AttrDict(
    {
        "figsize": (10, 2.6),
        "dpi": 300,
        "grid_gap_x": 0.06,
        "grid_gap_y": 0.12,
        "margins": [0.08, 0.02, 0.04, 0.12],
        "title_fontsize": 10,
        "title_pad": 4,
    },
)

# pen: frgba, lrgba, lw, ls, msty, msz
pens = recursive_attrdict(
    {
        "init": {"frgba": [0, 0, 0, 0.1], "lrgba": [0, 0, 0, 1.0], "lw": 1, "ls": "--", "msty": "", "msz": 3},
        "nl": {"frgba": [0, 0, 0, 0.1], "lrgba": [1, 0, 0, 1.0], "lw": 2, "ls": "-", "msty": "", "msz": 3},
        "opt": {"frgba": [0, 0, 0, 0.1], "lrgba": [0, 0, 1, 1.0], "lw": 1, "ls": "", "msty": "o", "msz": 3},
        "itr_opt": {"frgba": [0, 0, 0, 0.1], "lrgba": [0.7, 0, 0.3, 0.2], "lw": 1, "ls": "", "msty": "o", "msz": 3},
        "itr_nl": {"frgba": [0, 0, 0, 0.1], "lrgba": [0.7, 0, 0.3, 0.4], "lw": 1, "ls": "-", "msty": "", "msz": 3},
        "wt_opt": {"frgba": [0, 0, 0, 0.1], "lrgba": [0.7, 0, 0.3, 0.8], "lw": 1, "ls": "-", "msty": "o", "msz": 2},
        "opt1": {"frgba": [0, 0, 0, 0.1], "lrgba": [0, 0, 1, 1.0], "lw": 1, "ls": "", "msty": "o", "msz": 2},
        "nl1": {"frgba": [0, 0, 0, 0.1], "lrgba": [0, 0, 1, 1.0], "lw": 1, "ls": "-", "msty": "", "msz": 3},
        "opt2": {"frgba": [0, 0, 0, 0.1], "lrgba": [1, 0, 1, 1.0], "lw": 1, "ls": "", "msty": "o", "msz": 2},
        "nl2": {"frgba": [0, 0, 0, 0.1], "lrgba": [1, 0, 1, 1.0], "lw": 1, "ls": "-", "msty": "", "msz": 3},
        "opt3": {"frgba": [0, 0, 0, 0.1], "lrgba": [0, 1, 0, 1.0], "lw": 1, "ls": "", "msty": "o", "msz": 2},
        "nl3": {"frgba": [0, 0, 0, 0.1], "lrgba": [0, 1, 0, 1.0], "lw": 1, "ls": "-", "msty": "", "msz": 3},
        "wt1": {"frgba": [0, 0, 0, 0.1], "lrgba": [0, 0, 1, 1.0], "lw": 1, "ls": "-", "msty": "o", "msz": 2},
        "wt2": {"frgba": [0, 0, 0, 0.1], "lrgba": [1, 0, 1, 1.0], "lw": 1, "ls": "-", "msty": "o", "msz": 2},
        "wt3": {"frgba": [0, 0, 0, 0.1], "lrgba": [0, 1, 0, 1.0], "lw": 1, "ls": "-", "msty": "o", "msz": 2},
    },
)


def _set_trajectory_limits(
    axs_trajectories: dict,
    nominal_trajectory_data: dict,
    data: dict,
    limits_all_iters: bool = True,
    pad: float = 0.05,
) -> None:
    """Set axis limits for standalone trajectory plots, using data range across all displayed iters."""
    first_method = next(iter(data.keys()))
    all_iters = data[first_method]["runs"][-1]["iters"]

    for traj_group_name, traj_group_data in nominal_trajectory_data.items():
        for j, (traj_name, traj_data) in enumerate(traj_group_data.items()):
            ax = axs_trajectories[traj_group_name][j]

            iter_range = all_iters if limits_all_iters else all_iters[-1:]
            spatial_parts = []
            t_parts = []
            y_parts = []

            for it in iter_range:
                it_td = it.get("trajectory_data", {}).get(traj_group_name, {}).get(traj_name)
                if it_td is None:
                    continue
                ov = it_td.opt_vals["values"]
                nv = it_td.nl_vals["values"]
                if traj_data.type == "spatial":
                    spatial_parts.extend([ov, nv])
                elif traj_data.type == "time_series":
                    t_parts.extend([it["t_opt"], it["t_nl"]])
                    y_parts.extend([ov.ravel(), nv.ravel()])

                iv = it_td.get("init_nl_vals", {}).get("values") if it_td.get("init_nl_vals") else None
                if iv is not None:
                    if traj_data.type == "spatial":
                        spatial_parts.append(iv)
                    elif traj_data.type == "time_series":
                        t_init_nl = it.get("t_init_nl")
                        if t_init_nl is not None:
                            t_parts.append(t_init_nl)
                            y_parts.append(iv.ravel())

            if traj_data.type == "spatial" and spatial_parts:
                all_vals = np.concatenate(spatial_parts, axis=0)
                for dim_i, setter in enumerate(
                    [ax.set_xlim, ax.set_ylim]
                    + ([ax.set_zlim] if hasattr(ax, "set_zlim") and all_vals.shape[1] > 2 else []),
                ):
                    lo, hi = all_vals[:, dim_i].min(), all_vals[:, dim_i].max()
                    margin = (hi - lo) * pad if hi > lo else 0.5
                    if dim_i == 0 and traj_data.get("invert_x", False):
                        setter(hi + margin, lo - margin)
                    else:
                        setter(lo - margin, hi + margin)
            elif traj_data.type == "time_series" and t_parts:
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


def _set_mc_trajectory_limits(
    axs_trajectories: dict,
    nominal_trajectory_data: dict,
    data: dict,
    traj_configs: dict | None = None,
    pad: float = 0.05,
    show_runs: int | None = None,
) -> None:
    """Set axis limits for Monte Carlo trajectory plots, spanning all methods and runs."""
    if traj_configs is None:
        traj_configs = {}
    for traj_group_name, traj_group_data in nominal_trajectory_data.items():
        for j, (traj_name, traj_data) in enumerate(traj_group_data.items()):
            ax = axs_trajectories[traj_group_name][j]
            mc_scale = traj_configs.get(traj_name, {}).get("mc_data_scale", None)

            nl_path = f"trajectory_data.{traj_group_name}.{traj_name}.nl_vals.values"
            opt_path = f"trajectory_data.{traj_group_name}.{traj_name}.opt_vals.values"

            if traj_data.type == "spatial":
                all_vals = []
                for method_data in data.values():
                    n = len(method_data["runs"]) if show_runs is None else min(show_runs, len(method_data["runs"]))
                    for run_idx in range(n):
                        iter_data = method_data["runs"][run_idx]["iters"][-1]
                        for path in [nl_path, opt_path]:
                            arr = tools.get_from_path(iter_data, path)
                            if arr.size > 0:
                                all_vals.append(arr)
                if all_vals:
                    combined = np.concatenate(all_vals, axis=0)
                    setters = [ax.set_xlim, ax.set_ylim]
                    if hasattr(ax, "set_zlim") and combined.shape[1] > 2:
                        setters.append(ax.set_zlim)
                    for dim_i, setter in enumerate(setters):
                        lo, hi = combined[:, dim_i].min(), combined[:, dim_i].max()
                        margin = (hi - lo) * pad if hi > lo else 0.5
                        if dim_i == 0 and traj_data.get("invert_x", False):
                            setter(hi + margin, lo - margin)
                        else:
                            setter(lo - margin, hi + margin)

            elif traj_data.type == "time_series":
                cur_cfg = traj_configs.get(traj_name, {})
                all_t, all_y = [], []
                for method_data in data.values():
                    n = len(method_data["runs"]) if show_runs is None else min(show_runs, len(method_data["runs"]))
                    for run_idx in range(n):
                        iter_data = method_data["runs"][run_idx]["iters"][-1]
                        for path, t_key in [(nl_path, "t_nl"), (opt_path, "t_opt")]:
                            arr = tools.get_from_path(iter_data, path)
                            if arr.size > 0:
                                scaled = arr.ravel() * mc_scale if mc_scale is not None else arr.ravel()
                                all_t.append(iter_data[t_key])
                                all_y.append(scaled)
                upper = cur_cfg.get("upper_limit", None)
                lower = cur_cfg.get("lower_limit", None)
                if upper is None and lower is None:
                    pickle_limits = traj_data.nl_vals.get("limits", {})
                    if pickle_limits:
                        upper = pickle_limits.get("upper")
                        lower = pickle_limits.get("lower")
                for val in [upper, lower]:
                    if val is not None:
                        if isinstance(val, np.ndarray):
                            all_y.append(val.ravel())
                        else:
                            all_y.append(np.array([val]))
                if all_t and all_y:
                    all_x_cat = np.concatenate(all_t)
                    all_y_cat = np.concatenate(all_y)
                    xlo, xhi = all_x_cat.min(), all_x_cat.max()
                    ylo, yhi = all_y_cat.min(), all_y_cat.max()
                    xm = (xhi - xlo) * pad if xhi > xlo else 0.5
                    ym = (yhi - ylo) * pad if yhi > ylo else 0.5
                    ax.set_xlim(xlo - xm, xhi + xm)
                    ax.set_ylim(ylo - ym, yhi + ym)


def _mc_collect_batch(
    data: dict,
    method_name: str,
    n_runs: int,
    paths_and_cols: list[tuple[str, int | None]],
) -> list[np.ndarray]:
    """Collect NaN-separated trajectory segments for all MC runs into plottable arrays."""
    all_segs = [[] for _ in paths_and_cols]
    nan = np.array([np.nan])
    for run in range(n_runs):
        iter_data = data[method_name]["runs"][run]["iters"][-1]
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


def _mc_batch_plot(
    ax: Axes,
    data: dict,
    method_name: str,
    n_runs: int,
    paths_and_cols: list[tuple[str, int | None]],
    pen: AttrDict,
    y_scale: float | None = None,
) -> None:
    """Plot all MC run segments onto ax using _mc_collect_batch, optionally scaling the y data."""
    arrays = _mc_collect_batch(data, method_name, n_runs, paths_and_cols)
    if y_scale is not None and len(arrays) > 1:
        arrays[-1] = arrays[-1] * y_scale
    if arrays[0].size:
        ax.plot(
            *arrays,
            color=pen.lrgba[:3],
            alpha=pen.lrgba[3],
            linewidth=pen.lw,
            linestyle=pen.ls,
            marker=pen.msty,
            markersize=pen.msz,
            rasterized=True,
        )


def _plot_phase_lines(
    axs_trajectories: dict, nominal_trajectory_data: dict, problem: Problem, nominal_iter_data: dict,
) -> None:
    """Draw vertical dashed lines at phase transition times on all time-series trajectory axes."""
    if not problem.phases or len(problem.phases) <= 1:
        return
    t_opt = nominal_iter_data["t_opt"]
    phase_times = [t_opt[min(phase.start, len(t_opt) - 1)] for phase in problem.phases[1:]]
    for traj_group_name, traj_group_data in nominal_trajectory_data.items():
        for i, (traj_name, traj_data) in enumerate(traj_group_data.items()):
            if traj_data.type == "time_series":
                ax = axs_trajectories[traj_group_name][i]
                for t_phase in phase_times:
                    ax.axvline(t_phase, color="0.15", ls="--", lw=1.8, alpha=0.9, zorder=5)


def plot_default(
    trajopt_obj: Any,
    data: dict,
    analysis_type: str,
    show_iters: bool = False,
    show_runs: int | None = None,
) -> None:
    """Generate and save the full set of trajectory, constraint, and weight figures.

    Supports 'standalone' (single solve) and 'mc' (Monte Carlo) analysis types.
    Figures are saved as PDFs under plots/<analysis_type>/.
    """
    PLTS = SCVXPLOTS(data)

    method_keys = list(data.keys())
    first_method = method_keys[0]

    nominal_iter_data = data[method_keys[0]]["runs"][-1]["iters"][-1]

    iters = slice(1, None) if show_iters else [-1]

    problem = trajopt_obj.problem
    problem_config = problem.config.problem

    nominal_trajectory_data = nominal_iter_data["trajectory_data"]
    nominal_W_data = nominal_iter_data["W"]
    nominal_dual_data = nominal_iter_data["dual"]

    pcfg = problem_config.get("plot_config", {})
    traj_group_pcfg = pcfg.get("trajectory_groups", {})
    traj_configs = problem_config.get("trajectories", {})

    figs_trajectories = AttrDict({})
    axs_trajectories = AttrDict({})

    for trajectory_group_name, trajectory_group_data in nominal_trajectory_data.items():
        grp_cfg = traj_group_pcfg.get(trajectory_group_name, {})
        figs_trajectories[trajectory_group_name] = plt.figure(
            figsize=grp_cfg.get("figsize", plot_options.figsize), dpi=plot_options.dpi,
        )

        plt_type = AttrDict({})

        for i, (trajectory_name, current_trajectory_data) in enumerate(trajectory_group_data.items()):
            if current_trajectory_data.type == "spatial":
                dim = current_trajectory_data.opt_vals["values"].shape[1]
            elif current_trajectory_data.type == "time_series":
                dim = 2

            plt_type[i] = "3D" if dim == 3 else "2D"

        num_trajectories_in_group = len(trajectory_group_data.keys())
        grid2_ins = {"plt_typs": plt_type}
        if "pad_3d" in grp_cfg:
            grid2_ins["pad_3d"] = grp_cfg["pad_3d"]
        axs_trajectories[trajectory_group_name] = PLTS.createGrid2(
            figs_trajectories[trajectory_group_name],
            grid=create_grid(num_trajectories_in_group, cfg=grp_cfg),
            ins=grid2_ins,
        )

    # weights over time
    figs_weights_time = AttrDict({})
    axs_weights_time = AttrDict({})
    t_steps = nominal_iter_data["t_opt"].shape[0]
    active_penalty_names_time = [
        key
        for key, val in nominal_W_data.items()
        if val.size > 0 and (nominal_W_data[key].shape[0] == t_steps or nominal_W_data[key].shape[0] == t_steps - 1)
    ]
    active_dual_names_time = [
        key
        for key, val in nominal_dual_data.items()
        if val.size > 0
        and (val.shape[0] == t_steps or val.shape[0] == t_steps - 1)
    ]
    weight_groups_time = AttrDict({})
    for name in active_penalty_names_time:
        if name not in weight_groups_time:
            weight_groups_time[name] = []
        weight_groups_time[name].append("W")
    for name in active_dual_names_time:
        if name not in weight_groups_time:
            weight_groups_time[name] = []
        weight_groups_time[name].append("dual")
    for wg_name, wg in weight_groups_time.items():
        figs_weights_time[wg_name] = plt.figure(figsize=plot_options.figsize, dpi=plot_options.dpi)
        axs_weights_time[wg_name] = PLTS.createGrid(figs_weights_time[wg_name], grid=create_grid(len(wg)))

    # terminal weights (non-time-based)
    figs_weights_terminal = AttrDict({})
    axs_weights_terminal = AttrDict({})
    active_penalty_names_terminal = [
        key for key, val in nominal_W_data.items() if val.size > 0 and key not in active_penalty_names_time
    ]
    active_dual_names_terminal = [
        key for key, val in nominal_dual_data.items() if val.size > 0 and key not in active_dual_names_time
    ]
    weight_groups_terminal = AttrDict({})
    for name in active_penalty_names_terminal:
        if name not in weight_groups_terminal:
            weight_groups_terminal[name] = []
        weight_groups_terminal[name].append("W")
    for name in active_dual_names_terminal:
        if name not in weight_groups_terminal:
            weight_groups_terminal[name] = []
        weight_groups_terminal[name].append("dual")
    for wg_name, wg in weight_groups_terminal.items():
        figs_weights_terminal[wg_name] = plt.figure(figsize=plot_options.figsize, dpi=plot_options.dpi)
        axs_weights_terminal[wg_name] = PLTS.createGrid(figs_weights_terminal[wg_name], grid=create_grid(len(wg)))

    # weights over iterations
    figs_weights_iters = AttrDict({})
    axs_weights_iters = AttrDict({})
    active_penalty_names_iters = [key for key, val in nominal_W_data.items() if val.size > 0]
    active_dual_names_iters = [key for key, val in nominal_dual_data.items() if val.size > 0]
    weight_groups_iters = AttrDict({})
    for name in active_penalty_names_iters:
        if name not in weight_groups_iters:
            weight_groups_iters[name] = []
        weight_groups_iters[name].append("W")
    for name in active_dual_names_iters:
        if name not in weight_groups_iters:
            weight_groups_iters[name] = []
        weight_groups_iters[name].append("dual")
    for wg_name, wg in weight_groups_iters.items():
        figs_weights_iters[wg_name] = plt.figure(figsize=plot_options.figsize, dpi=plot_options.dpi)
        axs_weights_iters[wg_name] = PLTS.createGrid(figs_weights_iters[wg_name], grid=create_grid(len(wg)))

    if analysis_type == "standalone":
        itr_opt_pen = AttrDict(dict(pens.itr_opt))
        itr_nl_pen = AttrDict(dict(pens.itr_nl))
        if "iter_alpha" in pcfg:
            a = float(pcfg["iter_alpha"])
            itr_opt_pen.lrgba = [*list(itr_opt_pen.lrgba[:3]), a]
            itr_nl_pen.lrgba = [*list(itr_nl_pen.lrgba[:3]), a]
        if "iter_first_frac" in pcfg:
            itr_opt_pen["first_frac"] = float(pcfg["iter_first_frac"])
            itr_nl_pen["first_frac"] = float(pcfg["iter_first_frac"])
        standalone_pens = AttrDict(
            {
                "opt": pens.opt,
                "nl": pens.nl,
                "itr_opt": itr_opt_pen,
                "itr_nl": itr_nl_pen,
                "init": pens.init,
                "wt_opt": pens.wt_opt,
            },
        )
        plot_trajectories(
            PLTS,
            axs_trajectories,
            nominal_trajectory_data,
            method=first_method,
            run=0,
            iters=iters,
            pens=standalone_pens,
        )
        _plot_phase_lines(axs_trajectories, nominal_trajectory_data, problem, nominal_iter_data)
        plot_weights_time(
            PLTS, axs_weights_time, weight_groups_time, method=first_method, run=0, iters=iters, pens=standalone_pens,
        )
        plot_weights_iters(
            PLTS, axs_weights_iters, weight_groups_iters, method=first_method, run=0, pens=standalone_pens,
        )
        plot_weights_iters(
            PLTS, axs_weights_terminal, weight_groups_terminal, method=first_method, run=0, pens=standalone_pens,
        )

    elif analysis_type == "mc":
        for i, (method_name, method_data) in enumerate(data.items()):
            pen_nl = pens[f"nl{i + 1}"]
            pen_opt = pens[f"opt{i + 1}"]
            n_runs = len(method_data["runs"]) if show_runs is None else min(show_runs, len(method_data["runs"]))

            for traj_group_name, traj_group_data in nominal_trajectory_data.items():
                for j, (traj_name, traj_data) in enumerate(traj_group_data.items()):
                    ax = axs_trajectories[traj_group_name][j]

                    # current YAML config for this trajectory — used for labels and mc_data_scale
                    cur_cfg = traj_configs.get(traj_name, {})

                    if i == 0:
                        title = cur_cfg.get("title") or traj_data.get("title")
                        xlabel = cur_cfg.get("xlabel") or traj_data.get("xlabel")
                        ylabel = cur_cfg.get("ylabel") or traj_data.get("ylabel")
                        zlabel = cur_cfg.get("zlabel") or traj_data.get("zlabel")
                        if title:
                            ax.set_title(title, fontsize=plot_options.title_fontsize, pad=plot_options.title_pad)
                        ax.grid(True, alpha=0.3)
                        if xlabel:
                            ax.set_xlabel(xlabel)
                        if ylabel:
                            ax.set_ylabel(ylabel)
                        if zlabel and hasattr(ax, "set_zlabel"):
                            ax.set_zlabel(zlabel)
                        tick_nbins = cur_cfg.get("tick_nbins") or traj_data.get("tick_nbins")
                        if tick_nbins:
                            ax.xaxis.set_major_locator(MaxNLocator(nbins=tick_nbins))
                            ax.yaxis.set_major_locator(MaxNLocator(nbins=tick_nbins))
                            if hasattr(ax, "zaxis"):
                                ax.zaxis.set_major_locator(MaxNLocator(nbins=tick_nbins))

                    nl_path = f"trajectory_data.{traj_group_name}.{traj_name}.nl_vals.values"
                    opt_path = f"trajectory_data.{traj_group_name}.{traj_name}.opt_vals.values"

                    # mc_data_scale: rescales old pickle data to match current units
                    # (remove from mission YAML once MC data is regenerated with current code)
                    mc_scale = cur_cfg.get("mc_data_scale", None)

                    if traj_data.type == "spatial":
                        dim = traj_data.opt_vals["values"].shape[1]
                        _mc_batch_plot(ax, data, method_name, n_runs, [(nl_path, c) for c in range(dim)], pen_nl)
                        _mc_batch_plot(ax, data, method_name, n_runs, [(opt_path, c) for c in range(dim)], pen_opt)
                    elif traj_data.type == "time_series":
                        _mc_batch_plot(
                            ax, data, method_name, n_runs, [("t_nl", None), (nl_path, None)], pen_nl, y_scale=mc_scale,
                        )
                        _mc_batch_plot(
                            ax,
                            data,
                            method_name,
                            n_runs,
                            [("t_opt", None), (opt_path, None)],
                            pen_opt,
                            y_scale=mc_scale,
                        )
                        if i == 0:
                            upper = cur_cfg.get("upper_limit", None)
                            lower = cur_cfg.get("lower_limit", None)
                            if upper is None and lower is None:
                                pickle_limits = traj_data.nl_vals.get("limits", {})
                                if pickle_limits:
                                    upper = pickle_limits.get("upper")
                                    lower = pickle_limits.get("lower")
                            for val in [upper, lower]:
                                if val is not None:
                                    if isinstance(val, np.ndarray):
                                        t_nl = data[method_name]["runs"][0]["iters"][-1]["t_nl"]
                                        ax.plot(t_nl[: len(val)], val, color="k", ls="--", lw=1, alpha=0.5)
                                    else:
                                        ax.axhline(val, color="k", ls="--", lw=1, alpha=0.5)

                    if i == 0:
                        dim_m = traj_data.opt_vals["values"].shape[1] if traj_data.type == "spatial" else 2
                        plot_markers(ax, traj_data, dim_m)

            for wg_name, wg in weight_groups_time.items():
                for wi, weight_key in enumerate(wg):
                    ax = axs_weights_time[wg_name][wi]
                    if i == 0:
                        display_name = wg_name.replace("_", " ").title()
                        ax.set_title(
                            f"{display_name} — {WEIGHT_DISPLAY_NAMES[weight_key]}",
                            fontsize=plot_options.title_fontsize,
                            pad=plot_options.title_pad,
                        )
                        ax.set_xlabel("Time (s)")
                        ax.set_ylabel("Weight")
                        ax.grid(True, alpha=0.3)
                    w_path = f"{weight_key}.{wg_name}"
                    _mc_batch_plot(ax, data, method_name, n_runs, [("t_opt", None), (w_path, None)], pens[f"wt{i + 1}"])

            for wg_name, wg in weight_groups_terminal.items():
                for wi, weight_key in enumerate(wg):
                    ax = axs_weights_terminal[wg_name][wi]
                    if i == 0:
                        display_name = wg_name.replace("_", " ").title()
                        ax.set_title(
                            f"{display_name} — {WEIGHT_DISPLAY_NAMES[weight_key]}",
                            fontsize=plot_options.title_fontsize,
                            pad=plot_options.title_pad,
                        )
                        ax.set_xlabel("Run")
                        ax.set_ylabel("Weight")
                        ax.grid(True, alpha=0.3)
                    w_path = f"{weight_key}.{wg_name}"
                    pen_wt = pens[f"wt{i + 1}"]
                    vals = []
                    for run in range(n_runs):
                        arr = tools.get_from_path(data[method_name]["runs"][run]["iters"][-1], w_path)
                        if arr.size > 0:
                            vals.append(np.max(arr))
                    if vals:
                        ax.plot(
                            np.arange(len(vals)),
                            vals,
                            color=pen_wt.lrgba[:3],
                            alpha=pen_wt.lrgba[3],
                            linewidth=pen_wt.lw,
                            linestyle=pen_wt.ls,
                            marker=pen_wt.msty,
                            markersize=pen_wt.msz,
                        )

    limits_all_iters = pcfg.get("limits_all_iters", True)
    if analysis_type == "mc":
        _set_mc_trajectory_limits(
            axs_trajectories, nominal_trajectory_data, data, traj_configs=traj_configs, show_runs=show_runs,
        )
    else:
        _set_trajectory_limits(axs_trajectories, nominal_trajectory_data, data, limits_all_iters=limits_all_iters)

    all_figs = {}
    all_figs.update(dict(figs_trajectories.items()))
    all_figs.update({f"weights_time_{k}": v for k, v in figs_weights_time.items()})
    all_figs.update({f"weights_terminal_{k}": v for k, v in figs_weights_terminal.items()})
    all_figs.update({f"weights_iters_{k}": v for k, v in figs_weights_iters.items()})

    def _handle(pen, label) -> Line2D:
        p = pens[pen]
        ls = p.ls or "None"
        return Line2D(
            [],
            [],
            color=p.lrgba[:3],
            alpha=p.lrgba[3],
            lw=p.lw,
            ls=ls,
            marker=p.msty or None,
            markersize=p.msz,
            label=label,
        )

    if analysis_type == "standalone":
        handles = [_handle("init", "Initial Guess"), _handle("opt", "Optimal"), _handle("nl", "Nonlinear")]
        if show_iters:
            handles.append(_handle("itr_nl", "Iterations"))
        traj_names = set(figs_trajectories.keys())
        for fig_name, fig in all_figs.items():
            if fig_name in traj_names:
                fig.axes[0].legend(handles=handles, loc="best", fontsize=8, framealpha=0.8)
    elif analysis_type == "mc":
        handles = [_handle(f"nl{i + 1}", name) for i, name in enumerate(method_keys)]
        for fig in all_figs.values():
            fig.axes[0].legend(handles=handles, loc="best", fontsize=8, framealpha=0.8)

    save_dir = os.path.join("plots", analysis_type)
    os.makedirs(save_dir, exist_ok=True)

    save_dpi = 600 if analysis_type == "mc" else "figure"
    for name, fig in all_figs.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                fig.tight_layout()
            except Exception:
                pass
        fig.savefig(os.path.join(save_dir, f"{name}.pdf"), dpi=save_dpi, bbox_inches="tight", pad_inches=0.02)

    print(f"Saved {len(all_figs)} figures to {save_dir}/")

    plt.show()


WEIGHT_DISPLAY_NAMES = {"W": "Penalty Weights", "dual": "Dual Weights"}


def plot_weights_time(
    PLTS: SCVXPLOTS,
    axs: dict,
    weight_groups: dict,
    method: str,
    run: int = 0,
    iters: Any = [-1],
    pens: AttrDict = AttrDict({"opt": pens.opt, "nl": pens.nl, "itr_opt": pens.itr_opt, "itr_nl": pens.itr_nl}),
) -> None:
    """Plot penalty and dual weights vs time for each weight group."""
    for name, weight_key_list in weight_groups.items():
        for i, weight_key in enumerate(weight_key_list):
            ax = axs[name][i]
            w_path = f"{weight_key}.{name}"
            display_name = name.replace("_", " ").title()
            ax.set_title(
                f"{display_name} — {WEIGHT_DISPLAY_NAMES[weight_key]}",
                fontsize=plot_options.title_fontsize,
                pad=plot_options.title_pad,
            )
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Weight")
            ax.grid(True, alpha=0.3)
            ins_opt = {
                "label": "Optimal Solution",
                "x": "t_opt",
                "y": w_path,
                "method": method,
                "run": run,
                "iters": iters,
                "legend": "legend1",
            }
            PLTS.addPlot2D(ax, pen=pens.wt_opt, ins=ins_opt)


def plot_weights_iters(
    PLTS: SCVXPLOTS,
    axs: dict,
    weight_groups: dict,
    method: str,
    run: int = 0,
    pens: AttrDict = AttrDict({"opt": pens.opt, "nl": pens.nl, "itr_opt": pens.itr_opt, "itr_nl": pens.itr_nl}),
) -> None:
    """Plot max penalty and dual weights across SCP iterations for each weight group."""
    for name, weight_key_list in weight_groups.items():
        for i, weight_key in enumerate(weight_key_list):
            ax = axs[name][i]
            w_path = f"{weight_key}.{name}"
            display_name = name.replace("_", " ").title()
            ax.set_title(
                f"{display_name} — {WEIGHT_DISPLAY_NAMES[weight_key]}",
                fontsize=plot_options.title_fontsize,
                pad=plot_options.title_pad,
            )
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Weight")
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ins_opt = {
                "label": "Optimal Solution",
                "x": "t_opt",
                "y": w_path,
                "method": method,
                "run": run,
                "legend": "legend1",
            }
            PLTS.addPlot2D_iters(ax, pen=pens.wt_opt, ins=ins_opt)


def plot_trajectories(
    PLTS: SCVXPLOTS,
    axs: dict,
    nominal_trajectory_data: dict,
    method: str,
    run: int = 0,
    iters: list[int] = [-1],
    pens: AttrDict = AttrDict({"opt": pens.opt, "nl": pens.nl, "itr_opt": pens.itr_opt, "itr_nl": pens.itr_nl}),
    skip_init: bool = False,
) -> None:
    """Dispatch spatial and time-series trajectory plots for all groups in nominal_trajectory_data."""
    for traj_group_name, traj_group_data in nominal_trajectory_data.items():
        for i, (traj_name, current_traj_data) in enumerate(traj_group_data.items()):
            ax = axs[traj_group_name][i]

            if current_traj_data.get("title"):
                ax.set_title(
                    current_traj_data["title"], fontsize=plot_options.title_fontsize, pad=plot_options.title_pad,
                )
            ax.grid(True, alpha=0.3)
            if current_traj_data.get("xlabel"):
                ax.set_xlabel(current_traj_data["xlabel"])
            if current_traj_data.get("ylabel"):
                ax.set_ylabel(current_traj_data["ylabel"])
            if current_traj_data.get("zlabel") and hasattr(ax, "set_zlabel"):
                ax.set_zlabel(current_traj_data["zlabel"])
            if current_traj_data.get("tick_nbins"):
                nbins = current_traj_data["tick_nbins"]
                ax.xaxis.set_major_locator(MaxNLocator(nbins=nbins))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins))
                if hasattr(ax, "zaxis"):
                    ax.zaxis.set_major_locator(MaxNLocator(nbins=nbins))

            opt_vals_path = f"trajectory_data.{traj_group_name}.{traj_name}.opt_vals.values"
            nl_vals_path = f"trajectory_data.{traj_group_name}.{traj_name}.nl_vals.values"

            traj_type = current_traj_data.type

            per_traj_show = current_traj_data.get("show_iters", None)
            if per_traj_show is True:
                traj_iters = slice(1, None)
            elif per_traj_show is False:
                traj_iters = [-1]
            else:
                traj_iters = iters

            if traj_type == "spatial":
                plot_spatial_trajectories(
                    PLTS,
                    method,
                    run,
                    traj_iters,
                    current_traj_data,
                    nl_vals_path,
                    opt_vals_path,
                    ax,
                    pens,
                    skip_init=skip_init,
                )
            elif traj_type == "time_series":
                plot_time_series_trajectories(
                    PLTS,
                    method,
                    run,
                    traj_iters,
                    current_traj_data,
                    nl_vals_path,
                    opt_vals_path,
                    ax,
                    pens,
                    skip_init=skip_init,
                )


def plot_spatial_trajectories(
    PLTS: SCVXPLOTS,
    method: str,
    run: int,
    iters: list[int],
    current_traj_data: AttrDict,
    nl_vals_path: str,
    opt_vals_path: str,
    ax: Axes,
    pens: AttrDict,
    skip_init: bool = False,
) -> None:
    """Plot 2-D or 3-D spatial trajectory (initial guess, optimal, nonlinear propagation)."""
    dim = current_traj_data.opt_vals["values"].shape[1]

    if iters != [-1]:
        if dim == 3:
            ins_nl = {
                "label": "Nonlinear Propagation",
                "x": nl_vals_path,
                "x_idx": 0,
                "y": nl_vals_path,
                "y_idx": 1,
                "z": nl_vals_path,
                "z_idx": 2,
                "method": method,
                "run": run,
                "iters": iters,
                "legend": "legend1",
            }
            PLTS.addPlot3D(ax, pen=pens.itr_nl, ins=ins_nl)

            ins_opt = {
                "label": "Optimal Soltution",
                "x": opt_vals_path,
                "x_idx": 0,
                "y": opt_vals_path,
                "y_idx": 1,
                "z": opt_vals_path,
                "z_idx": 2,
                "method": method,
                "run": run,
                "iters": iters,
                "legend": "legend1",
            }
            PLTS.addPlot3D(ax, pen=pens.itr_opt, ins=ins_opt)
        else:
            ins_nl = {
                "label": "Nonlinear Propagation",
                "x": nl_vals_path,
                "x_idx": 0,
                "y": nl_vals_path,
                "y_idx": 1,
                "method": method,
                "run": run,
                "iters": iters,
                "legend": "legend1",
            }
            PLTS.addPlot2D(ax, pen=pens.itr_nl, ins=ins_nl)

            ins_opt = {
                "label": "Optimal Soltution",
                "x": opt_vals_path,
                "x_idx": 0,
                "y": opt_vals_path,
                "y_idx": 1,
                "method": method,
                "run": run,
                "iters": iters,
                "legend": "legend1",
            }
            PLTS.addPlot2D(ax, pen=pens.itr_opt, ins=ins_opt)

    if not skip_init:
        init_nl_vals_path = nl_vals_path.replace("nl_vals", "init_nl_vals")

    if dim == 3:
        if not skip_init:
            ins_init = {
                "label": "Initial Guess",
                "x": init_nl_vals_path,
                "x_idx": 0,
                "y": init_nl_vals_path,
                "y_idx": 1,
                "z": init_nl_vals_path,
                "z_idx": 2,
                "method": method,
                "run": run,
                "iters": [-1],
                "legend": "legend1",
            }
            PLTS.addPlot3D(ax, pen=pens.init, ins=ins_init)

        ins_nl = {
            "label": "Nonlinear Propagation",
            "x": nl_vals_path,
            "x_idx": 0,
            "y": nl_vals_path,
            "y_idx": 1,
            "z": nl_vals_path,
            "z_idx": 2,
            "method": method,
            "run": run,
            "iters": [-1],
            "legend": "legend1",
        }
        PLTS.addPlot3D(ax, pen=pens.nl, ins=ins_nl)

        ins_opt = {
            "label": "Optimal Soltution",
            "x": opt_vals_path,
            "x_idx": 0,
            "y": opt_vals_path,
            "y_idx": 1,
            "z": opt_vals_path,
            "z_idx": 2,
            "method": method,
            "run": run,
            "iters": [-1],
            "legend": "legend1",
        }
        PLTS.addPlot3D(ax, pen=pens.opt, ins=ins_opt)
    else:
        if not skip_init:
            ins_init = {
                "label": "Initial Guess",
                "x": init_nl_vals_path,
                "x_idx": 0,
                "y": init_nl_vals_path,
                "y_idx": 1,
                "method": method,
                "run": run,
                "iters": [-1],
                "legend": "legend1",
            }
            PLTS.addPlot2D(ax, pen=pens.init, ins=ins_init)

        ins_nl = {
            "label": "Nonlinear Propagation",
            "x": nl_vals_path,
            "x_idx": 0,
            "y": nl_vals_path,
            "y_idx": 1,
            "method": method,
            "run": run,
            "iters": [-1],
            "legend": "legend1",
        }
        PLTS.addPlot2D(ax, pen=pens.nl, ins=ins_nl)

        ins_opt = {
            "label": "Optimal Soltution",
            "x": opt_vals_path,
            "x_idx": 0,
            "y": opt_vals_path,
            "y_idx": 1,
            "method": method,
            "run": run,
            "iters": [-1],
            "legend": "legend1",
        }
        PLTS.addPlot2D(ax, pen=pens.opt, ins=ins_opt)

    plot_markers(ax, current_traj_data, dim)
    plot_quivers(ax, current_traj_data, dim)


def plot_quivers(ax: Axes, traj_data: AttrDict, dim: int) -> None:
    """Overlay quiver arrows defined in traj_data.opt_vals.quivers onto ax."""
    quivers = traj_data.opt_vals.get("quivers", [])
    if not quivers:
        return
    origins = traj_data.opt_vals["values"]
    for qdata in quivers:
        cfg = qdata["config"]
        dirs = qdata["dirs"]
        stride = int(cfg.get("stride", 1))
        scale = float(cfg.get("scale", 1.0))
        color = cfg.get("color", [0.2, 0.2, 0.2])
        alpha = float(cfg.get("alpha", 0.8))
        lw = float(cfg.get("linewidth", 1.5))
        negate = bool(cfg.get("negate", False))

        idx = np.arange(0, len(origins), stride)
        o = origins[idx]
        d = dirs[idx] * scale * (-1.0 if negate else 1.0)

        # build NaN-separated segments so one ax.plot call draws all lines
        nan_col = np.full((len(idx), 1), np.nan)
        tips = o + d
        # interleave: base, tip, NaN for each segment → shape (3*N, ndim)
        segs = np.concatenate(
            [np.stack([o, tips, np.full_like(o, np.nan)], axis=1).reshape(-1, o.shape[1])],
            axis=0,
        )

        if dim == 3:
            ax.plot(segs[:, 0], segs[:, 1], segs[:, 2], color=color, alpha=alpha, linewidth=lw, linestyle="-")
        else:
            ax.plot(segs[:, 0], segs[:, 1], color=color, alpha=alpha, linewidth=lw, linestyle="-")


MARKER_DEFAULTS = {
    "marker": "*",
    "color": [0.8, 0.0, 0.0],
    "size": 80,
    "edgecolor": "k",
    "edgewidth": 0.4,
    "zorder": 10,
    "fontsize": 7,
    "text_offset": [0.0, 0.0],
}


def plot_markers(ax: Axes, traj_data: AttrDict, dim: int) -> None:
    """Scatter user-defined markers from traj_data.markers onto ax with optional text labels."""
    markers = traj_data.get("markers", None)
    if not markers:
        return

    for m in markers:
        xy = m["xy"]
        label = m.get("label", None)

        mkr = m.get("marker", MARKER_DEFAULTS["marker"])
        color = m.get("color", MARKER_DEFAULTS["color"])
        sz = m.get("size", MARKER_DEFAULTS["size"])
        ec = m.get("edgecolor", MARKER_DEFAULTS["edgecolor"])
        ew = m.get("edgewidth", MARKER_DEFAULTS["edgewidth"])
        zo = m.get("zorder", MARKER_DEFAULTS["zorder"])
        fs = m.get("fontsize", MARKER_DEFAULTS["fontsize"])
        t_off = m.get("text_offset", MARKER_DEFAULTS["text_offset"])

        if dim == 3 and len(xy) >= 3:
            ax.scatter(xy[0], xy[1], xy[2], marker=mkr, s=sz, c=[color], edgecolors=ec, linewidths=ew, zorder=zo)
            if label:
                ax.text(
                    xy[0] + t_off[0],
                    xy[1] + t_off[1],
                    xy[2] + (t_off[2] if len(t_off) > 2 else 0),
                    label,
                    fontsize=fs,
                    ha="left",
                    va="bottom",
                )
        else:
            ax.scatter(xy[0], xy[1], marker=mkr, s=sz, c=[color], edgecolors=ec, linewidths=ew, zorder=zo)
            if label:
                ax.annotate(
                    label,
                    (xy[0], xy[1]),
                    textcoords="offset points",
                    xytext=(t_off[0] + 4, t_off[1] + 4),
                    fontsize=fs,
                    ha="left",
                    va="bottom",
                )


def plot_time_series_trajectories(
    PLTS: SCVXPLOTS,
    method: str,
    run: int,
    iters: slice[int| Any, int | Any, int | Any] | list[int],
    current_traj_data: AttrDict,
    nl_vals_path: str,
    opt_vals_path: str,
    ax: Axes,
    pens: AttrDict,
    skip_init: bool = False,
) -> None:
    """Plot time-series trajectory (initial guess, optimal, nonlinear propagation) with constraint limit lines."""
    if iters != [-1]:
        ins_nl = {
            "label": "Nonlinear Propagation",
            "x": "t_nl",
            "y": nl_vals_path,
            "method": method,
            "run": run,
            "iters": iters,
            "legend": "legend1",
        }
        PLTS.addPlot2D(ax, pen=pens.itr_nl, ins=ins_nl)

        ins_opt = {
            "label": "Optimal Soltution",
            "x": "t_opt",
            "y": opt_vals_path,
            "method": method,
            "run": run,
            "iters": iters,
            "legend": "legend1",
        }
        PLTS.addPlot2D(ax, pen=pens.itr_opt, ins=ins_opt)

    if not skip_init:
        init_nl_vals_path = nl_vals_path.replace("nl_vals", "init_nl_vals")
        ins_init = {
            "label": "Initial Guess",
            "x": "t_init_nl",
            "y": init_nl_vals_path,
            "method": method,
            "run": run,
            "iters": [-1],
            "legend": "legend1",
        }
        PLTS.addPlot2D(ax, pen=pens.init, ins=ins_init)

    ins_nl = {
        "label": "Nonlinear Propagation",
        "x": "t_nl",
        "y": nl_vals_path,
        "method": method,
        "run": run,
        "iters": [-1],
        "legend": "legend1",
    }
    PLTS.addPlot2D(ax, pen=pens.nl, ins=ins_nl)

    ins_opt = {
        "label": "Optimal Soltution",
        "x": "t_opt",
        "y": opt_vals_path,
        "method": method,
        "run": run,
        "iters": [-1],
        "legend": "legend1",
    }
    PLTS.addPlot2D(ax, pen=pens.opt, ins=ins_opt)

    limits = current_traj_data.nl_vals.get("limits", {})
    if limits:
        for val in [limits.get("upper"), limits.get("lower")]:
            if val is not None:
                if isinstance(val, np.ndarray):
                    t_nl = np.array(PLTS.data[method]["runs"][run]["iters"])[-1]["t_nl"]
                    ax.plot(t_nl[: len(val)], val, color="k", ls="--", lw=1, alpha=0.5)
                else:
                    ax.axhline(val, color="k", ls="--", lw=1, alpha=0.5)


def create_grid(num_groups: int, cfg: dict | None = None) -> dict:
    """Compute normalized axes bounding boxes for a subplot grid.

    Arranges num_groups panels in a grid layout using margins, gap, and optional
    width_ratios from cfg. Returns a dict mapping subplot index to [x, y, w, h].
    """
    if cfg is None:
        cfg = {}

    if num_groups == 3:
        num_columns = 3
        num_rows = 1
    else:
        num_columns = int(np.ceil(np.sqrt(num_groups)))
        num_rows = int(np.ceil(num_groups / num_columns))

    gap_x = cfg.get("grid_gap_x", plot_options.grid_gap_x)
    gap_y = cfg.get("grid_gap_y", plot_options.grid_gap_y)
    margins = cfg.get("margins", plot_options.margins)
    margin_left, margin_right, margin_top, margin_bottom = margins

    usable_w = 1.0 - margin_left - margin_right
    usable_h = 1.0 - margin_top - margin_bottom

    width_ratios = cfg.get("width_ratios")

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
