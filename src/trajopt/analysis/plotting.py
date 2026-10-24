import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from trajopt.utils.tools import AttrDict, recursive_attrdict, resolve_function_from_string

plt.rcParams["text.usetex"] = False
plt.rcParams.update({
    'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 10,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'lines.linewidth': 1.0,
    'axes.formatter.useoffset': False, 'axes.formatter.limits': [-1, 3],
    'path.simplify': True, 'path.simplify_threshold': 0.1,
})

def _result_segments(result):
    from trajopt.outputs.results import SolveResult
    if not isinstance(result, SolveResult):
        raise TypeError("Convergence plotting requires a SolveResult")
    return result.segment_data


plot_options = AttrDict({
    'figsize':     (12, 3.5),
    'save_dpi':    300,
    'grid_gap_x':  0.06,
    'grid_gap_y':  0.12,
    'margins':     [0.08, 0.02, 0.08, 0.14],
    'title_fontsize': 10,
    'title_pad':   4,
})

pens = recursive_attrdict({
    'init':    {'frgba': [0,0,0,.1], 'lrgba': [0,0,0,1.],   'lw': 1, 'ls': '--', 'msty': '',  'msz': 3},
    'nl':      {'frgba': [0,0,0,.1], 'lrgba': [1,0,0,1.],   'lw': 2, 'ls': '-',  'msty': '',  'msz': 3},
    'opt':     {'frgba': [0,0,0,.1], 'lrgba': [0,0,1,1.],   'lw': 1, 'ls': '',   'msty': 'o', 'msz': 1},
    'itr_opt': {'frgba': [0,0,0,.1], 'lrgba': [.7,0,.3,.2], 'lw': 1, 'ls': '',   'msty': 'o', 'msz': 3},
    'itr_nl':  {'frgba': [0,0,0,.1], 'lrgba': [.7,0,.3,.4], 'lw': 1, 'ls': '-',  'msty': '',  'msz': 3},
})

METHOD_CMAP = plt.cm.rainbow

def _method_colors(n):
    """Sample *n* maximally-spaced vibrant colors from METHOD_CMAP."""
    if n == 1:
        positions = [0.5]
    else:
        positions = np.linspace(0.05, 0.95, n)
    return [list(METHOD_CMAP(p)[:3]) for p in positions]

def _method_pens(color):
    return AttrDict({
        'nl':  AttrDict({'lrgba': color + [1.0], 'lw': 1, 'ls': '-',  'msty': '',  'msz': 3}),
        'opt': AttrDict({'lrgba': color + [1.0], 'lw': 1, 'ls': '',   'msty': 'o', 'msz': 3}),
    })

MARKER_DEFAULTS = {
    'marker': '*', 'color': [0.8, 0.0, 0.0], 'size': 80,
    'edgecolor': 'k', 'edgewidth': 0.4, 'zorder': 10, 'fontsize': 7, 'text_offset': [0.0, 0.0]
}


def save_figures(figs, save_dir, *, format="pdf", dpi=None):
    """Write each figure to save_dir/<name>.<format> and return the paths."""
    dpi = plot_options.save_dpi if dpi is None else dpi
    os.makedirs(save_dir, exist_ok=True)
    paths = []
    for name, fig in figs.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            fig.tight_layout()
        path = os.path.join(save_dir, f"{name}.{format}")
        fig.savefig(path, dpi=dpi, pad_inches=0.02)
        paths.append(path)
    print(f"Saved {len(figs)} figures to {save_dir}/")
    return paths


def show_figures(figs=None):
    """Display the figures."""
    plt.show()


def plot_result(result, evaluation=None, *, analysis_config=None, save=False,
                show=False, save_dir=None, format="pdf"):
    """Plot the named segments and available diagnostics from a solve result."""
    from trajopt.analysis.validation import EvaluationResult, evaluate
    from trajopt.outputs.results import SolveResult

    if not isinstance(result, SolveResult):
        raise TypeError("plot_result requires a SolveResult")
    if evaluation is not None and not isinstance(evaluation, EvaluationResult):
        raise TypeError("evaluation must be an EvaluationResult")
    analysis_config = AttrDict((evaluation.protocol if evaluation is not None else {})
                              if analysis_config is None else analysis_config)
    evaluation = evaluation if evaluation is not None else evaluate(result, analysis_config)
    if evaluation.solve_result is not result:
        raise ValueError("Evaluation must belong to the supplied solve result")
    figures = {}
    multi = len(evaluation.segments) > 1
    for name, segment in evaluation.segments.items():
        spec = result.problem.segments[name]
        for key, figure in _build_segment(
                spec.build(), spec.config.get("outputs", {}), segment.iterates, analysis_config).items():
            figures[f"{name}_{key}" if multi else key] = figure
    figures.update(_diagnostic_figures(result, analysis_config))
    return _finish_figures(figures, save, show, save_dir or os.path.join("plots", "standalone"), format)


def _diagnostic_figures(result, options):
    figures = {}
    for option, function, key in (
        ("show_convergence", convergence_plots, "convergence"),
        ("show_weights", convergence_weight_plots, "convergence_weights"),
        ("show_weights", convergence_weight_mean_plots, "convergence_weights_mean"),
        ("show_vb", convergence_vb_plots, "convergence_vb"),
    ):
        if options.get(option, False):
            value = function(result, save=False)
            if value is not None:
                figures.update(value if isinstance(value, dict) else {key: value})
    return figures


def plot_experiment(experiment, *, case=None, analysis_config=None, save=False,
                    show=False, save_dir=None, format="pdf"):
    """Compare variants for the selected case. Use the nominal case by default."""
    from trajopt.analysis.validation import evaluate
    from trajopt.analysis.experiments import ExperimentResult

    if not isinstance(experiment, ExperimentResult):
        raise TypeError("plot_experiment requires an ExperimentResult")
    case = experiment.primary_run().case if case is None else case
    runs = {run.variant: run for run in experiment.runs if run.case == case}
    if not runs:
        raise ValueError(f"Unknown experiment case {case!r}")
    if analysis_config is None:
        analysis_config = next(iter(runs.values())).config.get("analysis", {})
    evaluations = {name: (run.evaluation if run.evaluation is not None else
                         evaluate(run.solve_result, analysis_config))
                   for name, run in runs.items() if run.solve_result is not None}
    names = dict.fromkeys(name for report in evaluations.values() for name in report.segments)
    figures = {}
    for name in names:
        selected = {variant: report.segments[name].iterates
                    for variant, report in evaluations.items() if name in report.segments
                    and report.segments[name].iterates}
        if not selected:
            continue
        result = evaluations[next(iter(selected))].solve_result
        spec = result.problem.segments[name]
        problem, outputs = spec.build(), spec.config.get("outputs", {})
        segment_figures = (_build_segment(problem, outputs, next(iter(selected.values())), analysis_config)
                           if len(selected) == 1 else _build_comparison(problem, outputs, selected))
        for key, figure in segment_figures.items():
            figures[f"{name}_{key}" if len(names)>1 else key] = figure
    for variant, run in runs.items():
        if run.solve_result is not None:
            for key, figure in _diagnostic_figures(run.solve_result, analysis_config).items():
                figures[f"{variant}_{key}" if len(runs)>1 else key] = figure
    return _finish_figures(figures, save, show, save_dir or os.path.join("plots", "experiment"), format)


def _finish_figures(figures, save, show, directory, format):
    if save:
        save_figures(figures, directory, format=format)
    if show:
        show_figures(figures)
    return figures


#: figure name used when a declared output does not name a group
DEFAULT_GROUP = "outputs"


def _group_outputs(outputs, declared):
    """Group the declared outputs by figure, one figure per group and one subplot per output."""
    grouped = {}
    for name, output in outputs.items():
        if name not in declared:
            continue
        grouped.setdefault(output.meta.group or DEFAULT_GROUP, {})[name] = output
    return grouped


def _build_segment(problem, traj_configs, iters_all, analysis_cfg):
    """Construct the standalone trajectory figures."""
    show_iters = analysis_cfg.get("show_iters", False)
    if not iters_all:
        return {}
    last_iter      = iters_all[-1]
    traj_data      = _group_outputs(last_iter.outputs, traj_configs)
    first_phase = problem
    fcns           = first_phase.fcns

    figs, axs = {}, {}
    for group_name, group_data in traj_data.items():
        figsize = plot_options.figsize
        pad_3d  = 0.08
        grid    = _create_grid(len(group_data))
        is_3d   = {i: (d.meta.type == "spatial" and d.opt.shape[1] == 3)
                   for i, d in enumerate(group_data.values())}

        fig = plt.figure(figsize=figsize)
        axs[group_name] = {}
        for idx, rect in grid.items():
            if is_3d.get(idx):
                x, y, w, h = rect
                axs[group_name][idx] = fig.add_axes([x - pad_3d, y, w + pad_3d, h], projection='3d')
            else:
                axs[group_name][idx] = fig.add_axes(rect)
        figs[group_name] = fig

    iters_to_show = iters_all[1:] if show_iters else []

    for group_name, group_data in traj_data.items():
        for i, (traj_name, output) in enumerate(group_data.items()):
            ax = axs[group_name][i]
            _setup_ax(ax, output.meta)
            if output.meta.type == "spatial":
                _plot_spatial(ax, output, iters_to_show, last_iter)
            else:
                _plot_time_series(ax, output, iters_to_show, last_iter)

    for group_name, group_data in traj_data.items():
        for i, (traj_name, output) in enumerate(group_data.items()):
            if output.meta.type == "spatial":
                ax       = axs[group_name][i]
                dim      = output.opt.shape[1]
                traj_cfg = traj_configs.get(traj_name, output.meta)

                limits_opt_only = traj_cfg.get("limits_opt_only", False)
                if limits_opt_only:
                    xlim, ylim = ax.get_xlim(), ax.get_ylim()
                    zlim = ax.get_zlim() if hasattr(ax, 'get_zlim') else None

                _plot_overlays(ax, traj_cfg, dim, first_phase.params, fcns)

                if limits_opt_only:
                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)
                    if zlim is not None and hasattr(ax, 'set_zlim'):
                        ax.set_zlim(zlim)

    for group_name, group_data in traj_data.items():
        trigger_time = _find_trigger_time(group_data, last_iter)
        for i, (traj_name, output) in enumerate(group_data.items()):
            ax   = axs[group_name][i]
            vals = output.opt
            if output.meta.type == "spatial":
                traj_cfg = traj_configs.get(traj_name, {})
                equal_aspect = bool(traj_cfg.get("equal_aspect", False))
                all_vals = np.vstack([vals, output.init_guess])
                all_vals = _include_quiver_extents(all_vals, output)
                _set_limits_from_data(ax, all_vals, equal_aspect=equal_aspect)
                if traj_cfg.get("xlim") is not None:
                    ax.set_xlim(traj_cfg["xlim"])
                if traj_cfg.get("ylim") is not None:
                    ax.set_ylim(traj_cfg["ylim"])
            else:
                t = last_iter.t_opt
                _set_time_series_limits(ax, t[:vals.shape[0]], vals,
                                        limits=output.limits or {})
                if trigger_time is not None:
                    ax.axvline(trigger_time, color='gray', ls='--', lw=0.8, alpha=0.7, zorder=1)

    legend_entries = [('init', 'Initial Guess'), ('opt', 'Optimal'), ('nl', 'Nonlinear')]
    if show_iters:
        legend_entries.append(('itr_nl', 'Iterations'))
    handles = [_legend_handle(name, label) for name, label in legend_entries]
    for fig in figs.values():
        fig.axes[0].legend(handles=handles, loc='best', fontsize=8, framealpha=0.8)

    return figs


def _build_comparison(problem, traj_configs, iterates):
    """Overlay the trajectories of several methods on shared axes."""
    methods = list(iterates)
    if not methods:
        return {}
    ref_last = iterates[methods[0]][-1]
    ref_traj_data = _group_outputs(ref_last.outputs, traj_configs)
    for history in iterates.values():
        for group in ref_traj_data.values():
            for name, reference in group.items():
                output = history[-1].outputs.get(name)
                if (output is None or output.opt.shape[1:] != reference.opt.shape[1:]
                        or output.meta.units != reference.meta.units):
                    raise ValueError(f"Cannot compare output {name!r} with incompatible channels or units")

    first_phase = problem
    fcns = first_phase.fcns

    figs, axs = {}, {}
    for group_name, group_data in ref_traj_data.items():
        grid = _create_grid(len(group_data))
        is_3d = {i: (d.meta.type == "spatial" and d.opt.shape[1] == 3)
                 for i, d in enumerate(group_data.values())}

        fig = plt.figure(figsize=plot_options.figsize)
        axs[group_name] = {}
        pad_3d = 0.08
        for idx, rect in grid.items():
            if is_3d.get(idx):
                x, y, w, h = rect
                axs[group_name][idx] = fig.add_axes([x - pad_3d, y, w + pad_3d, h], projection='3d')
            else:
                axs[group_name][idx] = fig.add_axes(rect)
        figs[group_name] = fig

    for group_name, group_data in ref_traj_data.items():
        for i, (traj_name, output) in enumerate(group_data.items()):
            _setup_ax(axs[group_name][i], output.meta)

    all_spatial_vals = {}
    all_ts_ranges = {}
    colors = _method_colors(len(methods))
    for m_idx, method_name in enumerate(methods):
        color = colors[m_idx]
        mp = _method_pens(color)
        m_last = iterates[method_name][-1]
        m_traj_data = _group_outputs(m_last.outputs, traj_configs)

        for group_name, group_data in m_traj_data.items():
            for i, (traj_name, output) in enumerate(group_data.items()):
                ax = axs[group_name][i]
                key = (group_name, i)

                if output.meta.type == "spatial":
                    dim = output.opt.shape[1]
                    _is_3d = dim == 3

                    def _unpack(v, d=_is_3d):
                        return (v[:, 0], v[:, 1], v[:, 2]) if d else (v[:, 0], v[:, 1])

                    coords_nl = _unpack(output.nl_prop)
                    _draw(ax, *coords_nl[:2], mp.nl, z=coords_nl[2] if _is_3d else None)
                    coords_opt = _unpack(output.opt)
                    _draw(ax, *coords_opt[:2], mp.opt, z=coords_opt[2] if _is_3d else None)

                    combined = np.vstack([output.opt, output.nl_prop])
                    prev = all_spatial_vals.get(key)
                    all_spatial_vals[key] = np.vstack([prev, combined]) if prev is not None else combined
                else:
                    t_nl = m_last.t_nl
                    t_opt = m_last.t_opt
                    nl_v = output.nl_prop.squeeze()
                    opt_v = output.opt.squeeze()
                    _draw(ax, t_nl,  nl_v,  mp.nl)
                    _draw(ax, t_opt, opt_v, mp.opt)

                    t_all = np.concatenate([t_nl, t_opt])
                    v_all = np.concatenate([nl_v.ravel(), opt_v.ravel()])
                    if key not in all_ts_ranges:
                        all_ts_ranges[key] = {"t_min": t_all.min(), "t_max": t_all.max(),
                                              "v_min": np.nanmin(v_all), "v_max": np.nanmax(v_all)}
                    else:
                        r = all_ts_ranges[key]
                        r["t_min"] = min(r["t_min"], t_all.min())
                        r["t_max"] = max(r["t_max"], t_all.max())
                        r["v_min"] = min(r["v_min"], np.nanmin(v_all))
                        r["v_max"] = max(r["v_max"], np.nanmax(v_all))

    for group_name, group_data in ref_traj_data.items():
        trigger_time = _find_trigger_time(group_data, ref_last)
        for i, (traj_name, output) in enumerate(group_data.items()):
            ax = axs[group_name][i]
            key = (group_name, i)
            if output.meta.type == "spatial":
                traj_cfg = traj_configs.get(traj_name, output.meta)
                dim = output.opt.shape[1]
                _plot_overlays(ax, traj_cfg, dim, first_phase.params, fcns)

                if key in all_spatial_vals:
                    equal_aspect = bool(traj_cfg.get("equal_aspect", False))
                    _set_limits_from_data(ax, all_spatial_vals[key], equal_aspect=equal_aspect)
                    if traj_cfg.get("xlim") is not None:
                        ax.set_xlim(traj_cfg["xlim"])
                    if traj_cfg.get("ylim") is not None:
                        ax.set_ylim(traj_cfg["ylim"])
            elif key in all_ts_ranges:
                r = all_ts_ranges[key]
                t_arr = np.array([r["t_min"], r["t_max"]])
                v_arr = np.array([r["v_min"], r["v_max"]])
                _set_time_series_limits(ax, t_arr, v_arr,
                                        limits=output.limits or {})
                if trigger_time is not None:
                    ax.axvline(trigger_time, color='gray', ls='--', lw=0.8, alpha=0.7, zorder=1)

    handles = []
    for m_idx, method_name in enumerate(methods):
        color = colors[m_idx]
        handles.append(Line2D([], [], color=color, lw=2, ls='-', label=f'{method_name} (nl)'))
        handles.append(Line2D([], [], color=color, lw=0, marker='o', ms=3, label=f'{method_name} (opt)'))
    for fig in figs.values():
        fig.axes[0].legend(handles=handles, loc='best', fontsize=7, framealpha=0.8)

    return figs


def _setup_ax(ax, traj):
    if traj.get("title"):
        ax.set_title(traj["title"], fontsize=plot_options.title_fontsize, pad=plot_options.title_pad)
    if traj.get("xlabel"): ax.set_xlabel(traj["xlabel"])
    if traj.get("ylabel"): ax.set_ylabel(traj["ylabel"])
    if traj.get("zlabel") and hasattr(ax, 'set_zlabel'):
        ax.set_zlabel(traj["zlabel"])
    ax.grid(True, alpha=0.3)
    if traj.get("tick_nbins"):
        nbins = traj["tick_nbins"]
        ax.xaxis.set_major_locator(MaxNLocator(nbins=nbins))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins))
        if hasattr(ax, 'zaxis'):
            ax.zaxis.set_major_locator(MaxNLocator(nbins=nbins))


def _draw(ax, x, y, pen, z=None, n_iters=1, i=0):
    first_frac = pen.get('first_frac', 0.2)
    if n_iters > 1:
        alpha = pen.lrgba[3] * (first_frac + (1 - first_frac) * i / max(n_iters - 1, 1))
    else:
        alpha = pen.lrgba[3]

    kwargs = dict(
        color     = pen.lrgba[:3],
        alpha     = alpha,
        linewidth = pen.lw,
        linestyle = pen.ls or 'None',
        marker    = pen.msty or None,
        markersize= pen.msz,
    )
    if z is not None:
        ax.plot(x, y, z, **kwargs)
    else:
        ax.plot(x, y, **kwargs)


def _plot_spatial(ax, traj, iters_to_show, last_iter):
    dim   = traj.opt.shape[1]
    is_3d = dim == 3

    def unpack(v):
        return (v[:, 0], v[:, 1], v[:, 2]) if is_3d else (v[:, 0], v[:, 1])

    name = traj.meta.name
    n    = len(iters_to_show)
    for i, it in enumerate(iters_to_show):
        t = it.outputs.get(name)
        if t is None: continue

        coords = unpack(t.nl_prop)
        _draw(ax, *coords[:2], pens.itr_nl,  z=coords[2] if is_3d else None, n_iters=n, i=i)
        coords = unpack(t.opt)
        _draw(ax, *coords[:2], pens.itr_opt, z=coords[2] if is_3d else None, n_iters=n, i=i)

    coords = unpack(traj.init_guess)
    _draw(ax, *coords[:2], pens.init, z=coords[2] if is_3d else None)
    coords = unpack(traj.nl_prop)
    _draw(ax, *coords[:2], pens.nl,   z=coords[2] if is_3d else None)
    coords = unpack(traj.opt)
    _draw(ax, *coords[:2], pens.opt,  z=coords[2] if is_3d else None)

    _plot_markers(ax, traj.meta, dim)
    _plot_quivers(ax, traj, dim)


def _plot_time_series(ax, traj, iters_to_show, last_iter):
    name = traj.meta.name
    n    = len(iters_to_show)
    for i, it in enumerate(iters_to_show):
        t = it.outputs.get(name)
        if t is None: continue

        _draw(ax, it.t_nl,  t.nl_prop.squeeze(), pens.itr_nl,  n_iters=n, i=i)
        _draw(ax, it.t_opt, t.opt.squeeze(),     pens.itr_opt, n_iters=n, i=i)

    _draw(ax, last_iter.t_init_nl, traj.init_guess.squeeze(), pens.init)
    _draw(ax, last_iter.t_nl,      traj.nl_prop.squeeze(),    pens.nl)
    _draw(ax, last_iter.t_opt,     traj.opt.squeeze(),        pens.opt)

    limits = traj.limits or {}
    upper = limits.get("upper")
    lower = limits.get("lower")
    for val in (upper, lower):
        if val is None:
            continue
        if isinstance(val, np.ndarray):
            t = last_iter.t_nl
            ax.plot(t[:len(val)], val, color='k', ls='--', lw=1, alpha=0.5)
        else:
            ax.axhline(val, color='k', ls='--', lw=1, alpha=0.5)


def _plot_markers(ax, traj, dim):
    for m in (traj.get("markers") or []):
        xy  = m["xy"]
        cfg = {k: m.get(k, MARKER_DEFAULTS[k]) for k in MARKER_DEFAULTS}

        ax.scatter(*xy[:dim], marker=cfg['marker'], s=cfg['size'], c=[cfg['color']],
                   edgecolors=cfg['edgecolor'], linewidths=cfg['edgewidth'], zorder=cfg['zorder'])

        if not m.get("label"): continue
        off = cfg['text_offset']
        if dim == 3 and len(xy) >= 3:
            z_off = off[2] if len(off) > 2 else 0
            ax.text(xy[0]+off[0], xy[1]+off[1], xy[2]+z_off,
                    m["label"], fontsize=cfg['fontsize'], ha='left', va='bottom')
        else:
            ax.annotate(m["label"], (xy[0], xy[1]),
                        textcoords="offset points", xytext=(off[0]+4, off[1]+4),
                        fontsize=cfg['fontsize'], ha='left', va='bottom')


def _plot_quivers(ax, traj, dim):
    for q in (traj.quivers or []):
        cfg    = dict(q["config"]) if isinstance(q["config"], tuple) else q["config"]
        stride = int(cfg.get("stride", 1))
        scale  = float(cfg.get("scale", 1.0)) * (-1.0 if cfg.get("negate") else 1.0)

        idx  = np.arange(0, len(traj.opt), stride)
        o    = traj.opt[idx]
        if q.get("origins") is not None:
            o = o + q["origins"][idx]
        d    = q["dirs"][idx] * scale
        centered = bool(cfg.get("centered", False))
        start = o - d / 2 if centered else o
        segs = np.stack([start, start + d, np.full_like(o, np.nan)], axis=1).reshape(-1, o.shape[1])

        ax.plot(*[segs[:, c] for c in range(dim)],
                color    = cfg.get("color", [0.2, 0.2, 0.2]),
                alpha    = float(cfg.get("alpha", 0.8)),
                linewidth= float(cfg.get("linewidth", 1.5)))


def _plot_overlays(ax, traj, dim, params, fcns=None):
    for name, cfg in (traj.get("overlays") or {}).items():
        fcn = resolve_function_from_string(cfg["fcn"], fcns=fcns)
        if fcn is None: continue
        pts = fcn(params, ax)
        ax.plot(*[pts[:, c] for c in range(dim)],
                color = cfg.get("color", [0, 0, 0]),
                ls    = cfg.get("ls", ':'),
                lw    = float(cfg.get("lw", 1.2)),
                alpha = float(cfg.get("alpha", 0.6)),
                zorder= 3)


def _legend_handle(pen_name, label):
    p = pens[pen_name]
    return Line2D([], [], color=p.lrgba[:3], alpha=p.lrgba[3], lw=p.lw,
                  ls=p.ls or 'None', marker=p.msty or None, markersize=p.msz, label=label)


def _create_grid(n, cfg=None):
    cfg      = cfg or {}
    num_cols = 3 if n == 3 else int(np.ceil(np.sqrt(n)))
    num_rows = int(np.ceil(n / num_cols))

    gap_x                    = cfg.get('grid_gap_x', plot_options.grid_gap_x)
    gap_y                    = cfg.get('grid_gap_y', plot_options.grid_gap_y)
    margin_l, margin_r, margin_t, margin_b = cfg.get('margins', plot_options.margins)

    cell_h       = (1.0 - margin_t - margin_b - (num_rows - 1) * gap_y) / num_rows
    width_ratios = cfg.get('width_ratios')
    grid         = {}

    if width_ratios is not None and len(width_ratios) == num_cols:
        usable_w   = 1.0 - margin_l - margin_r - (num_cols - 1) * gap_x
        col_widths = [(r / sum(width_ratios)) * usable_w for r in width_ratios]
        for row in range(num_rows):
            x = margin_l
            for col in range(num_cols):
                y = (1.0 - margin_t) - (row + 1) * cell_h - row * gap_y
                grid[row * num_cols + col] = [x, y, col_widths[col], cell_h]
                x += col_widths[col] + gap_x
    else:
        cell_w = (1.0 - margin_l - margin_r - (num_cols - 1) * gap_x) / num_cols
        for row in range(num_rows):
            for col in range(num_cols):
                x = margin_l + col * (cell_w + gap_x)
                y = (1.0 - margin_t) - (row + 1) * cell_h - row * gap_y
                grid[row * num_cols + col] = [x, y, cell_w, cell_h]

    return grid


def _padded_lim(lo, hi, margin=0.08):
    pad = margin * (hi - lo) if hi > lo else 0.1
    return lo - pad, hi + pad


def _set_limits_from_data(ax, vals, margin=0.08, equal_aspect=False):
    # an output missing from a phase is NaN over that stretch
    ax.set_xlim(*_padded_lim(np.nanmin(vals[:, 0]), np.nanmax(vals[:, 0]), margin))
    ax.set_ylim(*_padded_lim(np.nanmin(vals[:, 1]), np.nanmax(vals[:, 1]), margin))
    if vals.shape[1] >= 3 and hasattr(ax, 'set_zlim'):
        ax.set_zlim(*_padded_lim(np.nanmin(vals[:, 2]), np.nanmax(vals[:, 2]), margin))
        if equal_aspect:
            _set_equal_aspect_3d(ax)
    elif equal_aspect:
        ax.set_aspect('equal', adjustable='box')


def _set_equal_aspect_3d(ax):
    xlim = np.array(ax.get_xlim())
    ylim = np.array(ax.get_ylim())
    zlim = np.array(ax.get_zlim())

    spans = np.array([xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]])
    max_span = spans.max()

    ax.set_xlim(xlim.mean() - max_span / 2, xlim.mean() + max_span / 2)
    ax.set_ylim(ylim.mean() - max_span / 2, ylim.mean() + max_span / 2)
    ax.set_zlim(zlim.mean() - max_span / 2, zlim.mean() + max_span / 2)


def _set_time_series_limits(ax, t, vals, margin=0.08, limits=None):
    ax.set_xlim(*_padded_lim(t.min(), t.max(), margin))
    vmin, vmax = np.nanmin(vals), np.nanmax(vals)
    if limits:
        for lim in (limits.get("upper"), limits.get("lower")):
            if lim is None:
                continue
            if isinstance(lim, np.ndarray):
                vmin, vmax = min(vmin, np.nanmin(lim)), max(vmax, np.nanmax(lim))
            elif isinstance(lim, (int, float)):
                vmin, vmax = min(vmin, lim), max(vmax, lim)
    ax.set_ylim(*_padded_lim(vmin, vmax, margin))


def _include_quiver_extents(all_vals, traj):
    for q in (traj.quivers or []):
        cfg      = dict(q["config"]) if isinstance(q["config"], tuple) else q["config"]
        stride   = int(cfg.get("stride", 1))
        scale    = float(cfg.get("scale", 1.0)) * (-1.0 if cfg.get("negate") else 1.0)
        centered = bool(cfg.get("centered", False))

        origins = traj.opt[::stride]
        if q.get("origins") is not None:
            origins = origins + q["origins"][::stride]
        dirs = q["dirs"][::stride] * scale
        if centered:
            tips = np.vstack([origins - dirs / 2, origins + dirs / 2])
        else:
            tips = origins + dirs
        all_vals = np.vstack([all_vals, tips])
    return all_vals


def convergence_plots(result, save=True):
    return _convergence_figures(result, _convergence_plot, "convergence", save)


def _convergence_figures(source, renderer, prefix, save):
    segments = _result_segments(source)
    result = source
    if (renderer is not _convergence_plot and result.iterations > 1
            and result.method.config.get("history", "scalars") != "full"):
        warnings.warn("Only the final penalty snapshot was retained; use method.history='full' "
                      "for per-iteration penalty or buffer plots", stacklevel=2)
    multi = len(segments) > 1
    figures = {}
    for name, segment in segments.items():
        suffix = f"_{name}" if multi else ""
        figure = renderer(segment, suffix, save)
        if figure is not None:
            figures[prefix + suffix] = figure
    return figures if multi else next(iter(figures.values()), None)


def _penalty_tolerances(subprob):
    """Match saved buffer names to their tolerances, including named penalty parts."""
    tolerances = {}

    def collect(name, penalty):
        parts = penalty.get("parts")
        if parts is not None:
            for part_name, part in parts.items():
                collect(f"{name}_{part_name}", part)
        else:
            tolerances[name] = np.atleast_1d(penalty.eps)

    for constraint in subprob.constraints.values():
        if constraint.shape is not None:
            collect(constraint.name, constraint.penalties)
    return tolerances


def _convergence_plot(subprob, suffix, save=True):
    scalar = subprob.get("scalar_history", ())
    history = (subprob.iter_data_list if len(subprob.iter_data_list) >= len(scalar) + 1
               else scalar)
    iters = [it for it in history
             if it.get("iter_num", 0) > 0]
    if not iters:
        return None
    k       = np.array([it.iter_num for it in iters])
    dz      = [it.chk.dz for it in iters]
    dcost   = [it.chk.dcost for it in iters]
    costs   = [it.cost for it in iters]

    eps_by_name = _penalty_tolerances(subprob)
    vb_types  = list(iters[0].vb.keys()) if hasattr(iters[0], 'vb') and iters[0].vb else []

    vb_series = {}
    for ct in vb_types:
        eps_ct = eps_by_name.get(ct, np.atleast_1d(1.0))
        vb_series[ct] = [float(np.max(np.abs(it.vb[ct]) / eps_ct, initial=0.)) for it in iters]

    grid  = _create_grid(2)
    fig   = plt.figure(figsize=(14, 4.5))
    axes  = [fig.add_axes(grid[i]) for i in range(2)]

    markers = ['o', 's', '^', 'v', 'D', 'P', 'X', 'h']

    ax = axes[0]
    ax.semilogy(k, dz,    'o-', ms=3, label=r'$\|\delta x\|/\epsilon_x$')
    ax.semilogy(k, dcost, 's-', ms=3, label=r'$|\delta J|/\epsilon_J$')
    for i, ct in enumerate(vb_types):
        m = markers[(i + 2) % len(markers)]
        ax.semilogy(k, vb_series[ct], f'{m}-', ms=3, label=f'$\\|vb_{{\\mathrm{{{ct}}}}}\\|/\\epsilon$')
    ax.axhline(1.0, color='k', ls='--', lw=0.8)
    ax.set_xlabel('Iteration'); ax.set_ylabel('Normalized metric')
    ax.set_title('Convergence History', fontsize=plot_options.title_fontsize, pad=plot_options.title_pad)
    ax.legend(fontsize=6); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(k, costs, 'o-', ms=3, color=pens.opt.lrgba[:3])
    ax.set_xlabel('Iteration'); ax.set_ylabel('Cost')
    ax.set_title('Objective', fontsize=plot_options.title_fontsize, pad=plot_options.title_pad)
    ax.grid(True, alpha=0.3)

    if save:
        save_dir = os.path.join("plots", "standalone")
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, f"convergence{suffix}.pdf"), dpi=plot_options.save_dpi, pad_inches=0.02)

    return fig


def convergence_weight_plots(result, save=True):
    return _convergence_figures(result, _convergence_weight_plot, "convergence_weights", save)


def _convergence_weight_plot(subprob, suffix, save=True):
    """Plot every individual W/dual component (not just the mean) as its own thin line, one color per constraint."""
    iters = subprob.iter_data_list[1:]
    if not iters:
        return

    has_W    = hasattr(iters[0], 'W')    and iters[0].W
    has_dual = hasattr(iters[0], 'dual') and iters[0].dual
    has_split = hasattr(iters[0], 'W_p') and iters[0].W_p

    if not has_W and not has_dual and not has_split:
        return

    k = np.array([it.iter_num for it in iters])
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    all_types = list(iters[0].W.keys()) if has_W else list(iters[0].dual.keys())
    colors    = dict(zip(all_types, _method_colors(len(all_types)))) if all_types else {}

    def _plot_components(ax, key, ct, ls):
        """Plot every flattened component of it.<key>[ct] across iterations as its own thin line."""
        stacked = np.stack([np.ravel(np.abs(np.asarray(getattr(it, key)[ct]))) for it in iters])
        color = colors.get(ct, [0, 0, 0])
        for j in range(stacked.shape[1]):
            ax.semilogy(k, stacked[:, j], ls, color=color, lw=0.8, alpha=0.6)

    def _legend_proxy(ax, ct, label, ls='-'):
        ax.plot([], [], ls, color=colors.get(ct, [0, 0, 0]), label=label)

    if has_W:
        ax = axes[0]
        for ct in all_types:
            if has_split and ct in iters[0].W_p:
                _plot_components(ax, 'W_p', ct, '-')
                _plot_components(ax, 'W_m', ct, '--')
                _legend_proxy(ax, ct, f'{ct} (+)', '-')
                _legend_proxy(ax, ct, f'{ct} (-)', '--')
            else:
                _plot_components(ax, 'W', ct, '-')
                _legend_proxy(ax, ct, ct, '-')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('W')
        ax.set_title('Quadratic Penalty Weights', fontsize=plot_options.title_fontsize, pad=plot_options.title_pad)
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    if has_dual:
        ax = axes[1]
        dual_types = list(iters[0].dual.keys())
        for ct in dual_types:
            if has_split and ct in iters[0].dual_p:
                _plot_components(ax, 'dual_p', ct, '-')
                _plot_components(ax, 'dual_m', ct, '--')
                _legend_proxy(ax, ct, f'{ct} (+)', '-')
                _legend_proxy(ax, ct, f'{ct} (-)', '--')
            else:
                _plot_components(ax, 'dual', ct, '-')
                _legend_proxy(ax, ct, ct, '-')
        ax.set_xlabel('Iteration')
        ax.set_ylabel(r'$|\lambda|$')
        ax.set_title('Linear (Dual) Weights', fontsize=plot_options.title_fontsize, pad=plot_options.title_pad)
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    if save:
        save_dir = os.path.join("plots", "standalone")
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, f"convergence_weights{suffix}.pdf"), dpi=plot_options.save_dpi, pad_inches=0.02)

    return fig


def convergence_weight_mean_plots(result, save=True):
    return _convergence_figures(result, _convergence_weight_mean_plot, "convergence_weights_mean", save)


def _convergence_weight_mean_plot(subprob, suffix, save=True):
    iters = subprob.iter_data_list[1:]
    if not iters:
        return

    has_W    = hasattr(iters[0], 'W')    and iters[0].W
    has_dual = hasattr(iters[0], 'dual') and iters[0].dual
    has_split = hasattr(iters[0], 'W_p') and iters[0].W_p

    if not has_W and not has_dual and not has_split:
        return

    k = np.array([it.iter_num for it in iters])
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    if has_split:
        ax = axes[0]
        for ct in iters[0].W_p.keys():
            ax.semilogy(k, [np.mean(it.W_p[ct]) for it in iters], 'o-', ms=3, label=f'{ct} (+)')
            ax.semilogy(k, [np.mean(it.W_m[ct]) for it in iters], 's--', ms=3, label=f'{ct} (-)')
        for ct in iters[0].W.keys():
            if ct not in iters[0].W_p:
                ax.semilogy(k, [np.mean(it.W[ct]) for it in iters], 'o-', ms=3, label=ct)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Mean W')
        ax.set_title('Quadratic Penalty Weights', fontsize=plot_options.title_fontsize, pad=plot_options.title_pad)
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        ax = axes[1]
        for ct in iters[0].dual_p.keys():
            ax.semilogy(k, [np.mean(np.abs(it.dual_p[ct])) for it in iters], 'o-', ms=3, label=f'{ct} (+)')
            ax.semilogy(k, [np.mean(np.abs(it.dual_m[ct])) for it in iters], 's--', ms=3, label=f'{ct} (-)')
        for ct in iters[0].dual.keys():
            if ct not in iters[0].dual_p:
                ax.semilogy(k, [np.mean(np.abs(it.dual[ct])) for it in iters], 'o-', ms=3, label=ct)
        ax.set_xlabel('Iteration')
        ax.set_ylabel(r'Mean $|\lambda|$')
        ax.set_title('Linear (Dual) Weights', fontsize=plot_options.title_fontsize, pad=plot_options.title_pad)
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    else:
        if has_W:
            ax = axes[0]
            for ct in iters[0].W.keys():
                ax.semilogy(k, [np.mean(it.W[ct]) for it in iters], 'o-', ms=3, label=ct)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Mean W')
            ax.set_title('Quadratic Penalty Weights', fontsize=plot_options.title_fontsize, pad=plot_options.title_pad)
            ax.legend(fontsize=7, loc='best')
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        if has_dual:
            ax = axes[1]
            for ct in iters[0].dual.keys():
                ax.semilogy(k, [np.mean(np.abs(it.dual[ct])) for it in iters], 'o-', ms=3, label=ct)
            ax.set_xlabel('Iteration')
            ax.set_ylabel(r'Mean $|\lambda|$')
            ax.set_title('Linear (Dual) Weights', fontsize=plot_options.title_fontsize, pad=plot_options.title_pad)
            ax.legend(fontsize=7, loc='best')
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    if save:
        save_dir = os.path.join("plots", "standalone")
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, f"convergence_weights_mean{suffix}.pdf"), dpi=plot_options.save_dpi, pad_inches=0.02)

    return fig


def convergence_vb_plots(result, save=True):
    return _convergence_figures(result, _convergence_vb_plot, "convergence_vb", save)


def _convergence_vb_plot(subprob, suffix, save=True):
    iters = subprob.iter_data_list[1:]
    if not iters:
        return

    has_vb    = hasattr(iters[0], 'vb')   and iters[0].vb
    has_split = hasattr(iters[0], 'vb_p') and iters[0].vb_p

    if not has_vb and not has_split:
        return

    eps_by_name = _penalty_tolerances(subprob)

    k = np.array([it.iter_num for it in iters])
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))

    def _series(vb_key, ct):
        eps_ct = eps_by_name.get(ct, np.atleast_1d(1.0))
        return [float(np.max(np.abs(getattr(it, vb_key)[ct]) / eps_ct, initial=0.)) for it in iters]

    if has_split:
        for ct in iters[0].vb_p.keys():
            ax.semilogy(k, _series('vb_p', ct), 'o-',  ms=3, label=f'{ct} (+)')
            ax.semilogy(k, _series('vb_m', ct), 's--', ms=3, label=f'{ct} (-)')
        for ct in iters[0].vb.keys():
            if ct not in iters[0].vb_p:
                ax.semilogy(k, _series('vb', ct), 'o-', ms=3, label=ct)
    else:
        for ct in iters[0].vb.keys():
            ax.semilogy(k, _series('vb', ct), 'o-', ms=3, label=ct)

    ax.axhline(1.0, color='k', ls='--', lw=0.8)
    ax.set_xlabel('Iteration')
    ax.set_ylabel(r'$\|vb\|_\infty / \epsilon$')
    ax.set_title('Constraint Buffer Violations', fontsize=plot_options.title_fontsize, pad=plot_options.title_pad)
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    if save:
        save_dir = os.path.join("plots", "standalone")
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, f"convergence_vb{suffix}.pdf"), dpi=plot_options.save_dpi, pad_inches=0.02)

    return fig


def _find_trigger_time(group_data, last_iter):
    """Return the time the first output in the group crosses its trigger, or None."""
    for output in group_data.values():
        tl = output.meta.get("trigger_line")
        if tl is None:
            continue
        threshold = float(tl["threshold"])
        direction = tl.get("direction", "below")
        vals = output.opt.squeeze()
        t = last_iter.t_opt[:len(vals)]
        if direction == "below":
            crossed = np.where(vals[:-1] >= threshold, vals[1:] < threshold, False)
        else:
            crossed = np.where(vals[:-1] <= threshold, vals[1:] > threshold, False)
        idx = np.argmax(crossed)
        if not crossed[idx]:
            continue
        v0, v1 = vals[idx], vals[idx + 1]
        dv = v1 - v0
        if abs(dv) < 1e-12:
            return float(t[idx])
        frac = (threshold - v0) / dv
        return float(t[idx] + frac * (t[idx + 1] - t[idx]))
    return None
