"""Default plotting routines for standalone and Monte Carlo SCP analysis results."""

import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
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

plot_options = AttrDict({
    'figsize':     (10, 2.6),
    'dpi':         300,
    'grid_gap_x':  0.06,
    'grid_gap_y':  0.12,
    'margins':     [0.08, 0.02, 0.04, 0.12],
    'title_fontsize': 10,
    'title_pad':   4,
})

# pen = {frgba, lrgba, lw, ls, msty, msz}  (fill rgba, line rgba, linewidth, linestyle, marker style/size)
pens = recursive_attrdict({
    'init':    {'frgba': [0,0,0,.1], 'lrgba': [0,0,0,1.],   'lw': 1, 'ls': '--', 'msty': '',  'msz': 3},
    'nl':      {'frgba': [0,0,0,.1], 'lrgba': [1,0,0,1.],   'lw': 2, 'ls': '-',  'msty': '',  'msz': 3},
    'opt':     {'frgba': [0,0,0,.1], 'lrgba': [0,0,1,1.],   'lw': 1, 'ls': '',   'msty': 'o', 'msz': 3},
    'itr_opt': {'frgba': [0,0,0,.1], 'lrgba': [.7,0,.3,.2], 'lw': 1, 'ls': '',   'msty': 'o', 'msz': 3},
    'itr_nl':  {'frgba': [0,0,0,.1], 'lrgba': [.7,0,.3,.4], 'lw': 1, 'ls': '-',  'msty': '',  'msz': 3},
})

MARKER_DEFAULTS = {
    'marker': '*', 'color': [0.8, 0.0, 0.0], 'size': 80,
    'edgecolor': 'k', 'edgewidth': 0.4, 'zorder': 10, 'fontsize': 7, 'text_offset': [0.0, 0.0]
}


# ======================================================================
# MAIN ENTRY POINT
# ======================================================================

def plot(trajopt_obj, data, show_iters=False):
    method         = list(data.keys())[0]
    iters_all      = data[method]["runs"][0]["iter_data_list"]
    last_iter      = iters_all[-1]
    traj_data      = last_iter["trajectory_data"]
    problem_config = trajopt_obj.problem.config.problem
    traj_group_cfg = problem_config.get('plot_config', {}).get('trajectory_groups', {})
    traj_configs   = problem_config.get('trajectories', {})
    fcns           = problem_config.get('fcns', {})

    # --- create figures and axes for each trajectory group ---
    figs, axs = {}, {}
    for group_name, group_data in traj_data.items():
        grp_cfg = traj_group_cfg.get(group_name, {})
        figsize = grp_cfg.get('figsize', plot_options.figsize)
        pad_3d  = grp_cfg.get('pad_3d', 0.08)
        grid    = _create_grid(len(group_data), cfg=grp_cfg)
        is_3d   = {i: (d.type == "spatial" and d.opt_vals["values"].shape[1] == 3)
                   for i, d in enumerate(group_data.values())}

        fig = plt.figure(figsize=figsize, dpi=plot_options.dpi)
        axs[group_name] = {}
        for idx, rect in grid.items():
            if is_3d.get(idx):
                x, y, w, h = rect
                axs[group_name][idx] = fig.add_axes([x - pad_3d, y, w + pad_3d, h], projection='3d')
            else:
                axs[group_name][idx] = fig.add_axes(rect)
        figs[group_name] = fig

    iters_to_show = iters_all[1:] if show_iters else []

    # --- draw trajectories ---
    for group_name, group_data in traj_data.items():
        for i, (traj_name, traj) in enumerate(group_data.items()):
            ax = axs[group_name][i]
            _setup_ax(ax, traj)
            if traj.type == "spatial":
                _plot_spatial(ax, traj, iters_to_show, last_iter)
            else:
                _plot_time_series(ax, traj, iters_to_show, last_iter)

    # --- draw overlays (ground contours, terrain, etc.) ---
    for group_name, group_data in traj_data.items():
        for i, (traj_name, traj) in enumerate(group_data.items()):
            if traj.type == "spatial":
                ax       = axs[group_name][i]
                dim      = traj.opt_vals["values"].shape[1]
                traj_cfg = traj_configs.get(traj_name, traj)

                limits_opt_only = traj_cfg.get("limits_opt_only", False)
                if limits_opt_only:
                    xlim, ylim = ax.get_xlim(), ax.get_ylim()
                    zlim = ax.get_zlim() if hasattr(ax, 'get_zlim') else None

                _plot_overlays(ax, traj_cfg, dim, trajopt_obj.problem.params, fcns)

                if limits_opt_only:
                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)
                    if zlim is not None and hasattr(ax, 'set_zlim'):
                        ax.set_zlim(zlim)

    # --- set axis limits from optimal + initial guess + quiver data ---
    for group_name, group_data in traj_data.items():
        for i, (traj_name, traj) in enumerate(group_data.items()):
            ax   = axs[group_name][i]
            vals = traj.opt_vals["values"]
            if traj.type == "spatial":
                traj_cfg = traj_configs.get(traj_name, {})
                equal_aspect = bool(traj_cfg.get("equal_aspect", False))
                init_vals = traj.init_nl_vals.get("values", None) if hasattr(traj, "init_nl_vals") else None
                all_vals = np.vstack([vals, init_vals]) if init_vals is not None else vals
                all_vals = _include_quiver_extents(all_vals, traj)
                _set_limits_from_data(ax, all_vals, equal_aspect=equal_aspect)
                if traj_cfg.get("xlim") is not None:
                    ax.set_xlim(traj_cfg["xlim"])
                if traj_cfg.get("ylim") is not None:
                    ax.set_ylim(traj_cfg["ylim"])
            else:
                t = last_iter["t_opt"]
                _set_time_series_limits(ax, t[:vals.shape[0]], vals)

    # --- legend ---
    legend_entries = [('init', 'Initial Guess'), ('opt', 'Optimal'), ('nl', 'Nonlinear')]
    if show_iters:
        legend_entries.append(('itr_nl', 'Iterations'))
    handles = [_legend_handle(name, label) for name, label in legend_entries]
    for fig in figs.values():
        fig.axes[0].legend(handles=handles, loc='best', fontsize=8, framealpha=0.8)

    # --- save ---
    save_dir = os.path.join("plots", "standalone")
    os.makedirs(save_dir, exist_ok=True)
    for name, fig in figs.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"{name}.pdf"), pad_inches=0.02)
    print(f"Saved {len(figs)} figures to {save_dir}/")
    plt.show()
    convergence_plots(trajopt_obj)


# ======================================================================
# AXIS SETUP
# ======================================================================

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


# ======================================================================
# DRAWING PRIMITIVES
# ======================================================================

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


# ======================================================================
# TRAJECTORY PLOTTING
# ======================================================================

def _plot_spatial(ax, traj, iters_to_show, last_iter):
    dim   = traj.opt_vals["values"].shape[1]
    is_3d = dim == 3

    def unpack(vals):
        v = vals["values"]
        return (v[:, 0], v[:, 1], v[:, 2]) if is_3d else (v[:, 0], v[:, 1])

    # iteration history (faded)
    n = len(iters_to_show)
    for i, it in enumerate(iters_to_show):
        traj_it = it["trajectory_data"]
        group   = _find_group(traj_it, traj.name)
        if group is _NOT_FOUND: continue
        t = traj_it[group][traj.name]

        coords = unpack(t.nl_vals)
        _draw(ax, *coords[:2], pens.itr_nl,  z=coords[2] if is_3d else None, n_iters=n, i=i)
        coords = unpack(t.opt_vals)
        _draw(ax, *coords[:2], pens.itr_opt, z=coords[2] if is_3d else None, n_iters=n, i=i)

    # final solution
    coords = unpack(traj.init_nl_vals)
    _draw(ax, *coords[:2], pens.init, z=coords[2] if is_3d else None)
    coords = unpack(traj.nl_vals)
    _draw(ax, *coords[:2], pens.nl,   z=coords[2] if is_3d else None)
    coords = unpack(traj.opt_vals)
    _draw(ax, *coords[:2], pens.opt,  z=coords[2] if is_3d else None)

    _plot_markers(ax, traj, dim)
    _plot_quivers(ax, traj, dim)


def _plot_time_series(ax, traj, iters_to_show, last_iter):
    # iteration history (faded)
    n = len(iters_to_show)
    for i, it in enumerate(iters_to_show):
        traj_it = it["trajectory_data"]
        group   = _find_group(traj_it, traj.name)
        if group is _NOT_FOUND: continue
        t = traj_it[group][traj.name]

        _draw(ax, it["t_nl"],  t.nl_vals["values"].squeeze(),  pens.itr_nl,  n_iters=n, i=i)
        _draw(ax, it["t_opt"], t.opt_vals["values"].squeeze(), pens.itr_opt, n_iters=n, i=i)

    # final solution
    _draw(ax, last_iter["t_init_nl"], traj.init_nl_vals["values"].squeeze(), pens.init)
    _draw(ax, last_iter["t_nl"],      traj.nl_vals["values"].squeeze(),      pens.nl)
    _draw(ax, last_iter["t_opt"],     traj.opt_vals["values"].squeeze(),     pens.opt)

    # constraint limits
    upper = traj.get("upper_limit") or traj.nl_vals.get("limits", {}).get("upper")
    lower = traj.get("lower_limit") or traj.nl_vals.get("limits", {}).get("lower")
    for val in filter(None, [upper, lower]):
        if isinstance(val, np.ndarray):
            t = last_iter["t_nl"]
            ax.plot(t[:len(val)], val, color='k', ls='--', lw=1, alpha=0.5)
        else:
            ax.axhline(val, color='k', ls='--', lw=1, alpha=0.5)


# ======================================================================
# OVERLAYS: MARKERS, QUIVERS, CUSTOM CURVES
# ======================================================================

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
    for q in (traj.opt_vals.get("quivers") or []):
        cfg    = q["config"]
        stride = int(cfg.get("stride", 1))
        scale  = float(cfg.get("scale", 1.0)) * (-1.0 if cfg.get("negate") else 1.0)

        idx  = np.arange(0, len(traj.opt_vals["values"]), stride)
        o    = traj.opt_vals["values"][idx]
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


# ======================================================================
# LEGEND
# ======================================================================

def _legend_handle(pen_name, label):
    p = pens[pen_name]
    return Line2D([], [], color=p.lrgba[:3], alpha=p.lrgba[3], lw=p.lw,
                  ls=p.ls or 'None', marker=p.msty or None, markersize=p.msz, label=label)


# ======================================================================
# GRID LAYOUT
# ======================================================================

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


# ======================================================================
# UTILITIES
# ======================================================================

def _padded_lim(lo, hi, margin=0.08):
    pad = margin * (hi - lo) if hi > lo else 0.1
    return lo - pad, hi + pad


def _set_limits_from_data(ax, vals, margin=0.08, equal_aspect=False):
    ax.set_xlim(*_padded_lim(vals[:, 0].min(), vals[:, 0].max(), margin))
    ax.set_ylim(*_padded_lim(vals[:, 1].min(), vals[:, 1].max(), margin))
    if vals.shape[1] >= 3 and hasattr(ax, 'set_zlim'):
        ax.set_zlim(*_padded_lim(vals[:, 2].min(), vals[:, 2].max(), margin))
        if equal_aspect:
            _set_equal_aspect_3d(ax)
    elif equal_aspect:
        ax.set_aspect('equal', adjustable='box')


def _set_equal_aspect_3d(ax):
    """Force equal scaling on all three axes of a 3D plot."""
    xlim = np.array(ax.get_xlim())
    ylim = np.array(ax.get_ylim())
    zlim = np.array(ax.get_zlim())

    spans = np.array([xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]])
    max_span = spans.max()

    mid_x = xlim.mean()
    mid_y = ylim.mean()
    mid_z = zlim.mean()

    ax.set_xlim(mid_x - max_span / 2, mid_x + max_span / 2)
    ax.set_ylim(mid_y - max_span / 2, mid_y + max_span / 2)
    ax.set_zlim(mid_z - max_span / 2, mid_z + max_span / 2)


def _set_time_series_limits(ax, t, vals, margin=0.08):
    ax.set_xlim(*_padded_lim(t.min(), t.max(), margin))
    lo, hi = vals.min(), vals.max()
    ax.set_ylim(*_padded_lim(lo, hi, margin))


def _include_quiver_extents(all_vals, traj):
    """Expand the point set with quiver tip positions so axis limits fit them."""
    for q in (traj.opt_vals.get("quivers") or []):
        cfg      = q["config"]
        stride   = int(cfg.get("stride", 1))
        scale    = float(cfg.get("scale", 1.0)) * (-1.0 if cfg.get("negate") else 1.0)
        centered = bool(cfg.get("centered", False))

        origins = traj.opt_vals["values"][::stride]
        if q.get("origins") is not None:
            origins = origins + q["origins"][::stride]
        dirs    = q["dirs"][::stride] * scale
        if centered:
            tips = np.vstack([origins - dirs / 2, origins + dirs / 2])
        else:
            tips = origins + dirs
        all_vals = np.vstack([all_vals, tips])
    return all_vals


def convergence_plots(traj, save=True):
    iters   = traj.method.iter_data_list[1:]
    k       = np.arange(1, len(iters) + 1)
    dz      = [it.chk.dz for it in iters]
    dcost   = [it.chk.dcost for it in iters]
    alphas  = [it.get("alpha", 1.0) for it in iters]
    costs   = [it.cost for it in iters]

    eps_stack = traj.method.eps_stack
    vb_types  = list(iters[0].vb.keys()) if hasattr(iters[0], 'vb') and iters[0].vb else []

    vb_series = {}
    for ct in vb_types:
        eps_ct = np.atleast_1d(eps_stack.get(ct, 1.0))
        vb_series[ct] = [float(np.max(np.abs(it.vb[ct]) / eps_ct)) for it in iters]

    grid  = _create_grid(3)
    fig   = plt.figure(figsize=(14, 3.5), dpi=plot_options.dpi)
    axes  = [fig.add_axes(grid[i]) for i in range(3)]

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

    ax = axes[2]
    ax.plot(k, alphas, 'o-', ms=3, color=pens.nl.lrgba[:3])
    ax.set_xlabel('Iteration'); ax.set_ylabel(r'$\alpha$')
    ax.set_title('Line-Search Step Size', fontsize=plot_options.title_fontsize, pad=plot_options.title_pad)
    ax.set_ylim([0, 1.1]); ax.grid(True, alpha=0.3)

    if save:
        save_dir = os.path.join("plots", "standalone")
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, "convergence.pdf"), pad_inches=0.02)
    plt.show()

    convergence_step_plots(traj, save=save)
    convergence_weight_plots(traj, save=save)

    return fig


def convergence_step_plots(traj, save=True):
    """Per-node |dx| and |du| across SCP iterations to identify which
    state/control components are still changing at convergence."""
    method = traj.method
    iters  = method.iter_data_list[1:]
    if not iters:
        return

    idx_state   = method.index_map.indices.z.state
    idx_control = method.index_map.indices.nu.control
    n_state     = method.index_map.n.state
    n_control   = method.index_map.n.control
    N           = method.index_map.N.time_grid
    nodes       = np.arange(N)

    state_cfg   = traj.config.problem.state
    control_cfg = traj.config.problem.control
    state_names   = sorted(state_cfg.keys(),   key=lambda s: min(state_cfg[s]["idx"]))
    control_names = sorted(control_cfg.keys(), key=lambda s: min(control_cfg[s]["idx"]))
    state_names   += [f"z_{j}" for j in range(len(state_names), n_state)]
    control_names += [f"u_{j}" for j in range(len(control_names), n_control)]

    n_iters = len(iters)
    cmap_state   = plt.cm.tab10(np.linspace(0, 1, 10))
    cmap_control = plt.cm.tab10(np.linspace(0, 1, 10))

    fig, (ax_dx, ax_du) = plt.subplots(1, 2, figsize=(14, 4), dpi=plot_options.dpi)

    for i, it in enumerate(iters):
        alpha_step = float(it.get("alpha", 1.0))
        dx = np.abs(alpha_step * np.asarray(it.dz)[:, idx_state])
        du = np.abs(alpha_step * np.asarray(it.dnu)[:, idx_control])

        frac = 0.08 + 0.92 * (i / max(n_iters - 1, 1))
        is_last = (i == n_iters - 1)
        lw = 1.6 if is_last else 0.6

        for j in range(n_state):
            ax_dx.plot(nodes, dx[:, j],
                       color=cmap_state[j % 10], alpha=frac, lw=lw,
                       label=state_names[j] if is_last else None)

        for j in range(n_control):
            ax_du.plot(nodes, du[:, j],
                       color=cmap_control[j % 10], alpha=frac, lw=lw,
                       label=control_names[j] if is_last else None)

    for ax, title, ylabel in [
        (ax_dx, r'State Step $|\alpha \cdot \delta x|$ per Node',   r'$|\alpha \cdot \delta x|$'),
        (ax_du, r'Control Step $|\alpha \cdot \delta u|$ per Node', r'$|\alpha \cdot \delta u|$'),
    ]:
        ax.set_yscale('log')
        ax.set_xlabel('Node')
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=plot_options.title_fontsize, pad=plot_options.title_pad)
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        save_dir = os.path.join("plots", "standalone")
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, "convergence_steps.pdf"), pad_inches=0.02)
    plt.show()
    return fig


def convergence_weight_plots(traj, save=True):
    iters = traj.method.iter_data_list[1:]
    if not iters:
        return

    has_W    = hasattr(iters[0], 'W')    and iters[0].W
    has_dual = hasattr(iters[0], 'dual') and iters[0].dual

    if not has_W and not has_dual:
        return

    k = np.arange(1, len(iters) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 3.5), dpi=plot_options.dpi)

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
        fig.savefig(os.path.join(save_dir, "convergence_weights.pdf"), pad_inches=0.02)
    plt.show()
    return fig


_NOT_FOUND = object()

def _find_group(traj_data, name):
    for group_name, group in traj_data.items():
        if name in group:
            return group_name
    return _NOT_FOUND
