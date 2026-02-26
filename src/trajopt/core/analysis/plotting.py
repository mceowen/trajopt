import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from trajopt.core.analysis.trajplots import SCVXPLOTS
plt.rcParams["text.usetex"] = False

# =============================================================================
# Pen Styles
# =============================================================================

PENS = {
    'init':    {'frgba': [0, 0, 0, .1], 'lrgba': [0, 0, 0, 1.],   'lw': 1, 'ls': '--', 'msty': '',  'msz': 3},
    'nl':      {'frgba': [0, 0, 0, .1], 'lrgba': [1, 0, 0, 1.],   'lw': 2, 'ls': '-',  'msty': '',  'msz': 3},
    'opt':     {'frgba': [0, 0, 0, .1], 'lrgba': [0, 0, 1, 1.],   'lw': 1, 'ls': '',   'msty': 'o', 'msz': 5},
    'opt_small': {'frgba': [0, 0, 0, .1], 'lrgba': [0, 0, 1, 1.], 'lw': 1, 'ls': '',   'msty': 'o', 'msz': 2},
    'itr_opt': {'frgba': [0, 0, 0, .1], 'lrgba': [.7, 0, .3, .2], 'lw': 1, 'ls': '',   'msty': 'o', 'msz': 3},
    'itr_nl':  {'frgba': [0, 0, 0, .1], 'lrgba': [.7, 0, .3, .4], 'lw': 1, 'ls': '-',  'msty': '',  'msz': 3},
}

# =============================================================================
# Helper Functions
# =============================================================================

def _make_grid_specs(num):
    """Create a grid layout for subplots."""
    colnum = int(np.ceil(np.sqrt(num)))
    rownum = int(np.ceil(num / colnum))
    gap_x, gap_y = 0.04, 0.06
    dx = (0.8 - (colnum - 1) * gap_x) / colnum
    dy = (0.8 - (rownum - 1) * gap_y) / rownum
    grid = {}
    for i in range(rownum):
        for j in range(colnum):
            left = 0.05 + j * (dx + gap_x)
            bottom = 0.95 - (i + 1) * dy - i * gap_y
            grid[i * colnum + j] = [left, bottom, dx, dy]
    return grid


def _get_plotting_config(trajopt_obj):
    """Extract plotting configuration from the problem config."""
    model_config = trajopt_obj.problem.config.get('model', {})
    return model_config.get('plotting', {})


def _method_key(trajopt_obj):
    return next(iter(trajopt_obj.scenario_data))


def _get_final_iter_data(trajopt_obj):
    """Get the final iteration data for setting axis limits."""
    key = _method_key(trajopt_obj)
    return trajopt_obj.scenario_data[key]['mc_data'][0]['iters'][-1]


def _set_axis_limits(ax, y_data, margin_frac=0.1):
    """Set y-axis limits based on data with a margin."""
    y_min, y_max = np.min(y_data), np.max(y_data)
    margin = (y_max - y_min) * margin_frac
    if margin < 1e-10:  # Handle constant data
        margin = abs(y_max) * 0.1 if y_max != 0 else 1.0
    ax.set_ylim(y_min - margin, y_max + margin)

# =============================================================================
# Individual Plot Functions
# =============================================================================

def _plot_states(PLTS, trajopt_obj, show_iters, iters, lgnd, opt_pen):
    """Plot state trajectories (individual or grouped)."""
    plotting_config = _get_plotting_config(trajopt_obj)
    state_groups = plotting_config.get('state_groups', None)
    final_data = _get_final_iter_data(trajopt_obj)
    if state_groups is None:
        state_groups = {f'State {i}': [i] for i in range(trajopt_obj.problem.index_map.n['state'])}
    num_plots = len(state_groups)
    fig = plt.figure(figsize=(20, 10), dpi=300)
    fig.suptitle('States')
    axs = PLTS.createGrid(fig, grid=_make_grid_specs(num_plots))
    PLTS.dumpLegend(lgnd)
    for j, (label, indices) in enumerate(state_groups.items()):
        ax = axs[j]
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
        for state_idx in indices:
            if show_iters:
                PLTS.addPlot2D(ax, pen=PENS['itr_nl'],  ins={'label': 'Iterations', 'x': 't_nl',  'y': ('z_nl', state_idx),  'iters': iters[1:], 'legend': lgnd})
                PLTS.addPlot2D(ax, pen=PENS['itr_opt'], ins={'label': 'Iterations', 'x': 't_opt', 'y': ('z_opt', state_idx), 'iters': iters[1:], 'legend': lgnd})
            PLTS.addPlot2D(ax, pen=PENS['nl'],  ins={'label': 'Propagated', 'x': 't_nl',  'y': ('z_nl', state_idx),  'iters': [-1], 'legend': lgnd})
            PLTS.addPlot2D(ax, pen=opt_pen,    ins={'label': 'Optimal Solution', 'x': 't_opt', 'y': ('z_opt', state_idx), 'iters': [-1], 'legend': lgnd})
        y_data = np.concatenate([final_data['z_opt'][:, idx] for idx in indices])
        _set_axis_limits(ax, y_data)


def _plot_controls(PLTS, trajopt_obj, show_iters, iters, lgnd, opt_pen):
    """Plot control trajectories."""
    num_controls = trajopt_obj.problem.index_map.n['control']
    final_data = _get_final_iter_data(trajopt_obj)
    fig = plt.figure(figsize=(20, 10), dpi=300)
    fig.suptitle('Controls')
    axs = PLTS.createGrid(fig, grid=_make_grid_specs(num_controls))
    PLTS.dumpLegend(lgnd)
    for j in range(num_controls):
        ax = axs[j]
        ax.grid(True, alpha=0.3)
        if show_iters:
            PLTS.addPlot2D(ax, pen=PENS['itr_nl'],  ins={'label': 'Iterations', 'x': 't_nl',  'y': ('nu_nl', j),  'iters': iters[1:], 'legend': lgnd})
            PLTS.addPlot2D(ax, pen=PENS['itr_opt'], ins={'label': 'Iterations', 'x': 't_opt', 'y': ('nu_opt', j), 'iters': iters[1:], 'legend': lgnd})
        PLTS.addPlot2D(ax, pen=PENS['nl'],  ins={'label': 'Propagated', 'x': 't_nl',  'y': ('nu_nl', j),  'iters': [-1], 'legend': lgnd})
        PLTS.addPlot2D(ax, pen=opt_pen,    ins={'label': 'Optimal Solution', 'x': 't_opt', 'y': ('nu_opt', j), 'iters': [-1], 'legend': lgnd})
        y_data = final_data['nu_opt'][:, j]
        _set_axis_limits(ax, y_data)


def _plot_trajectories(PLTS, trajopt_obj, lgnd, opt_pen, show_init=True):
    """Plot 2D/3D trajectory plots if state_traj_groups is defined."""
    state_traj_groups = _get_plotting_config(trajopt_obj).get('state_traj_groups', None)
    if not state_traj_groups:
        return
    if isinstance(state_traj_groups, dict):
        state_traj_groups = list(state_traj_groups.values())
    n = len(state_traj_groups)
    plt_types = {i: ('3D' if len(g) >= 3 else '2D') for i, g in enumerate(state_traj_groups)}
    fig = plt.figure(figsize=(20, 10), dpi=300)
    fig.suptitle('Trajectory')
    axs = PLTS.createGrid2(fig, grid=_make_grid_specs(n), ins={'plt_typs': plt_types})
    PLTS.dumpLegend(lgnd)
    for idx, group in enumerate(state_traj_groups):
        g = [int(i) for i in group]
        ax = axs[idx]
        ax.grid(True, alpha=0.3)
        if len(g) == 2:
            if show_init:
                PLTS.addPlot2D(ax, pen=PENS['init'], ins={'label': 'Initial guess', 'x': ('z_opt', g[0]), 'y': ('z_opt', g[1]), 'iters': [1], 'legend': lgnd})
            PLTS.addPlot2D(ax, pen=PENS['nl'],  ins={'label': 'Propagated', 'x': ('z_nl', g[0]), 'y': ('z_nl', g[1]), 'iters': [-1], 'legend': lgnd})
            PLTS.addPlot2D(ax, pen=opt_pen,     ins={'label': 'Optimal Solution', 'x': ('z_opt', g[0]), 'y': ('z_opt', g[1]), 'iters': [-1], 'legend': lgnd})
        else:
            if show_init:
                PLTS.addPlot3D(ax, pen=PENS['init'], ins={'label': 'Initial guess', 'x': ('z_opt', g[0]), 'y': ('z_opt', g[1]), 'z': ('z_opt', g[2]), 'iters': [1], 'legend': lgnd})
            PLTS.addPlot3D(ax, pen=PENS['nl'],  ins={'label': 'Propagated', 'x': ('z_nl', g[0]), 'y': ('z_nl', g[1]), 'z': ('z_nl', g[2]), 'iters': [-1], 'legend': lgnd})
            PLTS.addPlot3D(ax, pen=opt_pen,     ins={'label': 'Optimal Solution', 'x': ('z_opt', g[0]), 'y': ('z_opt', g[1]), 'z': ('z_opt', g[2]), 'iters': [-1], 'legend': lgnd})

def _plot_constraints(PLTS, data, show_iters, iters, lgnd, opt_pen):
    """Plot constraint data."""
    key = next(iter(data['scenario1']))
    constraint_data = data['scenario1'][key]['mc_data'][0]['iters'][-1].get('constraint_data', {})
    for constraint_group, group_data in constraint_data.items():
        num_constraints = len(group_data)
        if num_constraints == 0:
            continue
        fig = plt.figure(figsize=(20, 10), dpi=300)
        fig.suptitle(constraint_group)
        axs = PLTS.createGrid(fig, grid=_make_grid_specs(num_constraints))
        PLTS.dumpLegend(lgnd)
        for idx, (name, constraint) in enumerate(group_data.items()):
            ax = axs[idx]
            ax.set_title(name)
            ax.grid(True, alpha=0.3)
            nl_loc = ('constraint_data', constraint_group, name, 'nl_vals')
            opt_loc = ('constraint_data', constraint_group, name, 'opt_vals')
            num_cols = constraint['nl_vals']['values'].shape[1]
            for col in range(num_cols):
                if show_iters:
                    PLTS.addPlot2D(ax, pen=PENS['itr_nl'],  ins={'label': name, 'x': 't_nl',  'y': ('values', col), 'dataloc': nl_loc,  'iters': iters[1:], 'legend': lgnd})
                    PLTS.addPlot2D(ax, pen=PENS['itr_opt'], ins={'label': name, 'x': 't_opt', 'y': ('values', col), 'dataloc': opt_loc, 'iters': iters[1:], 'legend': lgnd})
                PLTS.addPlot2D(ax, pen=PENS['nl'],  ins={'label': name, 'x': 't_nl',  'y': ('values', col), 'dataloc': nl_loc,  'iters': [-1], 'legend': lgnd})
                PLTS.addPlot2D(ax, pen=opt_pen,    ins={'label': name, 'x': 't_opt', 'y': ('values', col), 'dataloc': opt_loc, 'iters': [-1], 'legend': lgnd})
            y_data = constraint['opt_vals']['values'].flatten()
            _set_axis_limits(ax, y_data)

# =============================================================================
# Main Plotting Function
# =============================================================================

def plot_default(trajopt_obj, show_iters=True, analysis_type="standalone"):
    """
    Plot states, controls, trajectories, and constraints.
    When analysis_type is "mc": no iteration curves, smaller blue dots, final nl for all MC runs.
    """
    data = {"scenario1": trajopt_obj.scenario_data}
    PLTS = SCVXPLOTS(data)
    lgnd = 'legend1'
    iters = list(range(1000))
    method_key = _method_key(trajopt_obj)
    is_mc = (analysis_type == "mc")
    n_runs = len(trajopt_obj.scenario_data[method_key]['mc_data'])
    runs = list(range(n_runs)) if is_mc else [0]
    show_iters = show_iters and not is_mc
    opt_pen = PENS['opt_small'] if is_mc else PENS['opt']
    show_init = not is_mc

    PLTS.setCurrent({
        'scenarios': ['scenario1'],
        'methods': [method_key],
        'runs': runs,
        'iters': iters
    })

    _plot_states(PLTS, trajopt_obj, show_iters, iters, lgnd, opt_pen)
    _plot_controls(PLTS, trajopt_obj, show_iters, iters, lgnd, opt_pen)
    _plot_trajectories(PLTS, trajopt_obj, lgnd, opt_pen, show_init=show_init)
    _plot_constraints(PLTS, data, show_iters, iters, lgnd, opt_pen)
    plt.show()


# Legacy alias for backward compatibility
makeGridSpecs = _make_grid_specs


def plot_animated(trajopt_obj, interval=200):
    from IPython.display import display, HTML

    key = _method_key(trajopt_obj)
    iters = trajopt_obj.scenario_data[key]['mc_data'][0]['iters'][1:]
    n_iters, n_states, n_ctrl = len(iters), trajopt_obj.problem.index_map.n['state'], trajopt_obj.problem.index_map.n['control']  # use unified index_map
    
    t_all = np.concatenate([it['t_opt'] for it in iters])
    t_lim = [np.nanmin(t_all) * 0.95, np.nanmax(t_all) * 1.05]

    def make_grid(n):
        nc = int(np.ceil(np.sqrt(n)))
        return int(np.ceil(n / nc)), nc

    def setup_axes(fig, axs, n_x, data_key, ylabel_prefix):
        lines_nl, lines_opt = [], []
        opt_key = data_key.replace('_nl', '_opt')
        for j in range(n_x):
            ax = axs.flatten()[j]
            y_all = np.concatenate([it[opt_key][:, j] for it in iters])
            y_min, y_max = np.nanmin(y_all), np.nanmax(y_all)
            if not np.isfinite(y_max - y_min):
                y_min, y_max = 0, 1
            margin = (y_max - y_min) * 0.1 + 1
            ax.set_xlim(t_lim)
            ax.set_ylim(y_min - margin, y_max + margin)
            ax.set_xlabel('Time [s]')
            ax.set_ylabel(f'{ylabel_prefix} {j}')
            ax.grid(True, alpha=0.3)
            ln, = ax.plot([], [], color=PENS['nl']['lrgba'], lw=PENS['nl']['lw'])
            lo, = ax.plot([], [], color=PENS['opt']['lrgba'], marker=PENS['opt']['msty'], ls='', ms=PENS['opt']['msz'])
            lines_nl.append(ln)
            lines_opt.append(lo)
        return lines_nl, lines_opt

    nr, nc = make_grid(n_states)
    fig_s, axs_s = plt.subplots(nr, nc, figsize=(5*nc, 4*nr), dpi=300)
    fig_s.suptitle('States - Iteration Convergence')
    axs_s = np.atleast_1d(axs_s)
    state_nl, state_opt = setup_axes(fig_s, axs_s, n_states, 'z_nl', 'State')
    txt_s = fig_s.text(0.5, 0.02, '', ha='center', fontsize=12)

    def anim_states(f):
        it = iters[f]
        for j in range(n_states):
            state_nl[j].set_data(it['t_nl'], it['z_nl'][:, j])
            state_opt[j].set_data(it['t_opt'], it['z_opt'][:, j])
        txt_s.set_text(f'Iteration {f+1}/{n_iters}')
        return state_nl + state_opt + [txt_s]

    fig_s.tight_layout(rect=[0, 0.03, 1, 0.95])
    anim_s = FuncAnimation(fig_s, anim_states, frames=n_iters, interval=interval, blit=True)
    plt.close(fig_s)
    display(HTML(anim_s.to_jshtml()))

    nr_c, nc_c = make_grid(n_ctrl)
    fig_c, axs_c = plt.subplots(nr_c, nc_c, figsize=(5*nc_c, 4*nr_c), dpi=300)
    fig_c.suptitle('Controls - Iteration Convergence')
    axs_c = np.atleast_1d(axs_c)
    ctrl_nl, ctrl_opt = setup_axes(fig_c, axs_c, n_ctrl, 'nu_nl', 'Control')
    txt_c = fig_c.text(0.5, 0.02, '', ha='center', fontsize=12)

    def anim_ctrl(f):
        it = iters[f]
        for j in range(n_ctrl):
            ctrl_nl[j].set_data(it['t_nl'], it['nu_nl'][:, j])
            ctrl_opt[j].set_data(it['t_opt'], it['nu_opt'][:, j])
        txt_c.set_text(f'Iteration {f+1}/{n_iters}')
        return ctrl_nl + ctrl_opt + [txt_c]

    fig_c.tight_layout(rect=[0, 0.03, 1, 0.95])
    anim_c = FuncAnimation(fig_c, anim_ctrl, frames=n_iters, interval=interval, blit=True)
    plt.close(fig_c)
    display(HTML(anim_c.to_jshtml()))

    constraint_data = iters[-1].get('constraint_data', {})
    for group_name, group_data in constraint_data.items():
        n_cnst = len(group_data)
        if n_cnst == 0:
            continue
        nr_cn, nc_cn = make_grid(n_cnst)
        fig_cn, axs_cn = plt.subplots(nr_cn, nc_cn, figsize=(5*nc_cn, 4*nr_cn), dpi=300)
        fig_cn.suptitle(f'{group_name} - Iteration Convergence')
        axs_cn = np.atleast_1d(axs_cn).flatten()
        
        lines_data = []
        for idx, (name, _) in enumerate(group_data.items()):
            ax = axs_cn[idx]
            ax.set_xlabel('Time [s]')
            ax.set_ylabel(name)
            ax.grid(True, alpha=0.3)
            
            opt_vals = [it['constraint_data'][group_name][name]['opt_vals']['values'] for it in iters]
            y_all = np.concatenate(opt_vals)
            y_min, y_max = np.nanmin(y_all), np.nanmax(y_all)
            if not np.isfinite(y_max - y_min):
                y_min, y_max = 0, 1
            margin = (y_max - y_min) * 0.1 + 1
            ax.set_xlim(t_lim)
            ax.set_ylim(y_min - margin, y_max + margin)
            
            n_cols = opt_vals[0].shape[1]
            ln_list, lo_list = [], []
            for _ in range(n_cols):
                ln, = ax.plot([], [], color=PENS['nl']['lrgba'], lw=PENS['nl']['lw'])
                lo, = ax.plot([], [], color=PENS['opt']['lrgba'], marker=PENS['opt']['msty'], ls='', ms=PENS['opt']['msz'])
                ln_list.append(ln)
                lo_list.append(lo)
            lines_data.append((name, ln_list, lo_list, n_cols))
        
        txt_cn = fig_cn.text(0.5, 0.02, '', ha='center', fontsize=12)
        
        def make_anim_func(group_name, lines_data, txt):
            def anim_fn(f):
                it = iters[f]
                all_lines = []
                for name, ln_list, lo_list, n_cols in lines_data:
                    nl_vals = it['constraint_data'][group_name][name]['nl_vals']['values']
                    opt_vals = it['constraint_data'][group_name][name]['opt_vals']['values']
                    for c in range(n_cols):
                        ln_list[c].set_data(it['t_nl'], nl_vals[:, c])
                        lo_list[c].set_data(it['t_opt'], opt_vals[:, c])
                    all_lines.extend(ln_list + lo_list)
                txt.set_text(f'Iteration {f+1}/{n_iters}')
                return all_lines + [txt]
            return anim_fn
        
        fig_cn.tight_layout(rect=[0, 0.03, 1, 0.95])
        anim_cn = FuncAnimation(fig_cn, make_anim_func(group_name, lines_data, txt_cn), frames=n_iters, interval=interval, blit=True)
        plt.close(fig_cn)
        display(HTML(anim_cn.to_jshtml()))
