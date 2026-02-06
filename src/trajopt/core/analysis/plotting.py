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
    dx, dy = 0.8 / colnum, 0.8 / rownum
    grid = {}
    for i in range(rownum):
        for j in range(colnum):
            grid[i * colnum + j] = [0.05 + dx * j, 0.95 - dy * (i + 1), dx, dy]
    return grid


def _get_plotting_config(trajopt_obj):
    """Extract plotting configuration from the problem config."""
    model_config = trajopt_obj.problem.config.get('model', {})
    return model_config.get('plotting', {})


def _get_final_iter_data(trajopt_obj):
    """Get the final iteration data for setting axis limits."""
    return trajopt_obj.scenario_data['autotune']['mc_data'][0]['iters'][-1]


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

def _plot_states(PLTS, trajopt_obj, show_iters, iters, lgnd):
    """Plot state trajectories (individual or grouped).
    
    Config format (dict only):
        plotting.state_groups:
          "label_name": [index1, index2, ...]
    
    If not provided, defaults to plotting each state individually.
    Axis limits are set based on the converged solution.
    """
    plotting_config = _get_plotting_config(trajopt_obj)
    state_groups = plotting_config.get('state_groups', None)
    final_data = _get_final_iter_data(trajopt_obj)
    
    # Default: plot each state individually
    if state_groups is None:
        state_groups = {f'State {i}': [i] for i in range(trajopt_obj.problem.n)}
    
    num_plots = len(state_groups)
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle('States')
    axs = PLTS.createGrid(fig, grid=_make_grid_specs(num_plots))
    PLTS.dumpLegend(lgnd)
    
    for j, (label, indices) in enumerate(state_groups.items()):
        ax = axs[j]
        ax.set_title(label)
        
        for state_idx in indices:
            if show_iters:
                PLTS.addPlot2D(ax, pen=PENS['itr_nl'],  ins={'label': 'Iterations',       'x': 't_nl',  'y': ('z_nl', state_idx),  'iters': iters[1:], 'legend': lgnd})
                PLTS.addPlot2D(ax, pen=PENS['itr_opt'], ins={'label': 'Iterations',       'x': 't_opt', 'y': ('z_opt', state_idx), 'iters': iters[1:], 'legend': lgnd})
            PLTS.addPlot2D(ax, pen=PENS['nl'],  ins={'label': 'Propagated',       'x': 't_nl',  'y': ('z_nl', state_idx),  'iters': [-1], 'legend': lgnd})
            PLTS.addPlot2D(ax, pen=PENS['opt'], ins={'label': 'Optimal Solution', 'x': 't_opt', 'y': ('z_opt', state_idx), 'iters': [-1], 'legend': lgnd})
        
        # Set axis limits based on converged solution
        y_data = np.concatenate([final_data['z_nl'][:, idx] for idx in indices] +
                                [final_data['z_opt'][:, idx] for idx in indices])
        _set_axis_limits(ax, y_data)


def _plot_controls(PLTS, trajopt_obj, show_iters, iters, lgnd):
    """Plot control trajectories. Axis limits based on converged solution."""
    num_controls = trajopt_obj.problem.m
    final_data = _get_final_iter_data(trajopt_obj)
    
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle('Controls')
    axs = PLTS.createGrid(fig, grid=_make_grid_specs(num_controls))
    PLTS.dumpLegend(lgnd)
    
    for j in range(num_controls):
        ax = axs[j]
        if show_iters:
            PLTS.addPlot2D(ax, pen=PENS['itr_nl'],  ins={'label': 'Iterations',       'x': 't_nl',  'y': ('nu_nl', j),  'iters': iters[1:], 'legend': lgnd})
            PLTS.addPlot2D(ax, pen=PENS['itr_opt'], ins={'label': 'Iterations',       'x': 't_opt', 'y': ('nu_opt', j), 'iters': iters[1:], 'legend': lgnd})
        PLTS.addPlot2D(ax, pen=PENS['nl'],  ins={'label': 'Propagated',       'x': 't_nl',  'y': ('nu_nl', j),  'iters': [-1], 'legend': lgnd})
        PLTS.addPlot2D(ax, pen=PENS['opt'], ins={'label': 'Optimal Solution', 'x': 't_opt', 'y': ('nu_opt', j), 'iters': [-1], 'legend': lgnd})
        
        # Set axis limits based on converged solution
        y_data = np.concatenate([final_data['nu_nl'][:, j], final_data['nu_opt'][:, j]])
        _set_axis_limits(ax, y_data)


def _plot_trajectories(PLTS, trajopt_obj, lgnd):
    """Plot 2D/3D trajectory plots if state_traj_groups is defined."""
    plotting_config = _get_plotting_config(trajopt_obj)
    state_traj_groups = plotting_config.get('state_traj_groups', None)
    
    # Skip trajectory plots if not configured
    if state_traj_groups is None or len(state_traj_groups) == 0:
        return
    
    num_traj_groups = len(state_traj_groups)
    
    # Determine plot types (2D or 3D) based on group size
    plt_types = {}
    for idx, group in enumerate(state_traj_groups):
        plt_types[idx] = '3D' if len(group) >= 3 else '2D'
    
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle('Trajectory')
    axs = PLTS.createGrid2(fig, grid=_make_grid_specs(num_traj_groups), ins={'plt_typs': plt_types})
    PLTS.dumpLegend(lgnd)
    
    for idx, group in enumerate(state_traj_groups):
        ax = axs[idx]
        
        if len(group) == 2:
            # 2D trajectory plot
            PLTS.addPlot2D(ax, pen=PENS['init'], ins={
                'label': 'Initial guess',
                'x': ('z_opt', group[0]),
                'y': ('z_opt', group[1]),
                'iters': [1],
                'legend': lgnd
            })
            PLTS.addPlot2D(ax, pen=PENS['nl'], ins={
                'label': 'Propagated',
                'x': ('z_nl', group[0]),
                'y': ('z_nl', group[1]),
                'iters': [-1],
                'legend': lgnd
            })
            PLTS.addPlot2D(ax, pen=PENS['opt'], ins={
                'label': 'Optimal Solution',
                'x': ('z_opt', group[0]),
                'y': ('z_opt', group[1]),
                'iters': [-1],
                'legend': lgnd
            })
        
        elif len(group) >= 3:
            # 3D trajectory plot
            PLTS.addPlot3D(ax, pen=PENS['init'], ins={
                'label': 'Initial guess',
                'x': ('z_opt', group[0]),
                'y': ('z_opt', group[1]),
                'z': ('z_opt', group[2]),
                'iters': [1],
                'legend': lgnd
            })
            PLTS.addPlot3D(ax, pen=PENS['nl'], ins={
                'label': 'Propagated',
                'x': ('z_nl', group[0]),
                'y': ('z_nl', group[1]),
                'z': ('z_nl', group[2]),
                'iters': [-1],
                'legend': lgnd
            })
            PLTS.addPlot3D(ax, pen=PENS['opt'], ins={
                'label': 'Optimal Solution',
                'x': ('z_opt', group[0]),
                'y': ('z_opt', group[1]),
                'z': ('z_opt', group[2]),
                'iters': [-1],
                'legend': lgnd
            })

def _plot_constraints(PLTS, data, show_iters, iters, lgnd):
    """Plot constraint data. Axis limits based on converged solution."""
    constraint_data = data['scenario1']['autotune']['mc_data'][0]['iters'][-1].get('constraint_data', {})
    
    for constraint_group, group_data in constraint_data.items():
        num_constraints = len(group_data)
        if num_constraints == 0:
            continue
        
        fig = plt.figure(figsize=(20, 10))
        fig.suptitle(constraint_group)
        axs = PLTS.createGrid(fig, grid=_make_grid_specs(num_constraints))
        PLTS.dumpLegend(lgnd)
        
        for idx, (name, constraint) in enumerate(group_data.items()):
            ax = axs[idx]
            ax.set_title(name)
            
            nl_loc = ('constraint_data', constraint_group, name, 'nl_vals')
            opt_loc = ('constraint_data', constraint_group, name, 'opt_vals')
            num_cols = constraint['nl_vals']['values'].shape[1]
            
            for col in range(num_cols):
                if show_iters:
                    PLTS.addPlot2D(ax, pen=PENS['itr_nl'],  ins={'label': name, 'x': 't_nl',  'y': ('values', col), 'dataloc': nl_loc,  'iters': iters[1:], 'legend': lgnd})
                    PLTS.addPlot2D(ax, pen=PENS['itr_opt'], ins={'label': name, 'x': 't_opt', 'y': ('values', col), 'dataloc': opt_loc, 'iters': iters[1:], 'legend': lgnd})
                PLTS.addPlot2D(ax, pen=PENS['nl'],  ins={'label': name, 'x': 't_nl',  'y': ('values', col), 'dataloc': nl_loc,  'iters': [-1], 'legend': lgnd})
                PLTS.addPlot2D(ax, pen=PENS['opt'], ins={'label': name, 'x': 't_opt', 'y': ('values', col), 'dataloc': opt_loc, 'iters': [-1], 'legend': lgnd})
            
            # Set axis limits based on converged solution
            y_data = np.concatenate([constraint['nl_vals']['values'].flatten(),
                                     constraint['opt_vals']['values'].flatten()])
            _set_axis_limits(ax, y_data)

# =============================================================================
# Main Plotting Function
# =============================================================================

def plot_default(trajopt_obj, show_iters=True):
    """
    Plot states, controls, trajectories, and constraints.
    
    By default, plots:
      - States: individual plots for each state (or grouped if state_groups is defined)
      - Controls: individual plots for each control
      - Trajectories: 2D/3D plots if state_traj_groups is defined in config
      - Constraints: plots for each constraint group
    
    Args:
        trajopt_obj: The trajectory optimization object with scenario_data
        show_iters: If True, overlay all iterations. If False, only show final solution.
    """
    # Setup
    data = {"scenario1": trajopt_obj.scenario_data}
    PLTS = SCVXPLOTS(data)
    lgnd = 'legend1'
    iters = list(range(1000))
    
    PLTS.setCurrent({
        'scenarios': ['scenario1'],
        'methods': ['autotune'],
        'runs': list(range(1000)),
        'iters': iters
    })
    
    # Generate all plots
    _plot_states(PLTS, trajopt_obj, show_iters, iters, lgnd)
    _plot_controls(PLTS, trajopt_obj, show_iters, iters, lgnd)
    _plot_trajectories(PLTS, trajopt_obj, lgnd)
    _plot_constraints(PLTS, data, show_iters, iters, lgnd)
    
    plt.show()


# Legacy alias for backward compatibility
makeGridSpecs = _make_grid_specs


def plot_animated(trajopt_obj, interval=200):
    from IPython.display import display, HTML
    
    iters = trajopt_obj.scenario_data['autotune']['mc_data'][0]['iters'][1:]
    n_iters, n_states, n_ctrl = len(iters), trajopt_obj.problem.n, trajopt_obj.problem.m  # use original state dimension
    
    t_all = np.concatenate([it['t_nl'] for it in iters])
    t_lim = [t_all.min() * 0.95, t_all.max() * 1.05]

    def make_grid(n):
        nc = int(np.ceil(np.sqrt(n)))
        return int(np.ceil(n / nc)), nc

    def setup_axes(fig, axs, n, data_key, ylabel_prefix):
        lines_nl, lines_opt = [], []
        for j in range(n):
            ax = axs.flatten()[j]
            y_all = np.concatenate([it[data_key][:, j] for it in iters])
            margin = (y_all.max() - y_all.min()) * 0.1 + 1
            ax.set_xlim(t_lim)
            ax.set_ylim(y_all.min() - margin, y_all.max() + margin)
            ax.set_xlabel('Time [s]')
            ax.set_ylabel(f'{ylabel_prefix} {j}')
            ax.grid(True, alpha=0.3)
            ln, = ax.plot([], [], color=PENS['nl']['lrgba'], lw=PENS['nl']['lw'])
            lo, = ax.plot([], [], color=PENS['opt']['lrgba'], marker=PENS['opt']['msty'], ls='', ms=PENS['opt']['msz'])
            lines_nl.append(ln)
            lines_opt.append(lo)
        return lines_nl, lines_opt

    nr, nc = make_grid(n_states)
    fig_s, axs_s = plt.subplots(nr, nc, figsize=(5*nc, 4*nr))
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
    fig_c, axs_c = plt.subplots(nr_c, nc_c, figsize=(5*nc_c, 4*nr_c))
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
        fig_cn, axs_cn = plt.subplots(nr_cn, nc_cn, figsize=(5*nc_cn, 4*nr_cn))
        fig_cn.suptitle(f'{group_name} - Iteration Convergence')
        axs_cn = np.atleast_1d(axs_cn).flatten()
        
        lines_data = []
        for idx, (name, _) in enumerate(group_data.items()):
            ax = axs_cn[idx]
            ax.set_xlabel('Time [s]')
            ax.set_ylabel(name)
            ax.grid(True, alpha=0.3)
            
            all_vals = [it['constraint_data'][group_name][name]['nl_vals']['values'] for it in iters]
            y_all = np.concatenate(all_vals)
            margin = (y_all.max() - y_all.min()) * 0.1 + 1
            ax.set_xlim(t_lim)
            ax.set_ylim(y_all.min() - margin, y_all.max() + margin)
            
            n_cols = all_vals[0].shape[1]
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
