import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from trajopt.core.analysis.trajplots import SCVXPLOTS
from trajopt.utils.tools import AttrDict

plt.rcParams["text.usetex"] = False

# ======================================================================
# CONFIG AND PENS
# ======================================================================

PLOT_OPTS = AttrDict({
    'figsize': (12, 6),
    'dpi': 300,
    'anim_figsize_per_subplot': (3, 2.5),
    'anim_dpi': 300,
    'anim_interval': 150,
    'anim_fps': 10,
})

# pen: frgba, lrgba, lw, ls, msty, msz
PENS = AttrDict({
    'init':         {'frgba': [0,0,0,.1], 'lrgba': [0,0,0,1.],   'lw': 1, 'ls': '--', 'msty': '',  'msz': 3},
    'nl':           {'frgba': [0,0,0,.1], 'lrgba': [1,0,0,1.],   'lw': 2, 'ls': '-',  'msty': '',  'msz': 3},
    'opt':          {'frgba': [0,0,0,.1], 'lrgba': [0,0,1,1.],   'lw': 1, 'ls': '',   'msty': 'o', 'msz': 5},
    'itr_opt':      {'frgba': [0,0,0,.1], 'lrgba': [.7,0,.3,.2], 'lw': 1, 'ls': '',   'msty': 'o', 'msz': 3},
    'itr_nl':       {'frgba': [0,0,0,.1], 'lrgba': [.7,0,.3,.4], 'lw': 1, 'ls': '-',  'msty': '',  'msz': 3},
    'standard_opt': {'frgba': [0,0,0,.1], 'lrgba': [0,0,1,1.],   'lw': 1, 'ls': '',   'msty': 'o', 'msz': 2},
    'autotune_opt':  {'frgba': [0,0,0,.1], 'lrgba': [1,0,1,1.],   'lw': 1, 'ls': '',   'msty': 'o', 'msz': 2},
    'standard_nl':  {'frgba': [0,0,0,.1], 'lrgba': [0,0,1,1.],   'lw': 1, 'ls': '-',  'msty': '',  'msz': 3},
    'autotune_nl':   {'frgba': [0,0,0,.1], 'lrgba': [1,0,1,1.],   'lw': 1, 'ls': '-',  'msty': '',  'msz': 3},
})

# ======================================================================
# DEFAULT PLOTTING SCRIPT
# ======================================================================

def plot_default(trajopt_obj, show_iters=True, analysis_type="standalone"):
    data = AttrDict({"scenario1": trajopt_obj.results})
    PLTS = SCVXPLOTS(data)
    lgnd = 'legend1'
    iters = list(range(1000))
    method_keys = list(trajopt_obj.results)
    mkey = method_keys[0]
    n_runs = len(trajopt_obj.results[mkey]['runs'])
    
    if (analysis_type == "mc"):
        runs = list(range(n_runs))
    else:
        runs = [0]
    
    show_iters = show_iters and not (analysis_type == "mc")
    
    if (analysis_type == "mc"):
        opt_pen = [PENS['standard_opt'], PENS['autotune_opt']]
        nl_pen = [PENS['standard_nl'], PENS['autotune_nl']]
        show_init = False
    else:
        opt_pen = PENS['opt']
        nl_pen = PENS['nl']
        show_init = True

    PLTS.setCurrent({
        'scenarios': ['scenario1'],
        'methods': method_keys,
        'runs': runs,
        'iters': iters,
    })
    
    plot_states(PLTS, trajopt_obj, show_iters, iters, lgnd, opt_pen, nl_pen)
    plot_controls(PLTS, trajopt_obj, show_iters, iters, lgnd, opt_pen, nl_pen)
    plot_trajectories(PLTS, trajopt_obj, lgnd, opt_pen, nl_pen, show_init=show_init)
    plot_constraints(PLTS, trajopt_obj, mkey, show_iters, iters, lgnd, opt_pen, nl_pen)
    
    plt.show()

# ======================================================================
# PLOT FUNCTIONS
# ======================================================================

def plot_states(PLTS, trajopt_obj, show_iters, iters, lgnd, opt_pen, nl_pen=None):
    if nl_pen is None:
        nl_pen = PENS['nl']

    problem = trajopt_obj.problem
    model_config = problem.config.problem.model
    
    plotting_config = model_config.get('plotting', {})
    state_gr = plotting_config.get('state_groups')
    
    if state_gr is None:
        n_x = problem.index_map.n['state']
        state_gr = {f'State {i}': [i] for i in range(n_x)}
    
    n_groups = len(state_gr)
    n_x = problem.index_map.n['state']
    all_z = _gather_final(trajopt_obj, 'z_opt', n_x)

    fig = plt.figure(figsize=PLOT_OPTS.figsize, dpi=PLOT_OPTS.dpi)
    fig.suptitle('States')
    axs = PLTS.createGrid(fig, grid=_grid(n_groups))
    PLTS.dumpLegend(lgnd)

    for j, (label, indices) in enumerate(state_gr.items()):
        ax = axs[j]
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
        for idx in indices:
            _add_series(ax, PLTS, show_iters, iters, lgnd, nl_pen, opt_pen,
                        't_nl', ('z_nl', idx), 't_opt', ('z_opt', idx))
        y_for_lim = np.concatenate([all_z[:, i] for i in indices])
        _ylim(ax, y_for_lim)

def plot_controls(PLTS, trajopt_obj, show_iters, iters, lgnd, opt_pen, nl_pen=None):
    if nl_pen is None:
        nl_pen = PENS['nl']

    n_ctrl = trajopt_obj.problem.index_map.n['control']
    all_nu = _gather_final(trajopt_obj, 'nu_opt', n_ctrl)

    fig = plt.figure(figsize=PLOT_OPTS.figsize, dpi=PLOT_OPTS.dpi)
    fig.suptitle('Controls')
    axs = PLTS.createGrid(fig, grid=_grid(n_ctrl))
    PLTS.dumpLegend(lgnd)

    for j in range(n_ctrl):
        ax = axs[j]
        ax.grid(True, alpha=0.3)
        _add_series(ax, PLTS, show_iters, iters, lgnd, nl_pen, opt_pen,
                    't_nl', ('nu_nl', j), 't_opt', ('nu_opt', j))
        _ylim(ax, all_nu[:, j])


def _add_traj(ax, PLTS, g, show_init, lgnd, nl_pen, opt_pen):
    if isinstance(nl_pen, list):
        pen_nl = nl_pen[0]
    else:
        pen_nl = nl_pen
    if isinstance(opt_pen, list):
        pen_opt = opt_pen[0]
    else:
        pen_opt = opt_pen

    is_3d = len(g) >= 3

    if show_init:
        init_ins = {
            'label': 'Initial guess',
            'iters': [1],
            'legend': lgnd,
            'x': ('z_opt', g[0]),
            'y': ('z_opt', g[1]),
        }
        if is_3d:
            init_ins['z'] = ('z_opt', g[2])
            PLTS.addPlot3D(ax, pen=PENS['init'], ins=init_ins)
        else:
            PLTS.addPlot2D(ax, pen=PENS['init'], ins=init_ins)

    nl_ins = {
        'label': 'Propagated',
        'iters': [-1],
        'legend': lgnd,
        'x': ('z_nl', g[0]),
        'y': ('z_nl', g[1]),
    }
    opt_ins = {
        'label': 'Optimal Solution',
        'iters': [-1],
        'legend': lgnd,
        'x': ('z_opt', g[0]),
        'y': ('z_opt', g[1]),
    }
    if is_3d:
        nl_ins['z'] = ('z_nl', g[2])
        opt_ins['z'] = ('z_opt', g[2])
    if isinstance(nl_pen, list):
        nl_ins['method_pens'] = nl_pen
    if isinstance(opt_pen, list):
        opt_ins['method_pens'] = opt_pen

    if is_3d:
        PLTS.addPlot3D(ax, pen=pen_nl, ins=nl_ins)
        PLTS.addPlot3D(ax, pen=pen_opt, ins=opt_ins)
    else:
        PLTS.addPlot2D(ax, pen=pen_nl, ins=nl_ins)
        PLTS.addPlot2D(ax, pen=pen_opt, ins=opt_ins)


def plot_trajectories(PLTS, trajopt_obj, lgnd, opt_pen, nl_pen=None, show_init=True):
    if nl_pen is None:
        nl_pen = PENS['nl']

    problem = trajopt_obj.problem
    model_config = problem.config.get('model', {})
    plotting_config = model_config.get('plotting', {})
    traj_gr = plotting_config.get('state_traj_groups')
    if not traj_gr:
        return

    if isinstance(traj_gr, dict):
        traj_gr = list(traj_gr.values())
    n = len(traj_gr)

    plt_ty = {}
    for i, g in enumerate(traj_gr):
        if len(g) >= 3:
            plt_ty[i] = '3D'
        else:
            plt_ty[i] = '2D'

    fig = plt.figure(figsize=PLOT_OPTS.figsize, dpi=PLOT_OPTS.dpi)
    fig.suptitle('Trajectory')
    axs = PLTS.createGrid2(fig, grid=_grid(n), ins={'plt_typs': plt_ty})
    PLTS.dumpLegend(lgnd)

    for idx, group in enumerate(traj_gr):
        g = [int(i) for i in group]
        ax = axs[idx]
        ax.grid(True, alpha=0.3)
        _add_traj(ax, PLTS, g, show_init, lgnd, nl_pen, opt_pen)

def plot_constraints(PLTS, trajopt_obj, mkey, show_iters, iters, lgnd, opt_pen, nl_pen=None):
    if nl_pen is None:
        nl_pen = PENS['nl']

    first_run = trajopt_obj.results[mkey]['runs'][0]
    final_iter = first_run['iters'][-1]
    cdata = final_iter.get('constraint_data', {})

    for group_name, group_data in cdata.items():
        n = len(group_data)
        if n == 0:
            continue

        fig = plt.figure(figsize=PLOT_OPTS.figsize, dpi=PLOT_OPTS.dpi)
        fig.suptitle(group_name)
        axs = PLTS.createGrid(fig, grid=_grid(n))
        PLTS.dumpLegend(lgnd)

        for idx, (name, constraint) in enumerate(group_data.items()):
            ax = axs[idx]
            ax.set_title(name)
            ax.grid(True, alpha=0.3)

            nl_loc = ('constraint_data', group_name, name, 'nl_vals')
            opt_loc = ('constraint_data', group_name, name, 'opt_vals')
            n_cols = constraint['nl_vals']['values'].shape[1]

            all_vals = []
            for _mk, method_data in trajopt_obj.results.items():
                for run in method_data['runs']:
                    it = run['iters'][-1]
                    cd = it.get('constraint_data', {})
                    cd_group = cd.get(group_name, {})
                    cd_constraint = cd_group.get(name, {})
                    if cd_constraint.get('opt_vals') is not None:
                        all_vals.append(cd_constraint['opt_vals']['values'].flatten())
            if all_vals:
                all_vals = np.concatenate(all_vals)
            else:
                all_vals = constraint['opt_vals']['values'].flatten()

            for col in range(n_cols):
                _add_series(ax, PLTS, show_iters, iters, lgnd, nl_pen, opt_pen,
                            't_nl', ('values', col), 't_opt', ('values', col),
                            name, name,
                            dataloc_nl=nl_loc, dataloc_opt=opt_loc)
            _ylim(ax, all_vals)

# ======================================================================
# HELPERS
# ======================================================================

def _grid(n):
    nc = int(np.ceil(np.sqrt(n)))
    nr = int(np.ceil(n / nc))
    
    gap_x = 0.04
    gap_y = 0.06
    
    dx = (0.8 - (nc - 1) * gap_x) / nc
    dy = (0.8 - (nr - 1) * gap_y) / nr
    
    grid = {}
    
    for i in range(nr):
        for j in range(nc):
            tag = i * nc + j
            x = 0.05 + j * (dx + gap_x)
            y = 0.95 - (i + 1) * dy - i * gap_y
            grid[tag] = [x, y, dx, dy]
    
    return grid

def _ylim(ax, y, margin_frac=0.1):
    y_lo = np.min(y)
    y_hi = np.max(y)
    span = y_hi - y_lo
    if span > 1e-10:
        margin = span * margin_frac
    else:
        if y_hi != 0:
            margin = abs(y_hi) * 0.1
        else:
            margin = 1.0
    ax.set_ylim(y_lo - margin, y_hi + margin)

def _gather_final(trajopt_obj, key, n_cols):
    out = []
    for _mk, method_data in trajopt_obj.results.items():
        for run in method_data['runs']:
            final_iter = run['iters'][-1]
            out.append(final_iter[key])
    if out:
        return np.concatenate(out, axis=0)
    else:
        return np.zeros((0, n_cols))

def _add_series(ax, PLTS, show_iters, iters, lgnd, nl_pen, opt_pen,
                nl_x, nl_y, opt_x, opt_y,
                label_nl='Propagated', label_opt='Optimal Solution',
                dataloc=None, dataloc_nl=None, dataloc_opt=None):

    if dataloc_nl is not None:
        d_nl = dataloc_nl
    else:
        d_nl = dataloc

    if dataloc_opt is not None:
        d_opt = dataloc_opt
    else:
        d_opt = dataloc

    if show_iters:
        itr_ins_nl = {
            'label': 'Iterations',
            'x': nl_x,
            'y': nl_y,
            'iters': iters[1:],
            'legend': lgnd,
        }
        if d_nl:
            itr_ins_nl['dataloc'] = d_nl
        PLTS.addPlot2D(ax, pen=PENS['itr_nl'], ins=itr_ins_nl)

        itr_ins_opt = {
            'label': 'Iterations',
            'x': opt_x,
            'y': opt_y,
            'iters': iters[1:],
            'legend': lgnd,
        }
        if d_opt:
            itr_ins_opt['dataloc'] = d_opt
        PLTS.addPlot2D(ax, pen=PENS['itr_opt'], ins=itr_ins_opt)

    nl_ins = {
        'label': label_nl,
        'x': nl_x,
        'y': nl_y,
        'iters': [-1],
        'legend': lgnd,
    }
    if d_nl:
        nl_ins['dataloc'] = d_nl
    if isinstance(nl_pen, list):
        nl_ins['method_pens'] = nl_pen

    opt_ins = {
        'label': label_opt,
        'x': opt_x,
        'y': opt_y,
        'iters': [-1],
        'legend': lgnd,
    }
    if d_opt:
        opt_ins['dataloc'] = d_opt
    if isinstance(opt_pen, list):
        opt_ins['method_pens'] = opt_pen

    pen_nl = nl_pen[0] if isinstance(nl_pen, list) else nl_pen
    pen_opt = opt_pen[0] if isinstance(opt_pen, list) else opt_pen
    PLTS.addPlot2D(ax, pen=pen_nl, ins=nl_ins)
    PLTS.addPlot2D(ax, pen=pen_opt, ins=opt_ins)

# ======================================================================
# ANIMATIONS
# ======================================================================

def _anim_show(fig, frame_fn, n_frames, interval):
    from IPython.display import display, HTML
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    anim = FuncAnimation(
        fig, frame_fn,
        frames=n_frames,
        interval=interval,
        blit=True,
        cache_frame_data=False,
    )
    plt.close(fig)
    display(HTML(anim.to_jshtml(fps=PLOT_OPTS.anim_fps)))


def _anim_states_or_controls(n, iters, key_opt, title, ylabel_prefix, t_lim, n_i, w, h, dpi):
    key_nl = key_opt.replace('_opt', '_nl')
    nc = int(np.ceil(np.sqrt(n)))
    nr = int(np.ceil(n / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(w * nc, h * nr), dpi=dpi)
    axs = np.atleast_1d(axs).flatten()
    ln = []
    lo = []
    for j in range(n):
        ax = axs[j]
        y = np.concatenate([it[key_opt][:, j] for it in iters])
        y_min = y.min()
        y_max = y.max()
        span = y_max - y_min
        if np.isfinite(span):
            margin = span * 0.1 + 1
        else:
            margin = 1
        ax.set_xlim(t_lim)
        ax.set_ylim(y_min - margin, y_max + margin)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(f'{ylabel_prefix} {j}')
        ax.grid(True, alpha=0.3)
        line_nl, = ax.plot([], [], color=PENS['nl']['lrgba'], lw=PENS['nl']['lw'])
        line_opt, = ax.plot([], [], color=PENS['opt']['lrgba'], marker=PENS['opt']['msty'], ls='', ms=PENS['opt']['msz'])
        ln.append(line_nl)
        lo.append(line_opt)
    fig.suptitle(title)
    txt = fig.text(0.5, 0.02, '', ha='center', fontsize=12)

    def frame(f):
        it = iters[f]
        for j in range(n):
            ln[j].set_data(it['t_nl'], it[key_nl][:, j])
            lo[j].set_data(it['t_opt'], it[key_opt][:, j])
        txt.set_text(f'Iteration {f+1}/{n_i}')
        return ln + lo + [txt]
    return fig, frame


def plot_animated(trajopt_obj, interval=None):
    if interval is None:
        interval = PLOT_OPTS.anim_interval
    mkey = list(trajopt_obj.results)[0]
    first_run = trajopt_obj.results[mkey]['runs'][0]
    iters = first_run['iters'][1:]
    n_i = len(iters)
    n_x = trajopt_obj.problem.index_map.n['state']
    n_u = trajopt_obj.problem.index_map.n['control']
    t_all = np.concatenate([it['t_opt'] for it in iters])
    t_lim = [t_all.min() * 0.95, t_all.max() * 1.05]
    w, h = PLOT_OPTS.anim_figsize_per_subplot
    dpi  = PLOT_OPTS.anim_dpi

    fig_s, frame_s = _anim_states_or_controls(
        n_x, iters, 'z_opt',
        'States - Iteration Convergence', 'State',
        t_lim, n_i, w, h, dpi,
    )
    _anim_show(fig_s, frame_s, n_i, interval)

    fig_c, frame_c = _anim_states_or_controls(
        n_u, iters, 'nu_opt',
        'Controls - Iteration Convergence', 'Control',
        t_lim, n_i, w, h, dpi,
    )
    _anim_show(fig_c, frame_c, n_i, interval)