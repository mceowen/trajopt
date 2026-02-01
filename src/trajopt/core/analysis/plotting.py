import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from trajopt.core.analysis.trajplots import SCVXPLOTS

PENS = {}
PENS['init']     = {'frgba':[0,0,0,.1], 'lrgba':[0,0,0,1.],     'lw':1, 'ls':'--', 'msty':'',  'msz':3}
PENS['nl']       = {'frgba':[0,0,0,.1], 'lrgba':[1,0,0,1.],     'lw':2, 'ls':'-',  'msty':'',  'msz':3}
PENS['opt']      = {'frgba':[0,0,0,.1], 'lrgba':[0,0,1,1.],     'lw':1, 'ls':'',   'msty':'o', 'msz':5}
PENS['itr_opt']  = {'frgba':[0,0,0,.1], 'lrgba':[.7,0,.3,.2],   'lw':1, 'ls':'',   'msty':'o', 'msz':3}
PENS['itr_nl']   = {'frgba':[0,0,0,.1], 'lrgba':[.7,0,.3,.4],   'lw':1, 'ls':'-',  'msty':'',  'msz':3}


def makeGridSpecs(num):
    colnum = int(np.ceil(np.sqrt(num)))
    rownum = int(np.ceil(num / colnum))
    dx, dy = 0.8/colnum, 0.8/rownum
    grid = {}
    for i in range(rownum):
        for j in range(colnum):
            grid[i*colnum + j] = [0.05 + dx*j, 0.95 - dy*(i+1), dx, dy]
    return grid


def plot_default(trajopt_obj, show_iters=True):
    """
    Plot states, controls, and constraints.
    
    Args:
        trajopt_obj: The trajectory optimization object with scenario_data
        show_iters: If True, overlay all iterations. If False, only show final solution.
    """
    data = {"scenario1": trajopt_obj.scenario_data}
    PLTS = SCVXPLOTS(data)
    lgnd = 'legend1'
    iters = list(range(1000))

    PLTS.setCurrent({'scenarios': ['scenario1'], 'methods': ['autotune'], 'runs': list(range(1000)), 'iters': iters})

    num_states = trajopt_obj.problem.n
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle('States')
    axs = PLTS.createGrid(fig, grid=makeGridSpecs(num_states))
    PLTS.dumpLegend(lgnd)

    for j in range(num_states):
        ax = axs[j]
        if show_iters:
            PLTS.addPlot2D(ax, pen=PENS['itr_nl'],  ins={'label': 'Iterations',       'x': 't_nl',  'y': ('z_nl', j),  'iters': iters[1:], 'legend': lgnd})
            PLTS.addPlot2D(ax, pen=PENS['itr_opt'], ins={'label': 'Iterations',       'x': 't_opt', 'y': ('z_opt', j), 'iters': iters[1:], 'legend': lgnd})
        PLTS.addPlot2D(ax, pen=PENS['nl'],      ins={'label': 'Propagated',       'x': 't_nl',  'y': ('z_nl', j),  'iters': [-1],      'legend': lgnd})
        PLTS.addPlot2D(ax, pen=PENS['opt'],     ins={'label': 'Optimal Solution', 'x': 't_opt', 'y': ('z_opt', j), 'iters': [-1],      'legend': lgnd})

    num_controls = trajopt_obj.problem.m
    fig_ctrl = plt.figure(figsize=(20, 10))
    fig_ctrl.suptitle('Controls')
    axs_ctrl = PLTS.createGrid(fig_ctrl, grid=makeGridSpecs(num_controls))
    PLTS.dumpLegend(lgnd)

    for j in range(num_controls):
        ax = axs_ctrl[j]
        if show_iters:
            PLTS.addPlot2D(ax, pen=PENS['itr_nl'],  ins={'label': 'Iterations',       'x': 't_nl',  'y': ('nu_nl', j),  'iters': iters[1:], 'legend': lgnd})
            PLTS.addPlot2D(ax, pen=PENS['itr_opt'], ins={'label': 'Iterations',       'x': 't_opt', 'y': ('nu_opt', j), 'iters': iters[1:], 'legend': lgnd})
        PLTS.addPlot2D(ax, pen=PENS['nl'],      ins={'label': 'Propagated',       'x': 't_nl',  'y': ('nu_nl', j),  'iters': [-1],      'legend': lgnd})
        PLTS.addPlot2D(ax, pen=PENS['opt'],     ins={'label': 'Optimal Solution', 'x': 't_opt', 'y': ('nu_opt', j), 'iters': [-1],      'legend': lgnd})

    constraint_data = data['scenario1']['autotune']['mc_data'][0]['iters'][-1]['constraint_data']

    for constraint_group in constraint_data.keys():
        group_data = constraint_data[constraint_group]
        num_constraints = len(group_data)

        fig2 = plt.figure(figsize=(20, 10))
        fig2.suptitle(constraint_group)
        axs2 = PLTS.createGrid(fig2, grid=makeGridSpecs(num_constraints))
        PLTS.dumpLegend(lgnd)

        for idx, (name, constraint) in enumerate(group_data.items()):
            print("constraint_name: " + name)
            ax = axs2[idx]
            nl_loc  = ('constraint_data', constraint_group, name, 'nl_vals')
            opt_loc = ('constraint_data', constraint_group, name, 'opt_vals')
            num_cols = constraint['nl_vals']['values'].shape[1]
            for col in range(num_cols):
                if show_iters:
                    PLTS.addPlot2D(ax, pen=PENS['itr_nl'],  ins={'label': name, 'x': 't_nl',  'y': ('values', col), 'dataloc': nl_loc,  'iters': iters[1:], 'legend': lgnd})
                    PLTS.addPlot2D(ax, pen=PENS['itr_opt'], ins={'label': name, 'x': 't_opt', 'y': ('values', col), 'dataloc': opt_loc, 'iters': iters[1:], 'legend': lgnd})
                PLTS.addPlot2D(ax, pen=PENS['nl'],      ins={'label': name, 'x': 't_nl',  'y': ('values', col), 'dataloc': nl_loc,  'iters': [-1],      'legend': lgnd})
                PLTS.addPlot2D(ax, pen=PENS['opt'],     ins={'label': name, 'x': 't_opt', 'y': ('values', col), 'dataloc': opt_loc, 'iters': [-1],      'legend': lgnd})

    plt.show()


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
