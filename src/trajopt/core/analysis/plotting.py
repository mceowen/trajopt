import numpy as np
import matplotlib.pyplot as plt
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


def plot_default(trajopt_obj):

    data = {"scenario1": trajopt_obj.scenario_data}
    PLTS = SCVXPLOTS(data)
    lgnd = 'legend1'
    iters = list(range(1000))

    PLTS.setCurrent({'scenarios': ['scenario1'], 'methods': ['autotune'], 'runs': list(range(1000)), 'iters': iters})

    num_states = trajopt_obj.problem.nz
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle('States')
    axs = PLTS.createGrid(fig, grid=makeGridSpecs(num_states))
    PLTS.dumpLegend(lgnd)

    for j in range(num_states):
        ax = axs[j]
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
                PLTS.addPlot2D(ax, pen=PENS['itr_nl'],  ins={'label': name, 'x': 't_nl',  'y': ('values', col), 'dataloc': nl_loc,  'iters': iters[1:], 'legend': lgnd})
                PLTS.addPlot2D(ax, pen=PENS['itr_opt'], ins={'label': name, 'x': 't_opt', 'y': ('values', col), 'dataloc': opt_loc, 'iters': iters[1:], 'legend': lgnd})
                PLTS.addPlot2D(ax, pen=PENS['nl'],      ins={'label': name, 'x': 't_nl',  'y': ('values', col), 'dataloc': nl_loc,  'iters': [-1],      'legend': lgnd})
                PLTS.addPlot2D(ax, pen=PENS['opt'],     ins={'label': name, 'x': 't_opt', 'y': ('values', col), 'dataloc': opt_loc, 'iters': [-1],      'legend': lgnd})