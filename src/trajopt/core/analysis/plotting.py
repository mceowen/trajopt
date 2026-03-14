import enum
import numpy as np
import matplotlib.pyplot as plt
from trajopt.core.analysis.trajplots import SCVXPLOTS
from trajopt.utils.tools import AttrDict, recursive_attrdict

plt.rcParams["text.usetex"] = False

plot_options = AttrDict({
    'figsize': (12, 4),
    'dpi': 300,
    'grid_gap_x': 0.05,
    'grid_gap_y': 0.12,
})

# pen: frgba, lrgba, lw, ls, msty, msz
pens = recursive_attrdict({
    'init':    {'frgba': [0,0,0,.1], 'lrgba': [0,0,0,1.],   'lw': 1, 'ls': '--', 'msty': '',  'msz': 3},
    'nl':      {'frgba': [0,0,0,.1], 'lrgba': [1,0,0,1.],   'lw': 2, 'ls': '-',  'msty': '',  'msz': 3},
    'opt':     {'frgba': [0,0,0,.1], 'lrgba': [0,0,1,1.],   'lw': 1, 'ls': '',   'msty': 'o', 'msz': 5},
    'itr_opt': {'frgba': [0,0,0,.1], 'lrgba': [.7,0,.3,.2], 'lw': 1, 'ls': '',   'msty': 'o', 'msz': 3},
    'itr_nl':  {'frgba': [0,0,0,.1], 'lrgba': [.7,0,.3,.4], 'lw': 1, 'ls': '-',  'msty': '',  'msz': 3},
    'opt1':    {'frgba': [0,0,0,.1], 'lrgba': [0,0,1,1.],   'lw': 1, 'ls': '',   'msty': 'o', 'msz': 2},
    'opt2':    {'frgba': [0,0,0,.1], 'lrgba': [1,0,1,1.],   'lw': 1, 'ls': '',   'msty': 'o', 'msz': 2},
    'nl1':     {'frgba': [0,0,0,.1], 'lrgba': [0,0,1,1.],   'lw': 1, 'ls': '-',  'msty': '',  'msz': 3},
    'nl2':     {'frgba': [0,0,0,.1], 'lrgba': [1,0,1,1.],   'lw': 1, 'ls': '-',  'msty': '',  'msz': 3},
})

def plot_default(trajopt_obj, show_iters=False, analysis_type='standalone'):
    data = AttrDict({"scenario1": trajopt_obj.results})
    
    # create trajplots object
    PLTS = SCVXPLOTS(data)

    method_keys = list(trajopt_obj.results.keys())
    runs = list(range(len(trajopt_obj.results[method_keys[0]]['runs'])))

    nominal_iter_data = trajopt_obj.results[method_keys[0]]["runs"][-1]["iters"][-1]

    # decide whether or not to plot the iterations
    if show_iters:
        iters = list(range(1, 1000))
    else:
        iters = [-1]

    PLTS.setCurrent({
        'scenarios': ['scenario1'],
        'methods': method_keys,
        'runs': runs,
        'iters': iters,
    })

    problem = trajopt_obj.problem
    problem_config = problem.config.problem

    # extract plotting groups from configs
    default_groups_state   = {f'State {i}': [i] for i in range(problem.index_map.n['state'])}
    default_groups_control = {f'Control {i}': [i] for i in range(problem.index_map.n['control'])}
    groups_state           = problem_config.get('state', default_groups_state)
    groups_control         = problem_config.get('control', default_groups_control)

    # extract constraint_groups
    nominal_constraint_data = nominal_iter_data["constraint_data"]
    nominal_trajectory_data = nominal_iter_data["trajectory_data"]

    # define figures and axes for each type of plot
    fig_control     = plt.figure(figsize=plot_options.figsize, dpi=plot_options.dpi)
    fig_state       = plt.figure(figsize=plot_options.figsize, dpi=plot_options.dpi)

    axs_state       = PLTS.createGrid(fig_state ,grid=create_grid(len(groups_state.keys())))
    axs_control     = PLTS.createGrid(fig_control ,grid=create_grid(len(groups_control.keys())))

    # axs_constraints_dict = AttrDict({constraint_group: })

    # create figs and axes for groups of constraints
    figs_constraints = AttrDict({})
    axs_constraints  = AttrDict({})

    # create figs and axes for groups of trajectories
    figs_trajectories = AttrDict({})
    axs_trajectories  = AttrDict({})

    for constraint_group_name, constraint_group_data in nominal_constraint_data.items():
        figs_constraints[constraint_group_name] = plt.figure(figsize=plot_options.figsize, dpi=plot_options.dpi)

        num_constraints_in_group = len(constraint_group_data.keys())
        axs_constraints[constraint_group_name] = PLTS.createGrid(figs_constraints[constraint_group_name] ,grid=create_grid(num_constraints_in_group))

    plt_type = {}
    for trajectory_group_name, trajectory_group_data in nominal_trajectory_data.items():
        figs_trajectories[trajectory_group_name] = plt.figure(figsize=plot_options.figsize, dpi=plot_options.dpi)

        plt_type = AttrDict({})

        for i, (trajectory_name,current_trajectory_data) in enumerate(trajectory_group_data.items()):
            
            if current_trajectory_data.type == "spatial":
                dim = current_trajectory_data.opt_vals["values"].shape[1]

            elif current_trajectory_data.type == "time_series":
                dim = 2
            
            if dim == 3:
                plt_type[i] = "3D"
            else:
                plt_type[i] = "2D"

        num_trajectories_in_group = len(trajectory_group_data.keys())
        axs_trajectories[trajectory_group_name] = PLTS.createGrid2(figs_trajectories[trajectory_group_name] ,grid=create_grid(num_trajectories_in_group), ins={'plt_typs':plt_type})

    if analysis_type == 'standalone':
        standalone_pens = AttrDict({"opt": pens.opt, "nl": pens.nl, "itr_opt": pens.itr_opt, "itr_nl": pens.itr_nl, "init": pens.init})
        
        # plot states, controls, trajectories, and constraints on their respective figures
        plot_states(PLTS, axs_state, groups_state, iters=iters, pens=standalone_pens)
        plot_controls(PLTS, axs_control, groups_control, pens=standalone_pens, iters=iters)
        plot_trajectories(PLTS, axs_trajectories, nominal_trajectory_data, iters=iters, pens=standalone_pens)
        plot_constraints(PLTS, axs_constraints, nominal_constraint_data, iters=iters, pens=standalone_pens)

    elif analysis_type == 'mc':
        for i, (method_name, method_data) in enumerate(trajopt_obj.results.items()):
            PLTS.setCurrent({
                'scenarios': ['scenario1'],
                'methods': [method_name],
                'runs': runs,
                'iters': iters,
            })
            current_method_pens = AttrDict({"opt": pens[f"opt{i+1}"], "nl": pens[f"nl{i+1}"], "itr_opt": pens.itr_opt, "itr_nl": pens.itr_nl, "init": pens.init})
            for run in method_data['runs']:
                plot_states(PLTS, axs_state, groups_state, iters=[-1], pens=current_method_pens)
                plot_controls(PLTS, axs_control, groups_control, pens=current_method_pens, iters=[-1])
                plot_trajectories(PLTS, axs_trajectories, nominal_trajectory_data, iters=[-1], pens=current_method_pens)
                plot_constraints(PLTS, axs_constraints, nominal_constraint_data, iters=[-1], pens=current_method_pens)

    plt.show()

def plot_states(PLTS, axs, groups, 
                iters=[-1], 
                pens = AttrDict({"opt": pens.opt, "nl": pens.nl, "itr_opt": pens.itr_opt, "itr_nl": pens.itr_nl})):

    for i, (group_name, group_data) in enumerate(groups.items()):
        # extract the current group subplot axis
        ax = axs[i]

        group_indices = group_data["idx"]

        # set properties for this subplot
        ax.set_title(group_name)
        ax.grid(True, alpha=0.3)

        for index in group_indices:
            # plot iters if specified
            if iters != [-1]:
                # plot the nonlinear propagation of  openloop solution
                ins_nl = {'label': 'Nonlinear Propagation', 'x': 't_nl', 'y': ('z_nl', index), 'iters': iters, 'legend': "legend1"}
                PLTS.addPlot2D(ax, pen=pens.itr_nl, ins=ins_nl)

                # plot the optimal solution
                ins_opt = {'label': 'Optimal Soltution', 'x': 't_opt', 'y': ('z_opt', index), 'iters': iters, 'legend': "legend1"}
                PLTS.addPlot2D(ax, pen=pens.itr_opt, ins=ins_opt)
            
            # plot the nonlinear propagation of  openloop solution
            ins_nl = {'label': 'Nonlinear Propagation', 'x': 't_nl', 'y': ('z_nl', index), 'iters': [-1], 'legend': "legend1"}
            PLTS.addPlot2D(ax, pen=pens.nl, ins=ins_nl)

            # plot the optimal solution
            ins_opt = {'label': 'Optimal Soltution', 'x': 't_opt', 'y': ('z_opt', index), 'iters': [-1], 'legend': "legend1"}
            PLTS.addPlot2D(ax, pen=pens.opt, ins=ins_opt)

def plot_controls(PLTS, axs, groups, pens, iters=[-1]):

    for i, (group_name, group_data) in enumerate(groups.items()):
        # extract the current group subplot axis
        ax = axs[i]

        # set properties for this subplot
        ax.set_title(group_name)
        ax.grid(True, alpha=0.3)

        group_indices = group_data["idx"]

        for index in group_indices:
            # plot iters if specified
            if iters != [-1]:
                # plot the nonlinear propagation of  openloop solution
                ins_nl = {'label': 'Nonlinear Propagation', 'x': 't_nl', 'y': ('nu_nl', index), 'iters': iters, 'legend': "legend1"}
                PLTS.addPlot2D(ax, pen=pens.itr_nl, ins=ins_nl)

                # plot the optimal solution
                ins_opt = {'label': 'Optimal Soltution', 'x': 't_opt', 'y': ('nu_opt', index), 'iters': iters, 'legend': "legend1"}
                PLTS.addPlot2D(ax, pen=pens.itr_opt, ins=ins_opt)

            # plot the nonlinear propagation of  openloop solution
            ins_nl = {'label': 'Nonlinear Propagation', 'x': 't_nl', 'y': ('nu_nl', index), 'iters': [-1], 'legend': "legend1"}
            PLTS.addPlot2D(ax, pen=pens.nl, ins=ins_nl)

            # plot the optimal solution
            ins_opt = {'label': 'Optimal Soltution', 'x': 't_opt', 'y': ('nu_opt', index), 'iters': [-1], 'legend': "legend1"}
            PLTS.addPlot2D(ax, pen=pens.opt, ins=ins_opt)

def plot_constraints(PLTS, axs, nominal_constraint_data, 
                iters=[-1], 
                pens = AttrDict({"opt": pens.opt, "nl": pens.nl, "itr_opt": pens.itr_opt, "itr_nl": pens.itr_nl})):

    for constraint_group_name, constraint_group_data in nominal_constraint_data.items():
        
        # extract the current group subplot axis
        for i, (constraint_name, current_constraint_data) in enumerate(constraint_group_data.items()):
            ax = axs[constraint_group_name][i]

            # set properties for this subplot
            ax.set_title(constraint_name)
            ax.grid(True, alpha=0.3)

            nl_loc  = ('constraint_data', constraint_group_name, constraint_name, 'nl_vals')
            opt_loc = ('constraint_data', constraint_group_name, constraint_name, 'opt_vals')

            # plot iters if specified
            if iters != [-1]:
                # plot the nonlinear propagation of  openloop solution
                ins_nl = {'label': 'Nonlinear Propagation', 'x': 't_nl', 'y': 'values', 'iters': iters, 'dataloc': nl_loc, 'legend': "legend1"}
                PLTS.addPlot2D(ax, pen=pens.itr_nl, ins=ins_nl)

                # plot the optimal solution
                ins_opt = {'label': 'Optimal Soltution', 'x': 't_opt', 'y': 'values', 'iters': iters, 'dataloc': opt_loc,'legend': "legend1"}
                PLTS.addPlot2D(ax, pen=pens.itr_opt, ins=ins_opt)
            
            # plot the nonlinear propagation of  openloop solution
            ins_nl = {'label': 'Nonlinear Propagation', 'x': 't_nl', 'y': 'values', 'iters': [-1], 'dataloc': nl_loc, 'legend': "legend1"}
            PLTS.addPlot2D(ax, pen=pens.nl, ins=ins_nl)

            # plot the optimal solution
            ins_opt = {'label': 'Optimal Soltution', 'x': 't_opt', 'y': 'values', 'iters': [-1], 'dataloc': opt_loc, 'legend': "legend1"}
            PLTS.addPlot2D(ax, pen=pens.opt, ins=ins_opt)

            # plot constraint limits
            if nominal_constraint_data[constraint_group_name][constraint_name]['opt_vals']["limits"] is not None:
                    ins_limits = {'label': 'Limits', 'x': 't_nl', 'y': 'limits', 'iters': iters, 'dataloc': nl_loc, 'legend': "legend1"}
                    PLTS.addPlot2D(ax, pen=pens.init, ins=ins_limits)

def plot_trajectories(PLTS, axs, nominal_trajectory_data, 
                iters=[-1], 
                pens = AttrDict({"opt": pens.opt, "nl": pens.nl, "itr_opt": pens.itr_opt, "itr_nl": pens.itr_nl})):

    for traj_group_name, traj_group_data in nominal_trajectory_data.items():
        
        # extract the current group subplot axis
        for i, (traj_name, current_traj_data) in enumerate(traj_group_data.items()):
            ax = axs[traj_group_name][i]

            # set properties for this subplot
            ax.set_title(traj_name)
            ax.grid(True, alpha=0.3)

            nl_loc  = ('trajectory_data', traj_group_name, traj_name, 'nl_vals')
            opt_loc = ('trajectory_data', traj_group_name, traj_name, 'opt_vals')

            traj_type = current_traj_data.type

            if traj_type == "spatial":

                plot_spatial_trajectories(PLTS, iters, current_traj_data, nl_loc, opt_loc, ax, pens)

            elif traj_type == "time_series":

                plot_time_series_trajectories(PLTS, iters, current_traj_data, nl_loc, opt_loc, ax, pens)

def plot_spatial_trajectories(PLTS, iters, current_traj_data, nl_loc, opt_loc, ax, pens):
    dim = current_traj_data.opt_vals["values"].shape[1]
    # plot iters if specified
    if iters != [-1]:

        if dim == 3:
            # plot the nonlinear propagation of  openloop solution
            ins_nl = {'label': 'Nonlinear Propagation', 'x': ("values", 0), 'y': ('values', 1), 'z': ('values', 2), 'iters': iters, 'dataloc': nl_loc, 'legend': "legend1"}
            
            PLTS.addPlot3D(ax, pen=pens.itr_nl, ins=ins_nl)

            # plot the optimal solution
            ins_opt = {'label': 'Optimal Soltution', 'x': ('values', 0), 'y': ('values', 1), 'z': ('values', 2), 'iters': iters, 'dataloc': opt_loc,'legend': "legend1"}
            PLTS.addPlot3D(ax, pen=pens.itr_opt, ins=ins_opt)
        
        else:
            # plot the nonlinear propagation of  openloop solution
            ins_nl = {'label': 'Nonlinear Propagation', 'x': ("values", 0), 'y': ('values', 1), 'iters': iters, 'dataloc': nl_loc, 'legend': "legend1"}
            
            PLTS.addPlot2D(ax, pen=pens.itr_nl, ins=ins_nl)

            # plot the optimal solution
            ins_opt = {'label': 'Optimal Soltution', 'x': ('values', 0), 'y': ('values', 1), 'iters': iters, 'dataloc': opt_loc,'legend': "legend1"}
            PLTS.addPlot2D(ax, pen=pens.itr_opt, ins=ins_opt)

    if dim == 3:
        # plot the nonlinear propagation of  openloop solution
        ins_nl = {'label': 'Nonlinear Propagation', 'x': ('values', 0), 'y': ('values', 1), 'z': ('values', 2), 'iters': [-1], 'dataloc': nl_loc, 'legend': "legend1"}
        PLTS.addPlot3D(ax, pen=pens.nl, ins=ins_nl)

        # plot the optimal solution
        ins_opt = {'label': 'Optimal Soltution', 'x': ('values', 0), 'y': ('values', 1), 'z': ('values', 2), 'iters': [-1], 'dataloc': opt_loc, 'legend': "legend1"}
        PLTS.addPlot3D(ax, pen=pens.opt, ins=ins_opt)
    
    else:
        # plot the nonlinear propagation of  openloop solution
        ins_nl = {'label': 'Nonlinear Propagation', 'x': ('values', 0), 'y': ('values', 1), 'iters': [-1], 'dataloc': nl_loc, 'legend': "legend1"}
        PLTS.addPlot2D(ax, pen=pens.nl, ins=ins_nl)

        # plot the optimal solution
        ins_opt = {'label': 'Optimal Soltution', 'x': ('values', 0), 'y': ('values', 1), 'iters': [-1], 'dataloc': opt_loc, 'legend': "legend1"}
        PLTS.addPlot2D(ax, pen=pens.opt, ins=ins_opt)

def plot_time_series_trajectories(PLTS, iters, current_traj_data, nl_loc, opt_loc, ax, pens):
    # plot iters if specified
    if iters != [-1]:

        # plot the nonlinear propagation of  openloop solution
        ins_nl = {'label': 'Nonlinear Propagation', 'x': 't_nl', 'y': 'values', 'iters': iters, 'dataloc': nl_loc, 'legend': "legend1"}
        
        PLTS.addPlot2D(ax, pen=pens.itr_nl, ins=ins_nl)

        # plot the optimal solution
        ins_opt = {'label': 'Optimal Soltution', 'x': 't_opt', 'y': 'values', 'iters': iters, 'dataloc': opt_loc,'legend': "legend1"}
        PLTS.addPlot2D(ax, pen=pens.itr_opt, ins=ins_opt)

    
    # plot the nonlinear propagation of  openloop solution
    ins_nl = {'label': 'Nonlinear Propagation', 'x': 't_nl', 'y': 'values', 'iters': [-1], 'dataloc': nl_loc, 'legend': "legend1"}
    PLTS.addPlot2D(ax, pen=pens.nl, ins=ins_nl)

    # plot the optimal solution
    ins_opt = {'label': 'Optimal Soltution', 'x': 't_opt', 'y': 'values', 'iters': [-1], 'dataloc': opt_loc, 'legend': "legend1"}
    PLTS.addPlot2D(ax, pen=pens.opt, ins=ins_opt)

def plot_animated(trajopt_obj):
    pass

def create_grid(num_groups):
    num_columns = int(np.ceil(np.sqrt(num_groups)))
    num_rows = int(np.ceil(num_groups / num_columns))


    dx = (0.8 - (num_columns - 1) * plot_options.grid_gap_x) / num_columns
    dy = (0.8 - (num_rows - 1) * plot_options.grid_gap_y) / num_rows

    grid = {}

    for i in range(num_rows):
        for j in range(num_columns):
            tag = i * num_columns + j
            x = 0.05 + j * (dx + plot_options.grid_gap_x)
            y = 0.95 - (i + 1) * dy - i * plot_options.grid_gap_y
            grid[tag] = [x, y, dx, dy]
    
    return grid