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
    
    # method1 pens
    'opt1':    {'frgba': [0,0,0,.1], 'lrgba': [0,0,1,1.],   'lw': 1, 'ls': '',   'msty': 'o', 'msz': 2},
    'nl1':     {'frgba': [0,0,0,.1], 'lrgba': [0,0,1,1.],   'lw': 1, 'ls': '-',  'msty': '',  'msz': 3},

    # method2 pens
    'opt2':    {'frgba': [0,0,0,.1], 'lrgba': [1,0,1,1.],   'lw': 1, 'ls': '',   'msty': 'o', 'msz': 2},
    'nl2':     {'frgba': [0,0,0,.1], 'lrgba': [1,0,1,1.],   'lw': 1, 'ls': '-',  'msty': '',  'msz': 3},

    # method3 pens
    'opt3':    {'frgba': [0,0,0,.1], 'lrgba': [0,1,0,1.],   'lw': 1, 'ls': '',   'msty': 'o', 'msz': 2},
    'nl3':     {'frgba': [0,0,0,.1], 'lrgba': [0,1,0,1.],   'lw': 1, 'ls': '-',  'msty': '',  'msz': 3},

    # TODO: make these not hardcoded lol, or put them in a config file or something
})

def plot_default(trajopt_obj, analysis_type='standalone', show_iters=False):
    data = trajopt_obj.results
    
    # create trajplots object
    PLTS = SCVXPLOTS(data)

    method_keys = list(trajopt_obj.results.keys())
    first_method = method_keys[0]

    nominal_iter_data = trajopt_obj.results[method_keys[0]]["runs"][-1]["iters"][-1]

    # decide whether or not to plot the iterations
    if show_iters:
        iters = slice(1, None)
    else:
        iters = [-1]

    problem = trajopt_obj.problem
    problem_config = problem.config.problem

    # extract plotting groups from configs
    default_groups_state   = {f'State {i}': [i] for i in range(problem.index_map.n.state)}
    default_groups_control = {f'Control {i}': [i] for i in range(problem.index_map.n.nu)}
    groups_state           = problem_config.get('state', default_groups_state)
    groups_control         = problem_config.get('control', default_groups_control)

    # extract constraint_groups
    nominal_constraint_data = nominal_iter_data["constraint_data"]
    nominal_trajectory_data = nominal_iter_data["trajectory_data"]
    nominal_W_data    = nominal_iter_data["W"]
    nominal_dual_data = nominal_iter_data["dual"]

    # define figures and axes for each type of plot
    fig_control     = plt.figure(figsize=plot_options.figsize, dpi=plot_options.dpi)
    fig_state       = plt.figure(figsize=plot_options.figsize, dpi=plot_options.dpi)

    axs_state       = PLTS.createGrid(fig_state ,grid=create_grid(len(groups_state.keys())))
    axs_control     = PLTS.createGrid(fig_control ,grid=create_grid(len(groups_control.keys())))

    # create figs and axes for groups of constraints
    figs_constraints = AttrDict({})
    axs_constraints  = AttrDict({})

    for constraint_group_name, constraint_group_data in nominal_constraint_data.items():
        figs_constraints[constraint_group_name] = plt.figure(figsize=plot_options.figsize, dpi=plot_options.dpi)

        num_constraints_in_group = len(constraint_group_data.keys())
        axs_constraints[constraint_group_name] = PLTS.createGrid(figs_constraints[constraint_group_name] ,grid=create_grid(num_constraints_in_group))

    # create figs and axes for groups of trajectories
    figs_trajectories = AttrDict({})
    axs_trajectories  = AttrDict({})

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
        axs_trajectories[trajectory_group_name] = PLTS.createGrid2(figs_trajectories[trajectory_group_name], grid=create_grid(num_trajectories_in_group), ins={'plt_typs':plt_type})

    # WEIGHTS OVER TIME PLOTS
    figs_weights_time = AttrDict({})
    axs_weights_time  = AttrDict({})

    # NOTE: this is pulling out the weights that are used by the alg and also checking if the weights are applied across time (i.e. filtering out terminal weights for example)
    # this is a bit hacky, i think a better solution is to store the weights to always have a time dimension even if they're just terminal weights, but this works for now
    t_steps = nominal_iter_data["t_opt"].shape[0]
    active_penalty_names_time = [key for key, val in nominal_W_data.items() if val.size > 0 and (nominal_W_data[key].shape[0] == t_steps or nominal_W_data[key].shape[0] == t_steps - 1)]
    active_dual_names_time   = [key for key, val in nominal_dual_data.items() if val.size > 0 and (nominal_dual_data[key].shape[0] == t_steps or nominal_dual_data[key].shape[0] == t_steps - 1)]

    weight_groups_time = AttrDict({})

    for name in active_penalty_names_time:
        if name not in weight_groups_time:
            weight_groups_time[name] = []
        weight_groups_time[name].append("W")
    
    for name in active_dual_names_time:
        if name not in weight_groups_time:
            weight_groups_time[name] = []
        weight_groups_time[name].append("dual")

    for weight_group_name, weight_group in weight_groups_time.items():
        figs_weights_time[weight_group_name] = plt.figure(figsize=plot_options.figsize, dpi=plot_options.dpi)
        axs_weights_time[weight_group_name]  = PLTS.createGrid(figs_weights_time[weight_group_name], grid=create_grid(len(weight_group)))
    
    # WEIGHTS OVER ITERS PLOTS
    figs_weights_iters = AttrDict({})
    axs_weights_iters  = AttrDict({})

    active_penalty_names_iters = [key for key, val in nominal_W_data.items() if val.size > 0]
    active_dual_names_iters   = [key for key, val in nominal_dual_data.items() if val.size > 0]

    weight_groups_iters = AttrDict({})

    for name in active_penalty_names_iters:
        if name not in weight_groups_iters:
            weight_groups_iters[name] = []
        weight_groups_iters[name].append("W")
    
    for name in active_dual_names_iters:
        if name not in weight_groups_iters:
            weight_groups_iters[name] = []
        weight_groups_iters[name].append("dual")

    for weight_name, weight_group in weight_groups_iters.items():
        figs_weights_iters[weight_name] = plt.figure(figsize=plot_options.figsize, dpi=plot_options.dpi)
        axs_weights_iters[weight_name]  = PLTS.createGrid(figs_weights_iters[weight_name], grid=create_grid(len(weight_group)))

    # plot the data onto the figs and axes created above
    if analysis_type == 'standalone':
        standalone_pens = AttrDict({"opt": pens.opt, "nl": pens.nl, "itr_opt": pens.itr_opt, "itr_nl": pens.itr_nl, "init": pens.init})
        
        # plot states, controls, trajectories, and constraints on their respective figures
        plot_states(PLTS, axs_state, groups_state, method=first_method, run=0, iters=iters, pens=standalone_pens)
        plot_controls(PLTS, axs_control, groups_control, method=first_method, run=0, pens=standalone_pens, iters=iters)
        plot_trajectories(PLTS, axs_trajectories, nominal_trajectory_data, method=first_method, run=0, iters=iters, pens=standalone_pens)
        plot_constraints(PLTS, axs_constraints, nominal_constraint_data, method=first_method, run=0, iters=iters, pens=standalone_pens)
        plot_weights_time(PLTS, axs_weights_time, weight_groups_time, method=first_method, run=0, iters=iters, pens=standalone_pens)
        plot_weights_iters(PLTS, axs_weights_iters, weight_groups_iters, method=first_method, run=0, pens=standalone_pens)

    elif analysis_type == 'mc':
        for i, (method_name, method_data) in enumerate(trajopt_obj.results.items()):
            current_method_pens = AttrDict({"opt": pens[f"opt{i+1}"], "nl": pens[f"nl{i+1}"], "itr_opt": pens.itr_opt, "itr_nl": pens.itr_nl, "init": pens.init})
            for run_number in range(len(method_data['runs'])):
                plot_states(PLTS, axs_state, groups_state, method=method_name, run=run_number, iters=[-1], pens=current_method_pens)
                plot_controls(PLTS, axs_control, groups_control, pens=current_method_pens, method=method_name, run=run_number, iters=[-1])
                plot_trajectories(PLTS, axs_trajectories, nominal_trajectory_data, method=method_name, run=run_number, iters=[-1], pens=current_method_pens)
                plot_constraints(PLTS, axs_constraints, nominal_constraint_data, method=method_name, run=run_number, iters=[-1], pens=current_method_pens)
                plot_weights_time(PLTS, axs_weights_time, weight_groups_time, method=method_name, run=run_number, iters=iters, pens=current_method_pens)
                plot_weights_iters(PLTS, axs_weights_iters, weight_groups_iters, method=method_name, run=run_number, pens=current_method_pens)

    plt.show()

def plot_states(PLTS, axs, groups, method, run, iters, 
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
                ins_nl = {'label': 'Nonlinear Propagation', 'x': 't_nl', 'y': 'z_nl', 'x_idx': slice(None), 'y_idx': index, 'method': method, 'run': run, 'iters': iters, 'legend': "legend1"}
                PLTS.addPlot2D(ax, pen=pens.itr_nl, ins=ins_nl)

                # plot the optimal solution
                ins_opt = {'label': 'Optimal Soltution', 'x': 't_opt', 'y': 'z_opt', 'x_idx': slice(None), 'y_idx': index, 'method': method, 'run': run, 'iters': iters, 'legend': "legend1"}
                PLTS.addPlot2D(ax, pen=pens.itr_opt, ins=ins_opt)
            
            # plot the nonlinear propagation of  openloop solution
            ins_nl = {'label': 'Nonlinear Propagation', 'x': 't_nl', 'y': 'z_nl', 'x_idx': slice(None), 'y_idx': index, 'method': method, 'run': run, 'iters': np.array([-1]), 'legend': "legend1"}
            PLTS.addPlot2D(ax, pen=pens.nl, ins=ins_nl)

            # plot the optimal solution
            ins_opt = {'label': 'Optimal Soltution', 'x': 't_opt', 'y': 'z_opt', 'x_idx': slice(None), 'y_idx': index, 'method': method, 'run': run, 'iters': np.array([-1]), 'legend': "legend1"}
            PLTS.addPlot2D(ax, pen=pens.opt, ins=ins_opt)

def plot_controls(PLTS, axs, groups, pens, method, run, iters):

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
                ins_nl = {'label': 'Nonlinear Propagation', 'x': 't_nl', 'y': 'nu_nl', 'x_idx': slice(None), 'y_idx': index, 'method': method, 'run': run, 'iters': iters, 'legend': "legend1"}
                PLTS.addPlot2D(ax, pen=pens.itr_nl, ins=ins_nl)

                # plot the optimal solution
                ins_opt = {'label': 'Optimal Soltution', 'x': 't_opt', 'y': 'nu_opt', 'x_idx': slice(None), 'y_idx': index, 'method': method, 'run': run, 'iters': iters, 'legend': "legend1"}
                PLTS.addPlot2D(ax, pen=pens.itr_opt, ins=ins_opt)

            # plot the nonlinear propagation of  openloop solution
            ins_nl = {'label': 'Nonlinear Propagation', 'x': 't_nl', 'y': 'nu_nl', 'x_idx': slice(None), 'y_idx': index, 'method': method, 'run': run, 'iters': [-1], 'legend': "legend1"}
            PLTS.addPlot2D(ax, pen=pens.nl, ins=ins_nl)

            # plot the optimal solution
            ins_opt = {'label': 'Optimal Soltution', 'x': 't_opt', 'y': 'nu_opt', 'x_idx': slice(None), 'y_idx': index, 'method': method, 'run': run, 'iters': [-1], 'legend': "legend1"}
            PLTS.addPlot2D(ax, pen=pens.opt, ins=ins_opt)

def plot_constraints(PLTS, axs, nominal_constraint_data, method, run, iters,
                pens = AttrDict({"opt": pens.opt, "nl": pens.nl, "itr_opt": pens.itr_opt, "itr_nl": pens.itr_nl})):

    for constraint_group_name, constraint_group_data in nominal_constraint_data.items():
        # extract the current group subplot axis
        for i, (constraint_name, current_constraint_data) in enumerate(constraint_group_data.items()):
            ax = axs[constraint_group_name][i]

            # set properties for this subplot
            ax.set_title(constraint_name)
            ax.grid(True, alpha=0.3)

            opt_vals_path = f"constraint_data.{constraint_group_name}.{constraint_name}.opt_vals.values"
            nl_vals_path = f"constraint_data.{constraint_group_name}.{constraint_name}.nl_vals.values"

            limits_path = f"constraint_data.{constraint_group_name}.{constraint_name}.nl_vals.limits"

            # plot iters if specified
            if iters != [-1]:
                # plot the nonlinear propagation of  openloop solution
                ins_nl = {'label': 'Nonlinear Propagation', 'x': 't_nl', 'y': nl_vals_path, 'x_idx': slice(None), 'y_idx': slice(None), 'method': method, 'run': run, 'iters': iters, 'legend': "legend1"}
                PLTS.addPlot2D(ax, pen=pens.itr_nl, ins=ins_nl)

                # plot the optimal solution
                ins_opt = {'label': 'Optimal Soltution', 'x': 't_opt', 'y': opt_vals_path, 'x_idx': slice(None), 'y_idx': slice(None), 'method': method, 'run': run, 'iters': iters, 'legend': "legend1"}
                PLTS.addPlot2D(ax, pen=pens.itr_opt, ins=ins_opt)
            
            # plot the nonlinear propagation of  openloop solution
            ins_nl = {'label': 'Nonlinear Propagation', 'x': 't_nl', 'y': nl_vals_path, 'x_idx': slice(None), 'y_idx': slice(None), 'method': method, 'run': run, 'iters': [-1], 'legend': "legend1"}
            PLTS.addPlot2D(ax, pen=pens.nl, ins=ins_nl)

            # plot the optimal solution
            ins_opt = {'label': 'Optimal Soltution', 'x': 't_opt', 'y': opt_vals_path, 'x_idx': slice(None), 'y_idx': slice(None), 'method': method, 'run': run, 'iters': [-1], 'legend': "legend1"}
            PLTS.addPlot2D(ax, pen=pens.opt, ins=ins_opt)

            # plot constraint limits
            if nominal_constraint_data[constraint_group_name][constraint_name]['opt_vals']["limits"] is not None:
                    ins_limits = {'label': 'Limits', 'x': 't_nl', 'y': limits_path, 'x_idx': slice(None), 'y_idx': slice(None), 'method': method, 'run': run, 'iters': iters, 'legend': "legend1"}
                    PLTS.addPlot2D(ax, pen=pens.init, ins=ins_limits)

def plot_trajectories(PLTS, axs, nominal_trajectory_data, method, run=0, iters=[-1], 
                pens = AttrDict({"opt": pens.opt, "nl": pens.nl, "itr_opt": pens.itr_opt, "itr_nl": pens.itr_nl})):

    for traj_group_name, traj_group_data in nominal_trajectory_data.items():
        
        # extract the current group subplot axis
        for i, (traj_name, current_traj_data) in enumerate(traj_group_data.items()):
            ax = axs[traj_group_name][i]

            # set properties for this subplot
            ax.set_title(traj_name)
            ax.grid(True, alpha=0.3)

            opt_vals_path = f"trajectory_data.{traj_group_name}.{traj_name}.opt_vals.values"
            nl_vals_path = f"trajectory_data.{traj_group_name}.{traj_name}.nl_vals.values"

            traj_type = current_traj_data.type

            if traj_type == "spatial":

                plot_spatial_trajectories(PLTS, method, run, iters, current_traj_data, nl_vals_path, opt_vals_path, ax, pens)

            elif traj_type == "time_series":

                plot_time_series_trajectories(PLTS, method, run, iters, current_traj_data, nl_vals_path, opt_vals_path, ax, pens)

def plot_spatial_trajectories(PLTS, method, run, iters, current_traj_data, nl_vals_path, opt_vals_path, ax, pens):
    dim = current_traj_data.opt_vals["values"].shape[1]
    # plot iters if specified
    if iters != [-1]:

        if dim == 3:
            # plot the nonlinear propagation of  openloop solution
            ins_nl = {'label': 'Nonlinear Propagation', 'x': nl_vals_path, 'x_idx': 0, 'y': nl_vals_path, 'y_idx': 1, 'z': nl_vals_path, 'z_idx': 2, 'method': method, 'run': run, 'iters': iters, 'legend': "legend1"}
            
            PLTS.addPlot3D(ax, pen=pens.itr_nl, ins=ins_nl)

            # plot the optimal solution
            ins_opt = {'label': 'Optimal Soltution', 'x': opt_vals_path, 'x_idx': 0, 'y': opt_vals_path, 'y_idx': 1, 'z': opt_vals_path, 'z_idx': 2, 'method': method, 'run': run, 'iters': iters,'legend': "legend1"}
            PLTS.addPlot3D(ax, pen=pens.itr_opt, ins=ins_opt)
        
        else:
            # plot the nonlinear propagation of  openloop solution
            ins_nl = {'label': 'Nonlinear Propagation', 'x': nl_vals_path, 'x_idx': 0, 'y': nl_vals_path, 'y_idx': 1, 'method': method, 'run': run, 'iters': iters, 'legend': "legend1"}
            
            PLTS.addPlot2D(ax, pen=pens.itr_nl, ins=ins_nl)

            # plot the optimal solution
            ins_opt = {'label': 'Optimal Soltution', 'x': opt_vals_path, 'x_idx': 0, 'y': opt_vals_path, 'y_idx': 1, 'method': method, 'run': run, 'iters': iters, 'legend': "legend1"}
            PLTS.addPlot2D(ax, pen=pens.itr_opt, ins=ins_opt)

    if dim == 3:
        # plot the nonlinear propagation of  openloop solution
        ins_nl = {'label': 'Nonlinear Propagation', 'x': nl_vals_path, 'x_idx': 0, 'y': nl_vals_path, 'y_idx': 1, 'z': nl_vals_path, 'z_idx': 2, 'method': method, 'run': run, 'iters': [-1], 'legend': "legend1"}
        PLTS.addPlot3D(ax, pen=pens.nl, ins=ins_nl)

        # plot the optimal solution
        ins_opt = {'label': 'Optimal Soltution', 'x': opt_vals_path, 'x_idx': 0, 'y': opt_vals_path, 'y_idx': 1, 'z': opt_vals_path, 'z_idx': 2, 'method': method, 'run': run, 'iters': [-1],'legend': "legend1"}
        PLTS.addPlot3D(ax, pen=pens.opt, ins=ins_opt)
    
    else:
        # plot the nonlinear propagation of  openloop solution
        ins_nl = {'label': 'Nonlinear Propagation', 'x': nl_vals_path, 'x_idx': 0, 'y': nl_vals_path, 'y_idx': 1, 'method': method, 'run': run, 'iters': [-1], 'legend': "legend1"}
        PLTS.addPlot2D(ax, pen=pens.nl, ins=ins_nl)

        # plot the optimal solution
        ins_opt = {'label': 'Optimal Soltution', 'x': opt_vals_path, 'x_idx': 0, 'y': opt_vals_path, 'y_idx': 1, 'method': method, 'run': run, 'iters': [-1], 'legend': "legend1"}
        PLTS.addPlot2D(ax, pen=pens.opt, ins=ins_opt)

def plot_time_series_trajectories(PLTS, method, run, iters, current_traj_data, nl_vals_path, opt_vals_path, ax, pens):
    # plot iters if specified
    if iters != [-1]:

        # plot the nonlinear propagation of  openloop solution
        ins_nl = {'label': 'Nonlinear Propagation', 'x': 't_nl', 'y': nl_vals_path, 'method': method, 'run': run, 'iters': iters, 'legend': "legend1"}
        
        PLTS.addPlot2D(ax, pen=pens.itr_nl, ins=ins_nl)

        # plot the optimal solution
        ins_opt = {'label': 'Optimal Soltution', 'x': 't_opt', 'y': opt_vals_path, 'method': method, 'run': run, 'iters': iters, 'legend': "legend1"}
        PLTS.addPlot2D(ax, pen=pens.itr_opt, ins=ins_opt)

    
    # plot the nonlinear propagation of  openloop solution
    ins_nl = {'label': 'Nonlinear Propagation', 'x': 't_nl', 'y': nl_vals_path, 'method': method, 'run': run, 'iters': [-1], 'legend': "legend1"}
    PLTS.addPlot2D(ax, pen=pens.nl, ins=ins_nl)

    # plot the optimal solution
    ins_opt = {'label': 'Optimal Soltution', 'x': 't_opt', 'y': opt_vals_path, 'method': method, 'run': run, 'iters': [-1], 'legend': "legend1"}
    PLTS.addPlot2D(ax, pen=pens.opt, ins=ins_opt)

def plot_weights_time(PLTS, axs, weight_groups,
                method=None,
                run=0,
                iters=[-1], 
                pens = AttrDict({"opt": pens.opt, "nl": pens.nl, "itr_opt": pens.itr_opt, "itr_nl": pens.itr_nl})):
    
    for name, weight_key_list in weight_groups.items():
        for i, weight_key in enumerate(weight_key_list):
            ax = axs[name][i]
            w_path = f"{weight_key}.{name}"

            ins_opt = {'label': 'Optimal Soltution', 'x': 't_opt', 'y': w_path, 'method': method, 'run': run, 'iters': iters, 'legend': "legend1"}
            PLTS.addPlot2D(ax, pen=pens.itr_opt, ins=ins_opt)

def plot_weights_iters(PLTS, axs, weight_groups, method, run,
                pens = AttrDict({"opt": pens.opt, "nl": pens.nl, "itr_opt": pens.itr_opt, "itr_nl": pens.itr_nl})):
    
    for name, weight_key_list in weight_groups.items():
        for i, weight_key in enumerate(weight_key_list):
            ax = axs[name][i]
            w_path = f"{weight_key}.{name}"

            ins_opt = {'label': 'Optimal Soltution', 'x': 't_opt', 'y': w_path, 'method': method, 'run': run, 'legend': "legend1"}
            PLTS.addPlot2D_iters(ax, pen=pens.itr_opt, ins=ins_opt)

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