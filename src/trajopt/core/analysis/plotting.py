import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from trajopt.core.analysis.trajplots import SCVXPLOTS
from trajopt.utils.tools import AttrDict, recursive_attrdict

plt.rcParams["text.usetex"] = False
plt.rcParams.update({
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'lines.linewidth': 1.0,
})

plot_options = AttrDict({
    'figsize': (10, 3.5),
    'dpi': 300,
    'grid_gap_x': 0.06,
    'grid_gap_y': 0.18,
    'margins': [0.07, 0.02, 0.08, 0.12],
    'title_fontsize': 9,
    'title_pad': 6,
})

# pen: frgba, lrgba, lw, ls, msty, msz
pens = recursive_attrdict({
    'init':    {'frgba': [0,0,0,.1], 'lrgba': [0,0,0,1.],   'lw': 1, 'ls': '--', 'msty': '',  'msz': 3},
    'nl':      {'frgba': [0,0,0,.1], 'lrgba': [1,0,0,1.],   'lw': 2, 'ls': '-',  'msty': '',  'msz': 3},
    'opt':     {'frgba': [0,0,0,.1], 'lrgba': [0,0,1,1.],   'lw': 1, 'ls': '',   'msty': 'o', 'msz': 3},
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

def plot_default(trajopt_obj, show_iters=False):
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

    # read per-group plot configuration from YAML (if present)
    pcfg = problem_config.get('plot_config', {})
    state_pcfg = pcfg.get('state', {})
    control_pcfg = pcfg.get('control', {})
    traj_group_pcfg = pcfg.get('trajectory_groups', {})

    # define figures and axes for each type of plot
    fig_state   = plt.figure(figsize=state_pcfg.get('figsize', plot_options.figsize), dpi=plot_options.dpi)
    fig_control = plt.figure(figsize=control_pcfg.get('figsize', plot_options.figsize), dpi=plot_options.dpi)

    axs_state       = PLTS.createGrid(fig_state ,grid=create_grid(len(groups_state.keys()), cfg=state_pcfg))
    axs_control     = PLTS.createGrid(fig_control ,grid=create_grid(len(groups_control.keys()), cfg=control_pcfg))

    # # create figs and axes for groups of constraints
    # figs_constraints = AttrDict({})
    # axs_constraints  = AttrDict({})

    # for constraint_group_name, constraint_group_data in nominal_constraint_data.items():
    #     figs_constraints[constraint_group_name] = plt.figure(figsize=plot_options.figsize, dpi=plot_options.dpi)

    #     num_constraints_in_group = len(constraint_group_data.keys())
    #     axs_constraints[constraint_group_name] = PLTS.createGrid(figs_constraints[constraint_group_name] ,grid=create_grid(num_constraints_in_group))

    # create figs and axes for groups of trajectories
    figs_trajectories = AttrDict({})
    axs_trajectories  = AttrDict({})

    plt_type = {}
    for trajectory_group_name, trajectory_group_data in nominal_trajectory_data.items():
        grp_cfg = traj_group_pcfg.get(trajectory_group_name, {})
        figs_trajectories[trajectory_group_name] = plt.figure(figsize=grp_cfg.get('figsize', plot_options.figsize), dpi=plot_options.dpi)

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
        grid2_ins = {'plt_typs': plt_type}
        if 'pad_3d' in grp_cfg:
            grid2_ins['pad_3d'] = grp_cfg['pad_3d']
        axs_trajectories[trajectory_group_name] = PLTS.createGrid2(figs_trajectories[trajectory_group_name], grid=create_grid(num_trajectories_in_group, cfg=grp_cfg), ins=grid2_ins)

    # WEIGHTS OVER TIME PLOTS
    figs_weights_time = AttrDict({})
    axs_weights_time  = AttrDict({})

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
    if trajopt_obj.analysis_type == 'standalone':
        standalone_pens = AttrDict({"opt": pens.opt, "nl": pens.nl, "itr_opt": pens.itr_opt, "itr_nl": pens.itr_nl, "init": pens.init})
        
        # plot states, controls, trajectories, and constraints on their respective figures
        plot_states(PLTS, axs_state, groups_state, method=first_method, run=0, iters=iters, pens=standalone_pens)
        plot_controls(PLTS, axs_control, groups_control, method=first_method, run=0, pens=standalone_pens, iters=iters)
        plot_trajectories(PLTS, axs_trajectories, nominal_trajectory_data, method=first_method, run=0, iters=iters, pens=standalone_pens)
        # plot_constraints(PLTS, axs_constraints, nominal_constraint_data, method=first_method, run=0, iters=iters, pens=standalone_pens)
        plot_weights_time(PLTS, axs_weights_time, weight_groups_time, method=first_method, run=0, iters=iters, pens=standalone_pens)
        plot_weights_iters(PLTS, axs_weights_iters, weight_groups_iters, method=first_method, run=0, pens=standalone_pens)

    elif trajopt_obj.analysis_type == 'mc':
        for i, (method_name, method_data) in enumerate(trajopt_obj.results.items()):
            current_method_pens = AttrDict({"opt": pens[f"opt{i+1}"], "nl": pens[f"nl{i+1}"], "itr_opt": pens.itr_opt, "itr_nl": pens.itr_nl, "init": pens.init})
            for run_number in range(len(method_data['runs'])):
                plot_states(PLTS, axs_state, groups_state, method=method_name, run=run_number, iters=[-1], pens=current_method_pens)
                plot_controls(PLTS, axs_control, groups_control, pens=current_method_pens, method=method_name, run=run_number, iters=[-1])
                plot_trajectories(PLTS, axs_trajectories, nominal_trajectory_data, method=method_name, run=run_number, iters=[-1], pens=current_method_pens)
                # plot_constraints(PLTS, axs_constraints, nominal_constraint_data, method=method_name, run=run_number, iters=[-1], pens=current_method_pens)
                plot_weights_time(PLTS, axs_weights_time, weight_groups_time, method=method_name, run=run_number, iters=iters, pens=current_method_pens)
                plot_weights_iters(PLTS, axs_weights_iters, weight_groups_iters, method=method_name, run=run_number, pens=current_method_pens)

    # build figure-level legend handles
    all_figs = {"state": fig_state, "control": fig_control}
    all_figs.update({f"trajectory_{k}": v for k, v in figs_trajectories.items()})
    all_figs.update({f"weights_time_{k}": v for k, v in figs_weights_time.items()})
    all_figs.update({f"weights_iters_{k}": v for k, v in figs_weights_iters.items()})

    def _handle(pen, label):
        p = pens[pen]
        ls = p.ls if p.ls else 'None'
        return Line2D([], [], color=p.lrgba[:3], alpha=p.lrgba[3], lw=p.lw,
                       ls=ls, marker=p.msty or None, markersize=p.msz, label=label)

    if trajopt_obj.analysis_type == 'standalone':
        handles = [_handle('opt', 'Optimal'), _handle('nl', 'Nonlinear')]
        if show_iters:
            handles.append(_handle('itr_nl', 'Iterations'))
    elif trajopt_obj.analysis_type == 'mc':
        handles = [_handle(f'nl{i+1}', name) for i, name in enumerate(method_keys)]

    for fig in all_figs.values():
        first_ax = fig.axes[0]
        first_ax.legend(handles=handles, loc='best', fontsize=8, framealpha=0.8)

    # save all figures
    atype = trajopt_obj.analysis_type
    save_dir = os.path.join("plots", atype)
    os.makedirs(save_dir, exist_ok=True)

    for name, fig in all_figs.items():
        fig.savefig(os.path.join(save_dir, f"{name}.pdf"), bbox_inches="tight")

    print(f"Saved {len(all_figs)} figures to {save_dir}/")
    plt.show()

def plot_states(PLTS, axs, groups, method, run, iters, 
                pens = AttrDict({"opt": pens.opt, "nl": pens.nl, "itr_opt": pens.itr_opt, "itr_nl": pens.itr_nl})):

    for i, (group_name, group_data) in enumerate(groups.items()):
        # extract the current group subplot axis
        ax = axs[i]

        group_indices = group_data["idx"]

        # set properties for this subplot
        if "title" in group_data:
            ax.set_title(group_data["title"], fontsize=plot_options.title_fontsize, pad=plot_options.title_pad)
        ax.grid(True, alpha=0.3)
        if "xlabel" in group_data:
            ax.set_xlabel(group_data["xlabel"])
        if "ylabel" in group_data:
            ax.set_ylabel(group_data["ylabel"])

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
        if "title" in group_data:
            ax.set_title(group_data["title"], fontsize=plot_options.title_fontsize, pad=plot_options.title_pad)
        ax.grid(True, alpha=0.3)
        if "xlabel" in group_data:
            ax.set_xlabel(group_data["xlabel"])
        if "ylabel" in group_data:
            ax.set_ylabel(group_data["ylabel"])

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
            if current_traj_data.get("title"):
                ax.set_title(current_traj_data["title"], fontsize=plot_options.title_fontsize, pad=plot_options.title_pad)
            ax.grid(True, alpha=0.3)
            if current_traj_data.get("xlabel"):
                ax.set_xlabel(current_traj_data["xlabel"])
            if current_traj_data.get("ylabel"):
                ax.set_ylabel(current_traj_data["ylabel"])
            if current_traj_data.get("tick_nbins"):
                nbins = current_traj_data["tick_nbins"]
                ax.xaxis.set_major_locator(MaxNLocator(nbins=nbins))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins))
                if hasattr(ax, 'zaxis'):
                    ax.zaxis.set_major_locator(MaxNLocator(nbins=nbins))

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

    # plot limits if specified
    limits = current_traj_data.opt_vals.get("limits", {})
    if limits:
        for val in [limits.get("upper"), limits.get("lower")]:
            if val is not None:
                ax.axhline(val, color='k', ls='--', lw=1, alpha=0.5)

WEIGHT_DISPLAY_NAMES = {"W": "Penalty Weights", "dual": "Dual Weights"}

def plot_weights_time(PLTS, axs, weight_groups,
                method=None,
                run=0,
                iters=[-1], 
                pens = AttrDict({"opt": pens.opt, "nl": pens.nl, "itr_opt": pens.itr_opt, "itr_nl": pens.itr_nl})):
    
    for name, weight_key_list in weight_groups.items():
        for i, weight_key in enumerate(weight_key_list):
            ax = axs[name][i]
            w_path = f"{weight_key}.{name}"

            display_name = name.replace('_', ' ').title()
            ax.set_title(f"{display_name} — {WEIGHT_DISPLAY_NAMES[weight_key]}", fontsize=plot_options.title_fontsize, pad=plot_options.title_pad)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Weight")
            ax.grid(True, alpha=0.3)

            ins_opt = {'label': 'Optimal Soltution', 'x': 't_opt', 'y': w_path, 'method': method, 'run': run, 'iters': iters, 'legend': "legend1"}
            PLTS.addPlot2D(ax, pen=pens.itr_opt, ins=ins_opt)

def plot_weights_iters(PLTS, axs, weight_groups, method, run,
                pens = AttrDict({"opt": pens.opt, "nl": pens.nl, "itr_opt": pens.itr_opt, "itr_nl": pens.itr_nl})):
    
    for name, weight_key_list in weight_groups.items():
        for i, weight_key in enumerate(weight_key_list):
            ax = axs[name][i]
            w_path = f"{weight_key}.{name}"

            display_name = name.replace('_', ' ').title()
            ax.set_title(f"{display_name} — {WEIGHT_DISPLAY_NAMES[weight_key]}", fontsize=plot_options.title_fontsize, pad=plot_options.title_pad)
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Weight")
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

            ins_opt = {'label': 'Optimal Soltution', 'x': 't_opt', 'y': w_path, 'method': method, 'run': run, 'legend': "legend1"}
            PLTS.addPlot2D_iters(ax, pen=pens.itr_opt, ins=ins_opt)

def plot_animated(trajopt_obj):
    pass

def create_grid(num_groups, cfg=None):
    if cfg is None:
        cfg = {}

    if num_groups == 3:
        num_columns = 3
        num_rows = 1
    else:
        num_columns = int(np.ceil(np.sqrt(num_groups)))
        num_rows = int(np.ceil(num_groups / num_columns))

    gap_x = cfg.get('grid_gap_x', plot_options.grid_gap_x)
    gap_y = cfg.get('grid_gap_y', plot_options.grid_gap_y)
    margins = cfg.get('margins', plot_options.margins)
    margin_left, margin_right, margin_top, margin_bottom = margins

    usable_w = 1.0 - margin_left - margin_right
    usable_h = 1.0 - margin_top - margin_bottom

    width_ratios = cfg.get('width_ratios', None)

    dy = (usable_h - (num_rows - 1) * gap_y) / num_rows

    grid = {}

    if width_ratios and len(width_ratios) == num_columns:
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