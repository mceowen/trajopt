
PENS = {};

# PENS['z_opt'] = {'frgba':[0,0,0,0.1],'lrgba':[0,0,0,0.1],'lw':2,'ls':'-','msty':'','msz':4};
# standalone 
PENS['init'] = {'frgba':[.0,.0,.0,.1],'lrgba':[.0,.0,.0,1.],'lw':1,'ls':'--','msty':'' ,'msz':3};
PENS['nl'] = {'frgba':[.0,.0,.0,.1],'lrgba':[1.,.0,.0,1.],'lw':2,'ls':'-' ,'msty':'' ,'msz':3};
PENS['opt']  = {'frgba':[.0,.0,.0,.1],'lrgba':[.0,.0,1.,1.],'lw':1,'ls':''  ,'msty':'o','msz':3};
# iteration values
PENS['itr_opt']  = {'frgba':[.0,.0,.0,.1],'lrgba':[0.7,.0,0.3,.2],'lw':1,'ls':'','msty':'o' ,'msz':3};
PENS['itr_nl']   = {'frgba':[.0,.0,.0,.1],'lrgba':[0.7,.0,0.3,.4],'lw':1,'ls':'-','msty':'' ,'msz':3};
# final iteration values
PENS['fitr_opt']  = {'frgba':[.0,.0,.0,.1],'lrgba':[.0,.0,1.,1.],'lw':2,'ls':'','msty':'o' ,'msz':3};
PENS['fitr_nl']   = {'frgba':[.0,.0,.0,.1],'lrgba':[0.,.0,1.,1.],'lw':2,'ls':'-','msty':'' ,'msz':3};

# convergence
PENS['opt2']  = {'frgba':[.0,.0,.0,.1],'lrgba':[.0,.0,1.,1.],'lw':1,'ls':'-'  ,'msty':'o','msz':3};
PENS['fitr_opt2']  = {'frgba':[.0,.0,.0,.1],'lrgba':[.0,.0,1.,1.],'lw':1,'ls':'-'  ,'msty':'o','msz':3};

PENS['ref']  = {'frgba':[.0,.0,.0,.1],'lrgba':[.0,.0,.0,1.],'lw':1,'ls':'--','msty':'*','msz':3};
PENS['standard']  = {'frgba':[.0,.0,.0,.1],'lrgba':[.0,.0,1.,1.],'lw':1,'ls':'-','msty':'o','msz':3};
PENS['standard_nl']  = {'frgba':[.0,.0,.0,.1],'lrgba':[.0,.0,1.,1.],'lw':1,'ls':'-','msty':'','msz':3};
PENS['standard_opt']  = {'frgba':[.0,.0,.0,.1],'lrgba':[.0,.0,1.,1.],'lw':1,'ls':'','msty':'o','msz':3};
PENS['standard_opt2']  = {'frgba':[.0,.0,.0,.1],'lrgba':[.0,.0,1.,1.],'lw':1,'ls':'-','msty':'','msz':3};

PENS['autotune']  = {'frgba':[.0,.0,.0,.1],'lrgba':[1.,.0,1.,1.],'lw':1,'ls':'-','msty':'o','msz':3};
PENS['autotune_nl']  = {'frgba':[.0,.0,.0,.1],'lrgba':[1.,.0,1.,1.],'lw':1,'ls':'-','msty':'','msz':3};
PENS['autotune_opt']  = {'frgba':[.0,.0,.0,.1],'lrgba':[1.,.0,1.,1.],'lw':1,'ls':'','msty':'o','msz':3};
PENS['autotune_opt2']  = {'frgba':[.0,.0,.0,.1],'lrgba':[1.,.0,1.,1.],'lw':1,'ls':'-','msty':'','msz':3};

# TODO(Skye): Tune this line thickness
PENS['max-value'] = {'frgba':[.0,.0,.0,.1],'lrgba':[0.0,.0,0.,0.7],'lw':2.5,'ls':'-','msty':'','msz':0};

# for 3D plot
PENS['u_vec']  = {'frgba':[.0,.0,.0,.1],'lrgba':[1.,0.25,0.,1.],'lw':2,'ls':'-','msty':'','msz':0};
PENS['body_vec'] = {'frgba':[.0,.0,.0,.1],'lrgba':[0.,0.,0.,1.],'lw':2,'ls':'-','msty':'','msz':0};
# for 2D plots...
PENS['u_vec2']  = {'frgba':[.0,.0,.0,.1],'lrgba':[1.,0.25,0.,1.],'lw':4,'ls':'-','msty':'','msz':0};
PENS['body_vec2'] = {'frgba':[.0,.0,.0,.1],'lrgba':[0.,0.,0.,1.],'lw':4,'ls':'-','msty':'','msz':0};



### potential new data structure
# DATA = {};
# DATA['meta'] = # database information stuff
# DATA['data'] = {('scenario1','auto',run):{'itrs':[]}


## yaml file should have 
# grid info -- how to setup a grid of subplots
# y-axes labels, and subplot titles  - for each subplot
# state name and indices - what to put on each subplot

PLTS1 = SCVXPLOTS(data);

### Manual grid
## make four subplots - [x position,y position, x width , y width]    
grid = {};
grid[0] = [0.05,0.6,0.35,0.35];
grid[1] = [0.51,0.6,0.35,0.35];
grid[2] = [0.05,0.05,0.35,0.35];
grid[3] = [0.51,0.05,0.35,0.35];

### Automatic grid 
def makeSquareGridSpecs(num_states):
  num = num_states;
  colnum = int(np.sqrt(num)) + 1;
  rownum = colnum;
  dx = 0.8/colnum; dy = dx;
  grid = {};
  for i in range(rownum):
    for j in range(colnum):
      x = 0.05 + dx*j; y = 0.05 + dy*i
      grid[i*rownum + colnum] = [x,y,dx,dy];
  return grid
grid = makeSquareGridSpecs(num_states)
########################################


fig = plt.figure(figsize=(10,10));
axs = PLTS1.createGrid(fig,grid = grid); # makes the subplot from the grid info
ax = axs[0] # grab the first subplot
################################################
scenarios = ['scenario1']; methods = ['autotune']; runs = list(range(1000)); iters = list(range(1000))
# set default scenarios, methods, and runs
PLTS1.setCurrent({'scenarios':scenarios,'methods':methods,'runs':runs,'iters':iters}) 

PLTS1.dumpLegend('legend1') # clears info from 'legend1'
#########################################
### all options to pass to addPlot2D and addPlot2DIters
params = {'scenarios': scenarios, # which scenarios to loop through
           'methods': methods, # which methods to loop through
           'runs': runs, # which runs to loop through
           'iters':iters, # which iterations to loop through
           ### only for addPlot2DIter 
           'tinds':[None], # i think this does nothing at the moment... but it does need to be [None]
           #######            
           'x':xtag, # where to get the x data. example: 't_nl', ('z_nl',0), etc
           'y':ytag, # where to get the ydata. 'z_nl',('z_nl',1),('z_nl'(1,2));
           'legend':'legend1', # which legend to put the plot on
           'label': 'label1', # what should the plot be called on the legend.
           'force_lens': True, # if x and y data are different lengths make it work
           'dataloc': 'weights' or 'conv_data', # if ydata not in regular place in iters, where to look for it a level deeper
           'use_quiver': False, #make a quiver plot -- hacky
           'skip': 2, # downsample data points
           }

for j in range(num_states):
  ### add plot of final iterations ie. iters = [-1]
  params = {'label':'state '+str(j),'x':'t_nl','y':('z_nl',j),'iters':[-1],'legend':'legend1'};
  PLTS1.addPLot2D(axs[j],pen=PENS['nl'],ins=params)
  ### add plot of all iterations ie. iters = iters
  params = {'label':'state '+str(j),'x':'t_nl','y':('z_nl',j),'iters':iters,'legend':'legend1'};
  PLTS1.addPLot2D(axs[j],pen=PENS['itr_nl'],ins=params)

params2 = {'label':'Iterations','tinds':[None],'y':tag,'iters':itrs,'legend':'legend1','dataloc':'conv_data'};
PLTS1.addPLot2DIter(ax,pen=PENS['basic_pen2'],ins=params2)


# add these for every subfigure
params = {};
params['title'] = {'text':'title','fontsize':20}
params['xlabel'] = {'label':'Time [$U_T$]','fontsize':16}
params['ylabel'] = {'label':'some state','fontsize':16}
params['ticks'] = {'labelsize':20}

PLTS1.setParams(ax,params)
PLTS1.addLegend(ax,'legend1',ins={'fontsize':12,'loc':'best'});

plt.savefig(filename,bbox_inches='tight',pad_inches = 0,transparent=True);






# reference 31 jan 9:02pm second meeting with dan:


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

# switch pens when doing methodvar
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
        trajopt_obj: The trajectory optimization object with results
        show_iters: If True, overlay all iterations. If False, only show final solution.
    """
    data = {"scenario1": trajopt_obj.results}
    PLTS = SCVXPLOTS(data)
    lgnd = 'legend1'
    iters = list(range(1000))

    # already doing mc
    PLTS.setCurrent({'scenarios': ['scenario1'], 'methods': ['autotune'], 'runs': list(range(1000)), 'iters': iters})

    # states plots
    num_states = trajopt_obj.problem.index_map.n['state']
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle('States')
    axs = PLTS.createGrid(fig, grid=makeGridSpecs(num_states))
    PLTS.dumpLegend(lgnd)

    state_groups = trajopt_obj.problem.config['model']['plotting'].get('state_traj_groups', range(trajopt_obj.problem.index_map.n['state']))

    for j, group in enumerate(state_groups):
        ax = axs[j]
        if show_iters:
            PLTS.addPlot2D(ax, pen=PENS['itr_nl'],  ins={'label': 'Iterations',   'x': 't_nl',  'y': ['z_nl', group],  'iters': iters[1:], 'legend': lgnd})
            PLTS.addPlot2D(ax, pen=PENS['itr_opt'], ins={'label': 'Iterations',   'x': 't_opt', 'y': ['z_opt', group], 'iters': iters[1:], 'legend': lgnd})
        PLTS.addPlot2D(ax, pen=PENS['nl'],      ins={'label': 'Propagated',       'x': 't_nl',  'y': ['z_nl', group],  'iters': [-1],      'legend': lgnd})
        PLTS.addPlot2D(ax, pen=PENS['opt'],     ins={'label': 'Optimal Solution', 'x': 't_opt', 'y': ['z_opt', group], 'iters': [-1],      'legend': lgnd})

    # controls plots
    num_controls = trajopt_obj.problem.index_map.n['control']
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


    # trajectory plots
    state_traj_groups = trajopt_obj.problem.config['model']['plotting']['state_traj_groups']
    num_traj_groups = len(state_traj_groups)
    plt_types = [str(len(group)) + "D" for group in state_traj_groups]

    fig_traj = plt.figure(figsize=(20, 10))
    fig_traj.suptitle('Trajectory')
    axs = PLTS.createGrid2(fig_traj, grid=makeGridSpecs(num_traj_groups), ins={'plt_typs': plt_types})
    PLTS.dumpLegend(lgnd)

    for group in state_traj_groups:
        if len(group) == 2:
            # 2d trajectory plots
            params1 = {'label':'Initial guess','x':('z_opt',group[0]),'y':('z_opt',group[1]),'iters':[1],'legend':lgnd,};
            params3 = {'label':'Propogated','x':('z_nl',group[0]),'y':('z_nl',group[1]),'iters':[-1],'legend':lgnd};
            params4 = {'label':'Optimal Solution','x':('z_opt',group[0]),'y':('z_opt',group[1]),'iters':[-1],'legend':lgnd};
            PLTS.addPlot2D(ax,pen=PENS['init'],ins=params1);
            PLTS.addPlot2D(ax,pen=PENS['nl'],ins=params3); 
            PLTS.addPlot2D(ax,pen=PENS['opt'] ,ins=params4);

        else: 
            # 3d trajectory plots
            params1 = {'label':'Initial guess','x':('z_opt',group[0]),'y':('z_opt',group[1]),'z':('z_opt',group[2]),'iters':[1],'legend':lgnd,};
            params3 = {'label':'Propogated','x':('z_nl',group[0]),'y':('z_nl',group[1]),'z':('z_nl',group[2]),'iters':[-1],'legend':lgnd};
            params4 = {'label':'Optimal Solution','x':('z_opt',group[0]),'y':('z_opt',group[1]),'z':('z_opt',group[2]),'iters':[-1],'legend':lgnd};
            PLTS.addPlot3D(ax,pen=PENS['init'],ins=params1);
            PLTS.addPlot3D(ax,pen=PENS['nl'],ins=params3); 
            PLTS.addPlot3D(ax,pen=PENS['opt'] ,ins=params4);
            
    if usequiver: 
        # TODO(carlos): add quiver computations in constraints library
        # q inds is the index of the quiver if it goes to 2d
        # skip works in general
        
        params5 = {'skip':skip,'label':'u quiver','quiver':('u_vec1_opt',qinds),'iters':[-1],'x':('z_opt',sindx),'y':('z_opt',sindy),'z':('z_opt',sindz)};
        params6 = {'skip':skip,'label':'body quiver','quiver':('body_vec1_opt',qinds),'iters':[-1],'x':('z_opt',sindx),'y':('z_opt',sindy),'z':('z_opt',sindz)};
        PLTS.addPlot3D(ax,pen=PENS['u_vec'],ins=params5)
        PLTS.addPlot3D(ax,pen=PENS['body_vec'],ins=params6)

    
    if usequiver: 
        params5 = {'skip':skip,'label':'u quiver','quiver':('u_vec2_opt',qinds),'iters':[-1],'x':('z_opt',sindx),'y':('z_opt',sindy)};
        params6 = {'skip':skip,'label':'body quiver','quiver':('body_vec2_opt',qinds),'iters':[-1],'x':('z_opt',sindx),'y':('z_opt',sindy)};
        PLTS.addPlot2D(ax,pen=PENS['u_vec2'],ins=params5)
        PLTS.addPlot2D(ax,pen=PENS['body_vec2'],ins=params6)


    params2 = {'label':'Iterations','tinds':[None],'y':tag,'iters':itrs,'legend':'legend1','dataloc':'conv_data'};
    PLTS.addPLot2DIter(ax,pen=PENS['basic_pen2'],ins=params2)

    # constraints plots 
    constraint_data = data['scenario1']['autotune']['runs'][0]['iters'][-1]['constraint_data']
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
    
    iters = trajopt_obj.results['autotune']['runs'][0]['iters'][1:]
    n_iters, n_states, n_ctrl = len(iters), trajopt_obj.problem.index_map.n['state'], trajopt_obj.problem.index_map.n['control']  # use unified index_map
    
    t_all = np.concatenate([it['t_nl'] for it in iters])
    t_lim = [t_all.min() * 0.95, t_all.max() * 1.05]

    def make_grid(n):
        nc = int(np.ceil(np.sqrt(n)))
        return int(np.ceil(n / nc)), nc

    def setup_axes(fig, axs, n_x, data_key, ylabel_prefix):
        lines_nl, lines_opt = [], []
        for j in range(n_x):
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


