import numpy as np
import jax 
import jax.numpy as jnp
import trajopt.utils.tools as tools
jax.config.update("jax_enable_x64", True)
import trajopt.core.modules.model.obstacles     as obstacles
from trajopt.analysis.custom_functions_dan import max_q_nonjax, max_Q_nonjax, max_load_nonjax, terminal_cost, compute_altitude
from trajopt.analysis.trajplots import *


import matplotlib



DPENS = {};

# DPENS['z_opt'] = {'frgba':[0,0,0,0.1],'lrgba':[0,0,0,0.1],'lw':2,'ls':'-','msty':'','msz':4};
# standalone 
DPENS['init'] = {'frgba':[.0,.0,.0,.1],'lrgba':[.0,.0,.0,1.],'lw':1,'ls':'--','msty':'' ,'msz':3};
DPENS['nl'] = {'frgba':[.0,.0,.0,.1],'lrgba':[1.,.0,.0,1.],'lw':2,'ls':'-' ,'msty':'' ,'msz':3};
DPENS['opt']  = {'frgba':[.0,.0,.0,.1],'lrgba':[.0,.0,1.,1.],'lw':1,'ls':''  ,'msty':'o','msz':5};

# iteration values
DPENS['itr_opt']  = {'frgba':[.0,.0,.0,.1],'lrgba':[0.7,.0,0.3,.2],'lw':1,'ls':'','msty':'o' ,'msz':3};
DPENS['itr_nl']   = {'frgba':[.0,.0,.0,.1],'lrgba':[0.7,.0,0.3,.4],'lw':1,'ls':'-','msty':'' ,'msz':3};

# final iteration values
DPENS['fitr_opt']  = {'frgba':[.0,.0,.0,.1],'lrgba':[.0,.0,1.,1.],'lw':3,'ls':'','msty':'o' ,'msz':5};
DPENS['fitr_nl']   = {'frgba':[.0,.0,.0,.1],'lrgba':[0.,.0,1.,1.],'lw':3,'ls':'-','msty':'' ,'msz':5};

# convergence
DPENS['opt2']  = {'frgba':[.0,.0,.0,.1],'lrgba':[.0,.0,1.,1.],'lw':1,'ls':'-'  ,'msty':'o','msz':3};
DPENS['fitr_opt2']  = {'frgba':[.0,.0,.0,.1],'lrgba':[.0,.0,1.,1.],'lw':1,'ls':'-'  ,'msty':'o','msz':3};


DPENS['ref']  = {'frgba':[.0,.0,.0,.1],'lrgba':[.0,.0,.0,1.],'lw':1,'ls':'--','msty':'*','msz':3};
DPENS['standard']  = {'frgba':[.0,.0,.0,.1],'lrgba':[.0,.0,1.,1.],'lw':1,'ls':'-','msty':'o','msz':3};
DPENS['standard_nl']  = {'frgba':[.0,.0,.0,.1],'lrgba':[.0,.0,1.,1.],'lw':1,'ls':'-','msty':'','msz':3};
DPENS['standard_opt']  = {'frgba':[.0,.0,.0,.1],'lrgba':[.0,.0,1.,1.],'lw':1,'ls':'','msty':'o','msz':3};
DPENS['standard_opt2']  = {'frgba':[.0,.0,.0,.1],'lrgba':[.0,.0,1.,1.],'lw':1,'ls':'-','msty':'','msz':3};

DPENS['autotune']  = {'frgba':[.0,.0,.0,.1],'lrgba':[1.,.0,1.,1.],'lw':1,'ls':'-','msty':'o','msz':3};
DPENS['autotune_nl']  = {'frgba':[.0,.0,.0,.1],'lrgba':[1.,.0,1.,1.],'lw':1,'ls':'-','msty':'','msz':3};
DPENS['autotune_opt']  = {'frgba':[.0,.0,.0,.1],'lrgba':[1.,.0,1.,1.],'lw':1,'ls':'','msty':'o','msz':3};
DPENS['autotune_opt2']  = {'frgba':[.0,.0,.0,.1],'lrgba':[1.,.0,1.,1.],'lw':1,'ls':'-','msty':'','msz':3};

# TODO(Skye): Tune this line thickness
DPENS['max-value'] = {'frgba':[.0,.0,.0,.1],'lrgba':[0.0,.0,0.,0.7],'lw':2.5,'ls':'-','msty':'','msz':0};


# cmap1 = matplotlib.cm.get_cmap('hsv')
cmap1 = matplotlib.cm.get_cmap('viridis')
cmap2 = matplotlib.cm.get_cmap('plasma')

clen = 20;
COLORVARS = {};
COLORVARS['autotune'] = {};
COLORVARS['standard'] = {};
COLORVARS['autotune']['lrgba'] = {'by':'runs','typ':'mod','values':[cmap1(jj/clen) for jj in range(clen)]}
COLORVARS['standard']['lrgba'] = {'by':'runs','typ':'mod','values':[cmap2(jj/clen) for jj in range(clen)]}


method_labels = {};
method_labels['autotune'] = 'Continuous-time Auto-SCvx'
method_labels['standard'] = 'Discrete-time Auto-SCvx'



def preProcess(PLTS1,problem,cases={}):
    newcases = {'scenarios':['scenario1'],'methods':['standard','autotune'],'runs':list(range(1000)),'iters':list(range(1000))[1:]}
    if len(cases)>0: newcases = {**newcases,**cases}
    PLTS1.setCurrent(newcases)
    tags = ['max_q','max_Q','max_load','terminal_cost','altitude'];
    for tag in tags:
        tag1 = tag + '_opt'; tag2 = tag + '_nl';
        #func_args1 = ['t_opt','z_opt',None,problem];
        func_args1 = ['t_opt','z_opt','nu_opt',problem];
        
        #func_args2 = ['t_nl','z_nl',None,problem];
        func_args2 = ['t_nl','z_nl','nu_nl',problem];
        if tag == 'max_q': func = max_q_nonjax
        if tag == 'max_Q': func = max_Q_nonjax
        if tag == 'max_load': func = max_load_nonjax
        if tag == 'terminal_cost': func = terminal_cost;
        if tag == 'altitude': func = compute_altitude;

        PLTS1.calcField(tag1,func,func_args = func_args1)
        PLTS1.calcField(tag2,func,func_args = func_args2)

def makePlotCtrls(PLTS1,ins={}):

    ### LOADING DATA
    problem = ins['problem'];
    data = ins['data'];
    versions = ins['versions'];
    NEWPENS = ins['PENS'];
    PENS = {**DPENS,**NEWPENS}
    figpaths = ins['figpaths']
    specs = ins['specs'];
    printfigs = True; displayfigs = True; transparentfigs = True;
    if 'printfigs' in ins: printfigs = ins['printfigs'];
    if 'displayfigs' in ins: displayfigs = ins['displayfigs'];
    if 'transparentfigs' in ins: transparentfigs = ins['transparentfigs']

    #########################################
    ######  DEFAULTS FIG INFORMATION ########
    figsize = (9,3);
    grid = {};
    grid[0] = [0.5,0.5,0.9,0.9];
    titles = {}; ylabels = {}; xlabels = {};
    titles[0] = 'Bank Angle vs. Time';
    xlabels[0] = 'Time [s]';
    ylabels[0] = 'Bank Angle, $\sigma$ [deg]';
    uselegend = [0]
    ##########################################
    #### OVERWRITING DEFAULTS...
    if 'figsize' in ins: figsize = ins['figsize'];
    if 'grid' in ins: grid = {**grid,**ins['grid']};
    if 'titles' in ins: titles = {**titles,**ins['titles']};
    if 'xlabels' in ins: xlabels = {**xlabels,**ins['xlabels']};
    if 'ylabels' in ins: ylabels = {**ylabels,**ins['ylabels']};
    if 'uselegend' in ins: uselegend = ins['uselegend'];
    
    titleinfo = {}; xlabelinfo = {}; ylabelinfo = {}; ticksinfo = {}; legendinfo = {};
    if 'titleinfo' in ins: titleinfo = {**titleinfo,**ins['titleinfo']}
    if 'xlabelinfo' in ins: xlabelinfo = {**xlabelinfo,**ins['xlabelinfo']}
    if 'ylabelinfo' in ins: ylabelinfo = {**ylabelinfo,**ins['ylabelinfo']}
    if 'ticksinfo' in ins:  ticksinfo = {**ticksinfo,**ins['ticksinfo']}
    if 'legendinfo' in ins: legendinfo = {**legendinfo,**ins['legendinfo']}

    state_inds = [0];

    for kk,version in enumerate(versions): 
        scenarios = ['scenario1'];
        methods = ['standard','autotune'];
        runs = list(range(1000))[1:];
        itrs = list(range(1000))[1:];
        if 'methods' in specs[version]: methods = specs[version]['methods']
        if 'runs' in specs[version]: runs = specs[version]['runs']
        if 'itrs' in specs[version]: itrs = specs[version]['itrs']
        ############################################
        
        # grid = PLTS1.specGrid(typ='2x2'); 
        fig = plt.figure(figsize=figsize);
        axs = PLTS1.createGrid(fig,grid = grid);
        lgnd = 'Fig5'; PLTS1.dumpLegend(lgnd);

        
        for j,sind in enumerate(state_inds):
            for method in methods:
                aind = sind;
                ax = axs[aind];
                PLTS1.setCurrent({'scenarios':scenarios,'methods':[method],'runs':runs})
                if version in ['standalone']:
                    params1 = {'label':'Initial guess','x':'t_init','y':('nu_init',sind),'iters':[1],'legend':lgnd};
                    # params2 = {'label':'Iterations','x':'t_opt','y':('nu_opt',sind),'iters':itrs};#,'legend':lgnd};
                    # params2b = {'label':'Iterations','x':'t_nl','y':('nu_nl',sind),'iters':itrs,'legend':lgnd};
                    params3 = {'label':'Propogated','x':'t_nl','y':('nu_nl',sind),'iters':[-1],'legend':lgnd};
                    params4 = {'label':'Optimal Solution','x':'t_opt','y':('nu_opt',sind),'iters':[-1],'legend':lgnd};
                    PLTS1.addPlot2D(ax,pen=PENS['init'],ins=params1);
                    # PLTS1.addPlot2D(ax,pen=PENS['itr_opt'] ,ins=params2);
                    # PLTS1.addPlot2D(ax,pen=PENS['itr_nl'] ,ins=params2b);
                    PLTS1.addPlot2D(ax,pen=PENS['nl'],ins=params3); 
                    PLTS1.addPlot2D(ax,pen=PENS['opt'] ,ins=params4); 

                if version in ['sa_iters']:
                    params1 = {'label':'Initial guess','x':'t_init','y':('nu_init',sind),'iters':[1],'legend':lgnd};
                    params2 = {'label':'Iterations','x':'t_opt','y':('nu_opt',sind),'iters':itrs};#,'legend':lgnd};
                    params2b = {'label':'Iterations','x':'t_nl','y':('nu_nl',sind),'iters':itrs,'legend':lgnd};
                    params3 = {'label':'Propogated','x':'t_nl','y':('nu_nl',sind),'iters':[-1],'legend':lgnd};
                    params4 = {'label':'Optimal Solution','x':'t_opt','y':('nu_opt',sind),'iters':[-1],'legend':lgnd};
                    PLTS1.addPlot2D(ax,pen=PENS['init'],ins=params1);
                    PLTS1.addPlot2D(ax,pen=PENS['itr_opt'] ,ins=params2);
                    PLTS1.addPlot2D(ax,pen=PENS['itr_nl'] ,ins=params2b);
                    PLTS1.addPlot2D(ax,pen=PENS['fitr_nl'],ins=params3); 
                    PLTS1.addPlot2D(ax,pen=PENS['fitr_opt'] ,ins=params4); 

                if version in ['methodvar','mvmc']:
                    params4 = {'label':method_labels[method],'x':'t_opt','y':('nu_opt',sind),'iters':[-1]};
                    params5 = {'label':method_labels[method],'x':'t_nl','y':('nu_nl',sind),'iters':[-1],'legend':lgnd};
                    PLTS1.addPlot2D(ax,pen=PENS[method + '_opt'] ,ins=params4); 
                    PLTS1.addPlot2D(ax,pen=PENS[method + '_nl'] ,ins=params4); 

                if version == 'montecarlo':
                    params4 = {'label':method_labels[method],'x':'t_opt','y':('nu_opt',sind),'iters':[-1],'color_vars':COLORVARS[method]};
                    params5 = {'label':method_labels[method],'x':'t_nl','y':('nu_nl',sind),'iters':[-1],'color_vars':COLORVARS[method],'legend':lgnd};
                    PLTS1.addPlot2D(ax,pen=PENS[method + '_opt'] ,ins=params4); 
                    PLTS1.addPlot2D(ax,pen=PENS[method + '_nl'] ,ins=params4); 

            # #### hack for adding max value line... not that hacky anyway
            umin = problem.mission.u_min[0]*(180/np.pi)
            umax = problem.mission.u_max[0]*(180/np.pi)   
            # line_tag = 'Max-Value'
            # maxval = problem.mission.path_limits[tag];
            # if tag == 'max_load': maxval = maxval/problem.mission.planet['g']
            penn = PENS['max-value'];
            lrgba = penn['lrgba']; ls = penn['ls']; lw = penn['lw']
            line_handle = ax.axhline(y=umin, color=lrgba, linestyle=ls, linewidth=lw); # label=line_tag)
            line_handle = ax.axhline(y=umax, color=lrgba, linestyle=ls, linewidth=lw); #, label=line_tag)
            # PLTS1.legends[lgnd][line_tag] = line_handle;

# # control min/max constraints
# u_min: [-50.0, 50]
# u_max: [230.0, 60]

            params = {};
            params['title'] = {'text':titles[j],'fontsize':20,**titleinfo}
            params['xlabel'] = {'label':xlabels[j],'fontsize':16,**xlabelinfo}
            params['ylabel'] = {'label':ylabels[j],'fontsize':16,**ylabelinfo}
            params['ticks'] = {'labelsize':20,'width':2,**ticksinfo};
            PLTS1.setParams(ax,params);
            if j in uselegend: PLTS1.addLegend(ax,lgnd,ins={'fontsize':14,'loc':'best',**legendinfo});


        if printfigs: 
            figadd = '';
            if version in ['standalone','sa_iters']: figadd = '_sa';
            if version == 'sa_iters': figadd = '_sa_iters';
            if version in ['methodvar','mvmc']: figadd = '_mv';
            if version == 'montecarlo': figadd = '_mc';
            if version == 'mvmc': figadd = '_mvmc';
            figname = figpaths[kk] + 'bank' + figadd + '.pdf'; #'bankangle1.pdf'
            plt.savefig(figname,bbox_inches='tight',pad_inches = 0,transparent=transparentfigs);
        if not(displayfigs): plt.clf();            




def makePlotCtrls2(PLTS1,ins={}):
    problem = ins['problem'];
    data = ins['data'];
    versions = ins['versions'];
    NEWPENS = ins['PENS'];
    PENS = {**DPENS,**NEWPENS}
    figpaths = ins['figpaths']
    specs = ins['specs'];
    printfigs = True; displayfigs = True; transparentfigs = True;
    if 'printfigs' in ins: printfigs = ins['printfigs'];
    if 'displayfigs' in ins: displayfigs = ins['displayfigs'];
    if 'transparentfigs' in ins: transparentfigs = ins['transparentfigs']


    #########################################
    ######  DEFAULTS FIG INFORMATION ########
    figsize=(10,3);
    grid = {};
    # grid[0] = [0.05,0.05,0.4,0.9];
    # grid[1] = [0.50,0.05,0.4,0.9];
    grid[0] = [0.05,0.05,0.37,0.9];
    grid[1] = [0.50,0.05,0.37,0.9];
    titles = {}; ylabels = {}; xlabels = {};
    titles[0] = 'Bank Angle vs. Time';
    titles[1] = 'Angle-of-attack vs. Time';
    ylabels[0] = 'Bank Angle, $\sigma$ [deg]';
    ylabels[1] = 'Angle-of-attack $\\alpha$ [deg]';
    xlabels[0] = 'Time [s]';
    xlabels[1] = 'Time [s]'
    uselegend = [1]

    ##########################################
    if 'figsize' in ins: figsize = ins['figsize'];
    if 'grid' in ins: grid = {**grid,**ins['grid']};
    if 'titles' in ins: titles = {**titles,**ins['titles']};
    if 'xlabels' in ins: xlabels = {**xlabels,**ins['xlabels']};
    if 'ylabels' in ins: ylabels = {**ylabels,**ins['ylabels']};
    if 'uselegend' in ins: uselegend = ins['uselegend'];
    
    titleinfo = {}; xlabelinfo = {}; ylabelinfo = {}; ticksinfo = {}; legendinfo = {};
    if 'titleinfo' in ins: titleinfo = {**titleinfo,**ins['titleinfo']}
    if 'xlabelinfo' in ins: xlabelinfo = {**xlabelinfo,**ins['xlabelinfo']}
    if 'ylabelinfo' in ins: ylabelinfo = {**ylabelinfo,**ins['ylabelinfo']}
    if 'ticksinfo' in ins:  ticksinfo = {**ticksinfo,**ins['ticksinfo']}
    if 'legendinfo' in ins: legendinfo = {**legendinfo,**ins['legendinfo']}

    for kk,version in enumerate(versions): 
        scenarios = ['scenario1'];
        methods = ['standard','autotune'];
        runs = itrs_all = list(range(1000))[1:];
        itrs = list(range(1000))[1:];
        if 'methods' in specs[version]: methods = specs[version]['methods']
        if 'runs' in specs[version]: runs = specs[version]['runs']
        if 'itrs' in specs[version]: itrs = specs[version]['itrs']
        ############################################



        # grid = PLTS1.specGrid(typ='2x2'); 
        fig = plt.figure(figsize=figsize);
        axs = PLTS1.createGrid(fig,grid = grid);

        sinds = [0,1];


        lgnd = 'Fig12'; PLTS1.dumpLegend(lgnd)
        for method in methods:
            PLTS1.setCurrent({'scenarios':scenarios,'methods':[method],'runs':runs})
            for j in sinds:
                ax = axs[j];

                if version in ['standalone']:
                    params1 = {'label':'Initial guess','x':'t_init','y':('nu_init',j),'iters':[1],'legend':lgnd};
                    # params2 = {'label':'Iterations','x':'t_opt','y':('nu_opt',j),'iters':itrs}; #,'legend':lgnd};
                    # params2b = {'label':'Iterations','x':'t_nl','y':('nu_nl',j),'iters':itrs,'legend':lgnd};
                    params3 = {'label':'Propogated','x':'t_nl','y':('nu_nl',j),'iters':[-1],'legend':lgnd};
                    params4 = {'label':'Optimal Solution','x':'t_opt','y':('nu_opt',j),'iters':[-1],'legend':lgnd};

                    PLTS1.addPlot2D(ax,pen=PENS['init'],ins=params1); 
                    # PLTS1.addPlot2D(ax,pen=PENS['itr_opt'] ,ins=params2); 
                    # PLTS1.addPlot2D(ax,pen=PENS['itr_nl'] ,ins=params2b); 
                    PLTS1.addPlot2D(ax,pen=PENS['nl'],ins=params3); 
                    PLTS1.addPlot2D(ax,pen=PENS['opt'] ,ins=params4); 

                if version in ['sa_iters']:
                    params1 = {'label':'Initial guess','x':'t_init','y':('nu_init',j),'iters':[1],'legend':lgnd};
                    params2 = {'label':'Iterations','x':'t_opt','y':('nu_opt',j),'iters':itrs}; #,'legend':lgnd};
                    params2b = {'label':'Iterations','x':'t_nl','y':('nu_nl',j),'iters':itrs,'legend':lgnd};
                    params3 = {'label':'Propogated','x':'t_nl','y':('nu_nl',j),'iters':[-1],'legend':lgnd};
                    params4 = {'label':'Optimal Solution','x':'t_opt','y':('nu_opt',j),'iters':[-1],'legend':lgnd};

                    PLTS1.addPlot2D(ax,pen=PENS['init'],ins=params1); 
                    PLTS1.addPlot2D(ax,pen=PENS['itr_opt'] ,ins=params2); 
                    PLTS1.addPlot2D(ax,pen=PENS['itr_nl'] ,ins=params2b); 
                    PLTS1.addPlot2D(ax,pen=PENS['fitr_nl'],ins=params3); 
                    PLTS1.addPlot2D(ax,pen=PENS['fitr_opt'] ,ins=params4); 

                
                if version in ['methodvar','mvmc']:
                    params4 = {'label':method_labels[method],'x':'t_opt','y':('nu_opt',j),'iters':[-1]};
                    params5 = {'label':method_labels[method],'x':'t_nl','y':('nu_nl',j),'iters':[-1],'legend':lgnd};
                    PLTS1.addPlot2D(ax,pen=PENS[method + '_opt'] ,ins=params4); 
                    PLTS1.addPlot2D(ax,pen=PENS[method + '_nl'] ,ins=params5); 

                if version == 'montecarlo':
                    print(method)
                    params4 = {'label':method_labels[method],'x':'t_opt','y':('nu_opt',j),'iters':[-1],'color_vars':COLORVARS[method],};
                    params5 = {'label':method_labels[method],'x':'t_nl','y':('nu_nl',j),'iters':[-1],'color_vars':COLORVARS[method],'legend':lgnd};
                    PLTS1.addPlot2D(ax,pen=PENS[method + '_opt'] ,ins=params4); 
                    PLTS1.addPlot2D(ax,pen=PENS[method + '_nl'] ,ins=params5); 






        for j in sinds:
            ax = axs[j]; #state_plot_inds[j]];

            # #### hack for adding max value line... not that hacky anyway
            umin = problem.mission.u_min[j]*(180/np.pi)
            umax = problem.mission.u_max[j]*(180/np.pi)   
            line_handle = ax.axhline(y=umin, xmin = 0, color=[0,0,0,0.7], linestyle='-', linewidth=1); # label=line_tag)
            line_handle = ax.axhline(y=umax, xmin = 0, color=[0,0,0,0.7], linestyle='-', linewidth=1); #, label=line_tag)
            # PLTS1.legends[lgnd][line_tag] = line_handle;



            params = {};
            params['title'] = {'text':titles[j],'fontsize':20,**titleinfo}
            params['xlabel'] = {'label':xlabels[j],'fontsize':16,**xlabelinfo}
            params['ylabel'] = {'label':ylabels[j],'fontsize':16,**ylabelinfo}
            params['ticks'] = {'labelsize':20,'width':2,**ticksinfo};
            PLTS1.setParams(ax,params);
            if j in uselegend: PLTS1.addLegend(ax,lgnd,ins={'fontsize':14,'loc':'best'},**legendinfo);

        if printfigs: 
            figadd = '';
            if version in ['standalone','sa_iters']: figadd = '_sa';
            if version == 'sa_iters': figadd = '_sa_iters';
            if version in ['methodvar','mvmc']: figadd = '_mv';
            if version == 'montecarlo': figadd = '_mc';
            if version == 'mvmc': figadd = '_mvmc';
            figname = figpaths[kk] + 'bank_w_aoa' + figadd + '.pdf'; #'bankangle1.pdf'
            plt.savefig(figname,bbox_inches='tight',pad_inches = 0,transparent=transparentfigs);
        if not(displayfigs): plt.clf();            


def makePlotTrajs(PLTS1,ins={}):
    problem = ins['problem'];
    data = ins['data'];
    versions = ins['versions'];
    NEWPENS = ins['PENS'];
    PENS = {**DPENS,**NEWPENS}
    figpaths = ins['figpaths']
    specs = ins['specs'];
    printfigs = True; displayfigs = True; transparentfigs = True;
    if 'printfigs' in ins: printfigs = ins['printfigs'];
    if 'displayfigs' in ins: displayfigs = ins['displayfigs'];
    if 'transparentfigs' in ins: transparentfigs = ins['transparentfigs']

    show_nfzs = True;
    if 'show_nfzs' in ins: show_nfzs = ins['show_nfzs']

    #########################################
    ######  DEFAULTS FIG INFORMATION ########
    figsize = (10,4);
    grid = {}; grid2D = {}; grid3D = {};
    # grid[0] = [0.05,0.05,0.5,0.9];
    # grid[1] = [0.70,0.05,0.4,0.9];
    grid[0] = [0.05,0.05,0.6,0.9];
    grid[1] = [0.75,0.05,0.25,0.9];
    titles = {}; ylabels = {}; xlabels = {}; zlabels = {};

    titles[0] = 'Position(3D) vs Time';
    xlabels[0] = 'Latitude $\phi$ [deg]';
    ylabels[0] = 'Longitude $\\theta$ [deg]';
    zlabels[0] = 'Altitude $h$ [km]';

    titles[1] = 'Position(2D) vs Time';
    ylabels[1] = 'Latitude $\phi$ [deg]';
    xlabels[1] = 'Longitude $\\theta$ [deg]';
    uselegend = [1]

    ##########################################
    if 'figsize' in ins: figsize = ins['figsize'];
    if 'grid' in ins: grid = {**grid,**ins['grid']};
    if 'titles' in ins: titles = {**titles,**ins['titles']};
    if 'xlabels' in ins: xlabels = {**xlabels,**ins['xlabels']};
    if 'ylabels' in ins: ylabels = {**ylabels,**ins['ylabels']};
    if 'uselegend' in ins: uselegend = ins['uselegend'];

    grid2D[1] = grid[1];
    grid3D[0] = grid[0];

    titleinfo = {}; xlabelinfo = {}; ylabelinfo = {}; zlabelinfo = {};
    ticksinfo = {}; legendinfo = {};
    if 'titleinfo' in ins: titleinfo = {**titleinfo,**ins['titleinfo']}
    if 'xlabelinfo' in ins: xlabelinfo = {**xlabelinfo,**ins['xlabelinfo']}
    if 'ylabelinfo' in ins: ylabelinfo = {**ylabelinfo,**ins['ylabelinfo']}
    if 'zlabelinfo' in ins: zlabelinfo = {**zlabelinfo,**ins['zlabelinfo']}
    if 'ticksinfo' in ins:  ticksinfo = {**ticksinfo,**ins['ticksinfo']}
    if 'legendinfo' in ins: legendinfo = {**legendinfo,**ins['legendinfo']}    

    for kk,version in enumerate(versions): 
        scenarios = ['scenario1'];
        methods = ['standard','autotune'];
        runs = itrs_all = list(range(1000))[1:];
        itrs = list(range(1000))[1:];
        if 'methods' in specs[version]: methods = specs[version]['methods']
        if 'runs' in specs[version]: runs = specs[version]['runs']
        if 'itrs' in specs[version]: itrs = specs[version]['itrs']
        ############################################

        # grid = PLTS1.specGrid(typ='2x2'); 
        fig = plt.figure(figsize=figsize);
        axs1 = PLTS1.createGrid(fig,grid = grid2D);

        # axs2 = PLTS1.createGrid(fig,grid = grid3D,ins={'plt_typ':'3d'});
        axs2 = {0: fig.add_axes(grid3D[0],projection='3d')} # colorbar axis]
        axs = {**axs1,**axs2};

        lgnd = 'Fig6'; PLTS1.dumpLegend(lgnd)
        
        for method in methods: 
            PLTS1.setCurrent({'scenarios':scenarios,'methods':[method],'runs':runs})

            j = 0;
            ax = axs[j];
            sindx = 1; sindy = 2; sindz = 0; 
            if version in ['standalone']:
                params1 = {'label':'Initial guess','x':('z_opt',sindx),'y':('z_opt',sindy),'z':'altitude_opt','iters':[1],'legend':lgnd,};
                # params2 = {'label':'Iterations','x':('z_opt',sindx),'y':('z_opt',sindy),'z':('z_opt',sindz),'iters':itrs}; #,'legend':lgnd};
                # params2b = {'label':'Iterations','x':('z_nl',sindx),'y':('z_nl',sindy),'z':('z_nl',sindz),'iters':itrs,'legend':lgnd};
                params3 = {'label':'Propogated','x':('z_nl',sindx),'y':('z_nl',sindy),'z':'altitude_nl','iters':[-1],'legend':lgnd};
                params4 = {'label':'Optimal Solution','x':('z_opt',sindx),'y':('z_opt',sindy),'z':'altitude_opt','iters':[-1],'legend':lgnd};

                PLTS1.addPlot3D(ax,pen=PENS['init'],ins=params1);
                # PLTS1.addPlot3D(ax,pen=PENS['itr_opt'] ,ins=params2);
                # PLTS1.addPlot3D(ax,pen=PENS['itr_nl'] ,ins=params2b);
                PLTS1.addPlot3D(ax,pen=PENS['nl'],ins=params3); 
                PLTS1.addPlot3D(ax,pen=PENS['opt'] ,ins=params4);  

            if version in ['sa_iters']:
                params1 = {'label':'Initial guess','x':('z_opt',sindx),'y':('z_opt',sindy),'z':'altitude_opt','iters':[1],'legend':lgnd,};
                params2 = {'label':'Iterations','x':('z_opt',sindx),'y':('z_opt',sindy),'z':'altitude_opt','iters':itrs}; #,'legend':lgnd};
                params2b = {'label':'Iterations','x':('z_nl',sindx),'y':('z_nl',sindy),'z':'altitude_nl','iters':itrs,'legend':lgnd};
                params3 = {'label':'Propogated','x':('z_nl',sindx),'y':('z_nl',sindy),'z':'altitude_nl','iters':[-1],'legend':lgnd};
                params4 = {'label':'Optimal Solution','x':('z_opt',sindx),'y':('z_opt',sindy),'z':'altitude_opt','iters':[-1],'legend':lgnd};

                PLTS1.addPlot3D(ax,pen=PENS['init'],ins=params1);
                PLTS1.addPlot3D(ax,pen=PENS['itr_opt'] ,ins=params2);
                PLTS1.addPlot3D(ax,pen=PENS['itr_nl'] ,ins=params2b);
                PLTS1.addPlot3D(ax,pen=PENS['fitr_nl'],ins=params3); 
                PLTS1.addPlot3D(ax,pen=PENS['fitr_opt'] ,ins=params4);                 

            if version in ['methodvar','mvmc']:
                params1 = {'label':method_labels[method],'x':('z_opt',sindx),'y':('z_opt',sindy),'z':'altitude_opt','iters':[-1]};
                params2 = {'label':method_labels[method],'x':('z_nl',sindx),'y':('z_nl',sindy),'z':'altitude_nl','iters':[-1],'legend':lgnd};
                PLTS1.addPlot3D(ax,pen=PENS[method + '_opt'],ins=params1);
                PLTS1.addPlot3D(ax,pen=PENS[method + '_nl'],ins=params2);

            if version == 'montecarlo':
                params1 = {'label':method_labels[method],'x':('z_opt',sindx),'y':('z_opt',sindy),'z':'altitude_opt','iters':[-1],'color_vars':COLORVARS[method]};
                params2 = {'label':method_labels[method],'x':('z_nl',sindx),'y':('z_nl',sindy),'z':'altitude_nl','iters':[-1],'color_vars':COLORVARS[method],'legend':lgnd};
                PLTS1.addPlot3D(ax,pen=PENS[method + '_opt'],ins=params1);
                PLTS1.addPlot3D(ax,pen=PENS[method + '_nl'],ins=params2);



            j = 1;
            ax = axs[j]; #state_plot_inds[j]];
            sindx = 1; sindy = 2;
            if version in ['standalone']:
                params1 = {'label':'Initial guess','x':('z_init',sindx),'y':('z_init',sindy),'iters':[1],'legend':lgnd,};
                # params2 = {'label':'Iterations','x':('z_opt',sindx),'y':('z_opt',sindy),'iters':itrs}; #,'legend':lgnd};
                # params2b = {'label':'Iterations','x':('z_nl',sindx),'y':('z_nl',sindy),'iters':itrs,'legend':lgnd};
                params3 = {'label':'Propogated','x':('z_nl',sindx),'y':('z_nl',sindy),'iters':[-1],'legend':lgnd};
                params4 = {'label':'Optimal Solution','x':('z_opt',sindx),'y':('z_opt',sindy),'iters':[-1],'legend':lgnd};

                PLTS1.addPlot2D(ax,pen=PENS['init'],ins=params1);
                # PLTS1.addPlot2D(ax,pen=PENS['itr_opt'] ,ins=params2);
                # PLTS1.addPlot2D(ax,pen=PENS['itr_nl'] ,ins=params2b);
                PLTS1.addPlot2D(ax,pen=PENS['nl'],ins=params3); 
                PLTS1.addPlot2D(ax,pen=PENS['opt'] ,ins=params4);

            if version in ['sa_iters']:
                params1 = {'label':'Initial guess','x':('z_init',sindx),'y':('z_init',sindy),'iters':[1],'legend':lgnd,};
                params2 = {'label':'Iterations','x':('z_opt',sindx),'y':('z_opt',sindy),'iters':itrs}; #,'legend':lgnd};
                params2b = {'label':'Iterations','x':('z_nl',sindx),'y':('z_nl',sindy),'iters':itrs,'legend':lgnd};
                params3 = {'label':'Propogated','x':('z_nl',sindx),'y':('z_nl',sindy),'iters':[-1],'legend':lgnd};
                params4 = {'label':'Optimal Solution','x':('z_opt',sindx),'y':('z_opt',sindy),'iters':[-1],'legend':lgnd};

                PLTS1.addPlot2D(ax,pen=PENS['init'],ins=params1);
                PLTS1.addPlot2D(ax,pen=PENS['itr_opt'] ,ins=params2);
                PLTS1.addPlot2D(ax,pen=PENS['itr_nl'] ,ins=params2b);
                PLTS1.addPlot2D(ax,pen=PENS['fitr_nl'],ins=params3); 
                PLTS1.addPlot2D(ax,pen=PENS['fitr_opt'] ,ins=params4);                

            if version in ['methodvar','mvmc']:
                params1 = {'label':method_labels[method],'x':('z_opt',sindx),'y':('z_opt',sindy),'iters':[-1]};
                params2 = {'label':method_labels[method],'x':('z_nl',sindx),'y':('z_nl',sindy),'iters':[-1],'legend':lgnd};
                PLTS1.addPlot2D(ax,pen=PENS[method + '_opt'],ins=params1);
                PLTS1.addPlot2D(ax,pen=PENS[method + '_nl'],ins=params2);

            if version == 'montecarlo':
                params1 = {'label':method_labels[method],'x':('z_opt',sindx),'y':('z_opt',sindy),'iters':[-1],'color_vars':COLORVARS[method]};
                params2 = {'label':method_labels[method],'x':('z_nl',sindx),'y':('z_nl',sindy),'iters':[-1],'color_vars':COLORVARS[method],'legend':lgnd};
                PLTS1.addPlot2D(ax,pen=PENS[method + '_opt'],ins=params1);
                PLTS1.addPlot2D(ax,pen=PENS[method + '_nl'],ins=params2);



        # ============================================================
        # Cylindrical Keepout Zones
        # ============================================================
        if show_nfzs:
            ax = axs[0]
            mission = problem.mission; method = problem.method
            n_nfz = mission.n_nfz;
            temp = problem.mission.zi[0] - mission.planet['r']
            height = temp; #/ method.nondim['nd']
            # z_traj_max = np.max(z_opt[:, 2])
            # z_traj_min = np.min(z_opt[:, 2])
            # height = 1e8; #(z_traj_max - z_traj_min) * 1.5  # cylinders taller than trajectory
            # z_bottom = z_traj_min - 0.25 * height
            # np.array([mission['planet']['r']+125e3])
            # z_top = z_bottom + height

            for i in range(n_nfz):
                xc = mission.obs['xc'][i] #/ method.nondim['nd']
                yc = mission.obs['yc'][i] #/ method.nondim['nd']
                rc = mission.obs['rc'][i] #/ method.nondim['nd']

                s = np.linspace(0, 2 * np.pi, 100)
                z = np.linspace(0,height, 50)
                S, Z = np.meshgrid(s, z)
                X = xc + rc * np.cos(S)
                Y = yc + rc * np.sin(S)
                ax.plot_surface(X, Y, Z, color='orange', alpha=0.3, linewidth=0, shade=True)

        if show_nfzs: 
            ax = axs[1];
            # Plot circular obstacle projections
            for i in range(n_nfz):
                xc = mission.obs['xc'][i]
                yc = mission.obs['yc'][i]
                rc = mission.obs['rc'][i]
                circle = plt.Circle((xc, yc), rc, color='orange', alpha=0.3, label="Keepout" if i == 0 else None)
                ax.add_patch(circle)



        
        for j in [0,1]:
            ax = axs[j];
            params = {};
            params['title'] = {'text':titles[j],'fontsize':20,**titleinfo}
            params['xlabel'] = {'label':xlabels[j],'fontsize':16,**xlabelinfo}
            params['ylabel'] = {'label':ylabels[j],'fontsize':16,**ylabelinfo}
            if j==0: params['zlabel'] = {'label':zlabels[j],'fontsize':16,**zlabelinfo}
            params['ticks'] = {'labelsize':20,'width':2,**ticksinfo};
            PLTS1.setParams(ax,params);
            if j in uselegend: PLTS1.addLegend(ax,lgnd,ins={'fontsize':12,'loc':'best',**legendinfo});
        
        if printfigs: 
            figadd = '';
            if version in ['standalone','sa_iters']: figadd = '_sa';
            if version == 'sa_iters': figadd = '_sa_iters';
            if version in ['methodvar','mvmc']: figadd = '_mv';
            if version == 'montecarlo': figadd = '_mc';
            if version == 'mvmc': figadd = '_mvmc';
            figname = figpaths[kk] + 'trajectories' + figadd + '.pdf'; #'bankangle1.pdf'
            plt.savefig(figname,bbox_inches='tight',pad_inches = 0,transparent=transparentfigs);
        if not(displayfigs): plt.clf();            



# makePlot3(PLTS1,ins=plotparams);
def makePlotStates(PLTS1,ins={}):
    problem = ins['problem'];
    data = ins['data'];
    versions = ins['versions'];
    NEWPENS = ins['PENS'];
    PENS = {**DPENS,**NEWPENS}
    figpaths = ins['figpaths']
    specs = ins['specs'];
    printfigs = True; displayfigs = True; transparentfigs = True;
    if 'printfigs' in ins: printfigs = ins['printfigs'];
    if 'displayfigs' in ins: displayfigs = ins['displayfigs'];
    if 'transparentfigs' in ins: transparentfigs = ins['transparentfigs']


    #########################################
    ######  DEFAULTS FIG INFORMATION ########
    figsize = (10,5);
    grid = {};
    
    grid[0] = [0.05,0.6,0.35,0.35];
    grid[1] = [0.51,0.6,0.35,0.35];
    grid[2] = [0.05,0.05,0.35,0.35];
    grid[3] = [0.51,0.05,0.35,0.35];    


    titles = {}; ylabels = {}; xlabels = {ind:'Time [s]' for ind in range(4)};
    titles[2] = 'Flight Path Angle vs Time';
    titles[3] = 'Heading vs Time';
    titles[0] = 'Altitude vs Time';
    titles[1] = 'Velocity vs Time';

    ylabels[2] = 'Flight Path Angle [deg]';
    ylabels[3] = 'Heading $\psi$ [deg]';
    ylabels[0] = 'Altitude [km]';
    ylabels[1] = 'Velocity [$m/s$]';    
    uselegend = [3];

    ##########################################
    if 'figsize' in ins: figsize = ins['figsize'];
    if 'grid' in ins: grid = {**grid,**ins['grid']};
    if 'titles' in ins: titles = {**titles,**ins['titles']};
    if 'xlabels' in ins: xlabels = {**xlabels,**ins['xlabels']};
    if 'ylabels' in ins: ylabels = {**ylabels,**ins['ylabels']};
    if 'uselegend' in ins: uselegend = ins['uselegend'];
    
    titleinfo = {}; xlabelinfo = {}; ylabelinfo = {}; ticksinfo = {}; legendinfo = {};
    if 'titleinfo' in ins: titleinfo = {**titleinfo,**ins['titleinfo']}
    if 'xlabelinfo' in ins: xlabelinfo = {**xlabelinfo,**ins['xlabelinfo']}
    if 'ylabelinfo' in ins: ylabelinfo = {**ylabelinfo,**ins['ylabelinfo']}
    if 'ticksinfo' in ins:  ticksinfo = {**ticksinfo,**ins['ticksinfo']}
    if 'legendinfo' in ins: legendinfo = {**legendinfo,**ins['legendinfo']}

    for kk,version in enumerate(versions): 
        scenarios = ['scenario1'];
        methods = ['standard','autotune'];
        runs = itrs_all = list(range(1000))[1:];
        itrs = list(range(1000))[1:];
        if 'methods' in specs[version]: methods = specs[version]['methods']
        if 'runs' in specs[version]: runs = specs[version]['runs']
        if 'itrs' in specs[version]: itrs = specs[version]['itrs']
        ############################################

        # grid = PLTS1.specGrid(typ='2x2'); 
        fig = plt.figure(figsize=figsize);
        axs = PLTS1.createGrid(fig,grid = grid);

        state_inds = [0,3,4,5] # replace with appropriate state indices
        
        lgnd = 'Fig7'; PLTS1.dumpLegend(lgnd)

        
        for j,sind in enumerate(state_inds):
            ax = axs[j];
            for method in methods: 
                PLTS1.setCurrent({'scenarios':scenarios,'methods':[method],'runs':runs})


                ytag_init = ('z_init',sind)
                ytag_nl = ('z_nl',sind)
                ytag_opt = ('z_opt',sind)
                if sind == 0: 
                    # ytag_init = 'altitude_init'
                    ytag_nl = 'altitude_nl'
                    ytag_opt = 'altitude_opt'


                if version in ['standalone']:
                    #TODO (CARLOS / SKYE): init should be in a better place, currently need ot index into iter k >= 1
                    params1 = {'label':'Initial guess','x':'t_init','y':ytag_nl,'iters':[1],'legend':lgnd,};
                    # params2 = {'label':'Iterations','x':'t_opt','y':('z_opt',sind),'iters':itrs}; #,'legend':lgnd};
                    # params2b = {'label':'Iterations','x':'t_nl','y':('z_nl',sind),'iters':itrs,'legend':lgnd};

                    params3 = {'label':'Propogated','x':'t_nl','y':ytag_nl,'iters':[-1],'legend':lgnd};
                    params4 = {'label':'Optimal Solution','x':'t_opt','y':ytag_opt,'iters':[-1],'legend':lgnd};

                    PLTS1.addPlot2D(ax,pen=PENS['init'],ins=params1);
                    # PLTS1.addPlot2D(ax,pen=PENS['itr_opt'] ,ins=params2);
                    # PLTS1.addPlot2D(ax,pen=PENS['itr_nl'] ,ins=params2b);
                    PLTS1.addPlot2D(ax,pen=PENS['nl'],ins=params3); 
                    PLTS1.addPlot2D(ax,pen=PENS['opt'] ,ins=params4); 
                if version in ['sa_iters']:
                    #TODO (CARLOS / SKYE): init should be in a better place, currently need ot index into iter k >= 1
                    params1 = {'label':'Initial guess','x':'t_init','y':ytag_nl,'iters':[1],'legend':lgnd,};
                    params2 = {'label':'Iterations','x':'t_opt','y':ytag_opt,'iters':itrs}; #,'legend':lgnd};
                    params2b = {'label':'Iterations','x':'t_nl','y':ytag_nl,'iters':itrs,'legend':lgnd};
                    params3 = {'label':'Propogated','x':'t_nl','y':ytag_nl,'iters':[-1],'legend':lgnd};
                    params4 = {'label':'Optimal Solution','x':'t_opt','y':ytag_opt,'iters':[-1],'legend':lgnd};

                    PLTS1.addPlot2D(ax,pen=PENS['init'],ins=params1);
                    PLTS1.addPlot2D(ax,pen=PENS['itr_opt'] ,ins=params2);
                    PLTS1.addPlot2D(ax,pen=PENS['itr_nl'] ,ins=params2b);
                    PLTS1.addPlot2D(ax,pen=PENS['fitr_nl'],ins=params3); 
                    PLTS1.addPlot2D(ax,pen=PENS['fitr_opt'] ,ins=params4);                 

                if version in ['methodvar','mvmc']:
                    params4 = {'label':method_labels[method],'x':'t_opt','y':ytag_opt,'iters':[-1]};
                    params5 = {'label':method_labels[method],'x':'t_nl','y':ytag_nl,'iters':[-1],'legend':lgnd};
                    PLTS1.addPlot2D(ax,pen=PENS[method + '_opt'] ,ins=params4); 
                    PLTS1.addPlot2D(ax,pen=PENS[method + '_nl'] ,ins=params5); 
        
                if version == 'montecarlo':
                    params4 = {'label':method_labels[method],'x':'t_opt','y':ytag_opt,'iters':[-1],'color_vars':COLORVARS[method]};
                    params5 = {'label':method_labels[method],'x':'t_nl','y':ytag_nl,'iters':[-1],'color_vars':COLORVARS[method],'legend':lgnd};
                    PLTS1.addPlot2D(ax,pen=PENS[method + '_opt'] ,ins=params4); 
                    PLTS1.addPlot2D(ax,pen=PENS[method + '_nl'] ,ins=params5); 
        

            params = {};
            params['title'] = {'text':titles[j],'fontsize':20,**titleinfo}
            params['xlabel'] = {'label':'Time [s]','fontsize':16,**xlabelinfo}
            params['ylabel'] = {'label':ylabels[j],'fontsize':16,**ylabelinfo}
            params['ticks'] = {'labelsize':20,'width':2,**ticksinfo};
            PLTS1.setParams(ax,params);
            if j in uselegend: PLTS1.addLegend(ax,lgnd,ins={'fontsize':12,'loc':'best',**legendinfo});


        if printfigs: 
            figadd = '';
            if version in ['standalone','sa_iters']: figadd = '_sa';
            if version == 'sa_iters': figadd = '_sa_iters';
            if version in ['methodvar','mvmc']: figadd = '_mv';
            if version == 'montecarlo': figadd = '_mc';
            if version == 'mvmc': figadd = '_mvmc';
            figname = figpaths[kk] + 'states' + figadd + '.pdf'; #'bankangle1.pdf'
            plt.savefig(figname,bbox_inches='tight',pad_inches = 0,transparent=transparentfigs);
        if not(displayfigs): plt.clf();            



def makePlotLoads(PLTS1,ins={}):
    problem = ins['problem'];
    data = ins['data'];
    versions = ins['versions'];
    NEWPENS = ins['PENS'];
    PENS = {**DPENS,**NEWPENS}
    figpaths = ins['figpaths']
    specs = ins['specs'];

    printfigs = True; displayfigs = True; transparentfigs = True;
    if 'printfigs' in ins: printfigs = ins['printfigs'];
    if 'displayfigs' in ins: displayfigs = ins['displayfigs'];
    if 'transparentfigs' in ins: transparentfigs = ins['transparentfigs']

    #########################################
    ######  DEFAULTS FIG INFORMATION ########
    figsize = (10,3)
    grid = {};
    grid[0] = [0.05,0.05,0.25,0.9];
    grid[1] = [0.42,0.05,0.25,0.9];
    grid[2] = [0.75,0.05,0.25,0.9];
    tags = ['max_Q','max_q','max_load']
        # state_names = {0:'Altitude',1:'Velocity',2:'Flight Path Angle',3:'Heading'};
    titles = {}; ylabels = {}; xlabels = {ind:'Time [s]' for ind in range(3)};
    titles[0] = 'Heat Rate Constraint';
    titles[1] = 'Dynamic Pressure Constraint';
    titles[2] = 'Normal Load Constraint';
    ylabels[0] = 'Heating Rate [kW/$m^2$]';
    ylabels[1] = 'Dynamic Pressure [kPa]';
    ylabels[2] = 'Normal Load [$g\'s$]';
    uselegend = [2];


    ##########################################
    if 'figsize' in ins: figsize = ins['figsize'];
    if 'grid' in ins: grid = {**grid,**ins['grid']};
    if 'titles' in ins: titles = {**titles,**ins['titles']};
    if 'xlabels' in ins: xlabels = {**xlabels,**ins['xlabels']};
    if 'ylabels' in ins: ylabels = {**ylabels,**ins['ylabels']};
    if 'uselegend' in ins: uselegend = ins['uselegend'];
    
    titleinfo = {}; xlabelinfo = {}; ylabelinfo = {}; ticksinfo = {}; legendinfo = {};
    if 'titleinfo' in ins: titleinfo = {**titleinfo,**ins['titleinfo']}
    if 'xlabelinfo' in ins: xlabelinfo = {**xlabelinfo,**ins['xlabelinfo']}
    if 'ylabelinfo' in ins: ylabelinfo = {**ylabelinfo,**ins['ylabelinfo']}
    if 'ticksinfo' in ins:  ticksinfo = {**ticksinfo,**ins['ticksinfo']}
    if 'legendinfo' in ins: legendinfo = {**legendinfo,**ins['legendinfo']}

    for kk,version in enumerate(versions): 
        scenarios = ['scenario1'];
        methods = ['standard','autotune'];
        runs = itrs_all = list(range(1000))[1:];
        itrs = list(range(1000))[1:];
        if 'methods' in specs[version]: methods = specs[version]['methods']
        if 'runs' in specs[version]: runs = specs[version]['runs']
        if 'itrs' in specs[version]: itrs = specs[version]['itrs']
        

        # grid = PLTS1.specGrid(typ='2x2'); 
        fig = plt.figure(figsize=figsize);
        axs = PLTS1.createGrid(fig,grid = grid);

        lgnd = 'Fig8'; PLTS1.dumpLegend(lgnd);

        for j,tag in enumerate(tags):            
            for method in methods: 
                ax = axs[j];
                PLTS1.setCurrent({'scenarios':scenarios,'methods':[method],'runs':runs})

                if version in ['standalone']:
                    params1 = {'label':'Initial guess','x':'t_opt','y':tag + '_opt','iters':[1],'legend':lgnd,};
                    # params2 = {'label':'Iterations','x':'t_opt','y':tag + '_opt','iters':itrs}; #,'legend':lgnd};
                    # params2b = {'label':'Iterations','x':'t_nl','y':tag + '_nl','iters':itrs,'legend':lgnd};
                    params3 = {'label':'Propogated','x':'t_nl','y':tag + '_nl','iters':[-1],'legend':lgnd};
                    params4 = {'label':'Optimal Solution','x':'t_opt','y':tag + '_opt','iters':[-1],'legend':lgnd};

                    PLTS1.addPlot2D(ax,pen=PENS['init'],ins=params1);
                    # PLTS1.addPlot2D(ax,pen=PENS['itr_opt'] ,ins=params2);
                    # PLTS1.addPlot2D(ax,pen=PENS['itr_nl'] ,ins=params2b);
                    PLTS1.addPlot2D(ax,pen=PENS['nl'],ins=params3); 
                    PLTS1.addPlot2D(ax,pen=PENS['opt'] ,ins=params4);

                if version in ['sa_iters']:
                    params1 = {'label':'Initial guess','x':'t_opt','y':tag + '_opt','iters':[1],'legend':lgnd,};
                    params2 = {'label':'Iterations','x':'t_opt','y':tag + '_opt','iters':itrs}; #,'legend':lgnd};
                    params2b = {'label':'Iterations','x':'t_nl','y':tag + '_nl','iters':itrs,'legend':lgnd};
                    params3 = {'label':'Propogated','x':'t_nl','y':tag + '_nl','iters':[-1],'legend':lgnd};
                    params4 = {'label':'Optimal Solution','x':'t_opt','y':tag + '_opt','iters':[-1],'legend':lgnd};

                    PLTS1.addPlot2D(ax,pen=PENS['init'],ins=params1);
                    PLTS1.addPlot2D(ax,pen=PENS['itr_opt'] ,ins=params2);
                    PLTS1.addPlot2D(ax,pen=PENS['itr_nl'] ,ins=params2b);
                    PLTS1.addPlot2D(ax,pen=PENS['fitr_nl'],ins=params3); 
                    PLTS1.addPlot2D(ax,pen=PENS['fitr_opt'] ,ins=params4);

                if version in ['methodvar','mvmc']:
                    # params1 = {'label':'Initial guess','x':'t_nl','y':tag,'iters':[1],'legend':lgnd,};
                    # params2 = {'label':'Iterations','x':'t_nl','y':tag,'iters':itrs_all,'legend':lgnd};
                    params3 = {'label':method_labels[method],'x':'t_opt','y':tag + '_opt','iters':[-1]};#,'legend':lgnd};
                    params4 = {'label':method_labels[method],'x':'t_nl','y':tag + '_nl','iters':[-1],'legend':lgnd};
                    PLTS1.addPlot2D(ax,pen=PENS[method + '_opt'] ,ins=params3); # TODO(Skye/Carlos): change to method pen to have dot and line     
                    PLTS1.addPlot2D(ax,pen=PENS[method + '_nl'] ,ins=params4); 
        
                if version == 'montecarlo':
                    # params1 = {'label':'Initial guess','x':'t_nl','y':tag,'iters':[1],'legend':lgnd,};
                    # params2 = {'label':'Iterations','x':'t_nl','y':tag,'iters':itrs_all,'legend':lgnd};
                    params3 = {'label':method_labels[method],'x':'t_opt','y':tag + '_opt','iters':[-1],'color_vars':COLORVARS[method]};#,'legend':lgnd};
                    params4 = {'label':method_labels[method],'x':'t_nl','y':tag + '_nl','iters':[-1],'color_vars':COLORVARS[method],'legend':lgnd};
                    PLTS1.addPlot2D(ax,pen=PENS[method + '_opt'] ,ins=params3); 
                    PLTS1.addPlot2D(ax,pen=PENS[method + '_nl'] ,ins=params4); 

            #### hack for adding max value line... not that hacky anyway
            line_tag = 'Max-Value'
            maxval = problem.mission.path_limits[tag];

            penn = PENS['max-value'];
            lrgba = penn['lrgba']; ls = penn['ls']; lw = penn['lw']
            if tag == 'max_load': maxval = maxval/problem.mission.planet['g']
            line_handle = ax.axhline(y=maxval, color=lrgba, linestyle=ls, linewidth=lw, label=line_tag)
            PLTS1.legends[lgnd][line_tag] = line_handle;

            
            params = {};
            params['title'] = {'text':titles[j],'fontsize':20,**titleinfo}
            params['xlabel'] = {'label':xlabels[j],'fontsize':16,**xlabelinfo}
            params['ylabel'] = {'label':ylabels[j],'fontsize':16,**ylabelinfo}
            params['ticks'] = {'labelsize':20,'width':2,**ticksinfo};
            PLTS1.setParams(ax,params);
            if j in uselegend: PLTS1.addLegend(ax,lgnd,ins={'fontsize':12,'loc':'upper left',**legendinfo});

        if printfigs: 
            figadd = '';
            if version in ['standalone','sa_iters']: figadd = '_sa';
            if version == 'sa_iters': figadd = '_sa_iters';
            if version in ['methodvar','mvmc']: figadd = '_mv';
            if version == 'montecarlo': figadd = '_mc';
            if version == 'mvmc': figadd = '_mvmc';
            figname = figpaths[kk] + 'loads' + figadd + '.pdf'; #'bankangle1.pdf'
            plt.savefig(figname,bbox_inches='tight',pad_inches = 0,transparent=transparentfigs);
        if not(displayfigs): plt.clf();      


def makePlotWghts(PLTS1,ins={}):
    problem = ins['problem'];
    data = ins['data'];
    versions = ins['versions'];
    NEWPENS = ins['PENS'];
    PENS = {**DPENS,**NEWPENS}
    figpaths = ins['figpaths']
    specs = ins['specs'];

    printfigs = True; displayfigs = True; transparentfigs = True;
    if 'printfigs' in ins: printfigs = ins['printfigs'];
    if 'displayfigs' in ins: displayfigs = ins['displayfigs'];
    if 'transparentfigs' in ins: transparentfigs = ins['transparentfigs']


    #########################################
    ######  DEFAULTS FIG INFORMATION ########
    figsize = (10,4);
    grid = {};
    grid[3] = [0.55,0.05,0.35,0.35];
    grid[2] = [0.55,0.60,0.35,0.35];
    grid[1] = [0.05,0.05,0.35,0.35];
    grid[0] = [0.05,0.60,0.35,0.35];
    state_inds = [0,3,4,5] # replace with appropriate state indices
    titles = {}; ylabels = {}; xlabels = {ind:'Time [s]' for ind in range(4)}
    titles[0] = ''; titles[1] = ''; titles[2] = ''; titles[3] = '';
    ylabels[0] = 'No-fly zone quadratic \n penalty weights';
    ylabels[1] = 'Path constraint quadratic \n penalty weights';
    ylabels[2] = 'No-fly zone linear \n penalty weights';
    ylabels[3] = 'Path constraint linear \n penalty weights';
    uselegend = [3];
    #'W_term','W_dyn']; #,'W_plus','W_minus']
    # 'W_ineq' -> path constraints
    # 'W_term' -> terminal condition
    # 'W_dyn', -> dynamics
    # 'W_plus', 'W_minus', -> the weird quadratic 1-norm 
    # 'dual_ineq', 'dual_term', 'dual_dyn', 'dual_plus', 'dual_minus', <- dual versions
    # weight_info = weights = 
    nfz_inds = problem.indices.constraints.nonlinear_inequality['nfz'];
    pth_inds = winds = problem.indices.constraints.nonlinear_inequality['path'];
    weight_info = [['W_ineq',nfz_inds],['dual_ineq',nfz_inds],['W_ineq',pth_inds],['dual_ineq',pth_inds]];



    ##########################################
    if 'figsize' in ins: figsize = ins['figsize'];
    if 'grid' in ins: grid = {**grid,**ins['grid']};
    if 'titles' in ins: titles = {**titles,**ins['titles']};
    if 'xlabels' in ins: xlabels = {**xlabels,**ins['xlabels']};
    if 'ylabels' in ins: ylabels = {**ylabels,**ins['ylabels']};
    if 'uselegend' in ins: uselegend = ins['uselegend'];
    
    titleinfo = {}; xlabelinfo = {}; ylabelinfo = {}; ticksinfo = {}; legendinfo = {};
    if 'titleinfo' in ins: titleinfo = {**titleinfo,**ins['titleinfo']}
    if 'xlabelinfo' in ins: xlabelinfo = {**xlabelinfo,**ins['xlabelinfo']}
    if 'ylabelinfo' in ins: ylabelinfo = {**ylabelinfo,**ins['ylabelinfo']}
    if 'ticksinfo' in ins:  ticksinfo = {**ticksinfo,**ins['ticksinfo']}
    if 'legendinfo' in ins: legendinfo = {**legendinfo,**ins['legendinfo']}

    for kk,version in enumerate(versions): 
        scenarios = ['scenario1'];
        methods = ['standard','autotune'];
        runs = itrs_all = list(range(1000))[1:];
        itrs = list(range(1000))[1:];
        if 'methods' in specs[version]: methods = specs[version]['methods']
        if 'runs' in specs[version]: runs = specs[version]['runs']
        if 'itrs' in specs[version]: itrs = specs[version]['itrs']
        
        # grid = PLTS1.specGrid(typ='2x2'); 
        fig = plt.figure(figsize=figsize);
        axs = PLTS1.createGrid(fig,grid = grid);

        lgnd = 'Fig9'; PLTS1.dumpLegend(lgnd)

        for j,info in enumerate(weight_info):
            ax = axs[j];
            for method in methods: 
                weight = info[0]; winds = info[1]
                
                PLTS1.setCurrent({'scenarios':scenarios,'methods':[method],'runs':runs})

                if version in ['standalone']: 
                    params1 = {'label':'Initial guess','x':'t_opt','y':(weight,winds),'iters':[1],'legend':lgnd,'dataloc':'weights'};
                    params2 = {'label':'Iterations','x':'t_opt','y':(weight,winds),'iters':itrs,'dataloc':'weights'};
                    # params2 = {'label':'Iterations','x':'t_opt','y':(weight,winds),'iters':itrs,'legend':lgnd,'dataloc':'weights'};
                    # params3 = {'label':'Propogated','x':'t_nl','y':weight,'iters':[-1],'legend':lgnd,'dataloc':'weights'};
                    params4 = {'label':'Optimal Solution','x':'t_opt','y':(weight,winds),'iters':[-1],'legend':lgnd,'dataloc':'weights'};

                    # try: 
                    PLTS1.addPlot2D(ax,pen=PENS['init'],ins=params1);
                    # PLTS1.addPlot2D(ax,pen=PENS['itr_opt'] ,ins=params2);
                    # PLTS1.addPlot2D(ax,pen=PENS['nl'],ins=params3); 
                    PLTS1.addPlot2D(ax,pen=PENS['opt2'] ,ins=params4);

                if version in ['sa_iters']: 
                    params1 = {'label':'Initial guess','x':'t_opt','y':(weight,winds),'iters':[1],'legend':lgnd,'dataloc':'weights'};
                    params2 = {'label':'Iterations','x':'t_opt','y':(weight,winds),'iters':itrs,'dataloc':'weights'};
                    # params2 = {'label':'Iterations','x':'t_opt','y':(weight,winds),'iters':itrs,'legend':lgnd,'dataloc':'weights'};
                    # params3 = {'label':'Propogated','x':'t_nl','y':weight,'iters':[-1],'legend':lgnd,'dataloc':'weights'};
                    params4 = {'label':'Optimal Solution','x':'t_opt','y':(weight,winds),'iters':[-1],'legend':lgnd,'dataloc':'weights'};

                    # try: 
                    PLTS1.addPlot2D(ax,pen=PENS['init'],ins=params1);
                    PLTS1.addPlot2D(ax,pen=PENS['itr_opt'] ,ins=params2);
                    # PLTS1.addPlot2D(ax,pen=PENS['nl'],ins=params3); 
                    PLTS1.addPlot2D(ax,pen=PENS['fitr_opt2'] ,ins=params4);

                    # except: pass
                if version in ['methodvar','mvmc']:
                    params4 = {'label':method_labels[method],'x':'t_opt','y':(weight,winds),'iters':[-1],'legend':lgnd,'dataloc':'weights'};
                    PLTS1.addPlot2D(ax,pen=PENS[method + '_opt2'] ,ins=params4);

                if version == 'montecarlo':
                    params4 = {'label':method_labels[method],'x':'t_opt','y':(weight,winds),'iters':[-1],'legend':lgnd,'color_vars':COLORVARS[method],'dataloc':'weights'};
                    PLTS1.addPlot2D(ax,pen=PENS[method + '_opt2'] ,ins=params4);

            



            params = {};
            params['title'] = {'text':titles[j],'fontsize':20,**titleinfo}
            params['xlabel'] = {'label':xlabels[j],'fontsize':16,**xlabelinfo}
            params['ylabel'] = {'label':ylabels[j],'fontsize':16,**ylabelinfo}
            params['ticks'] = {'labelsize':20,'width':2,**ticksinfo};
            PLTS1.setParams(ax,params);
            if j in uselegend: PLTS1.addLegend(ax,lgnd,ins={'fontsize':14,'loc':'best',**legendinfo});

        if printfigs: 
            figadd = '';
            if version in ['standalone','sa_iters']: figadd = '_sa';
            if version == 'sa_iters': figadd = '_sa_iters';
            if version in ['methodvar','mvmc']: figadd = '_mv';
            if version == 'montecarlo': figadd = '_mc';
            if version == 'mvmc': figadd = '_mvmc';
            figname = figpaths[kk] + 'weights' + figadd + '.pdf'; #'bankangle1.pdf'
            plt.savefig(figname,bbox_inches='tight',pad_inches = 0,transparent=transparentfigs);
        if not(displayfigs): plt.clf();            


# makePlot6(PLTS1,ins=plotparams);
def makePlotWghts2(PLTS1,ins={}):
    problem = ins['problem'];
    data = ins['data'];
    versions = ins['versions'];
    NEWPENS = ins['PENS'];
    PENS = {**DPENS,**NEWPENS}
    figpaths = ins['figpaths']
    specs = ins['specs'];
    printfigs = True; displayfigs = True; transparentfigs = True;
    if 'printfigs' in ins: printfigs = ins['printfigs'];
    if 'displayfigs' in ins: displayfigs = ins['displayfigs'];
    if 'transparentfigs' in ins: transparentfigs = ins['transparentfigs']


    #########################################
    ######  DEFAULTS FIG INFORMATION ########
    figsize = (10,4);
    grid = {};

    grid[0] = [0.05,0.05,0.4,0.9];    
    grid[1] = [0.6,0.05,0.4,0.9];

    titles = {}; ylabels = {};
    titles[0] = ''; titles[1] = ''; titles[2] = ''; titles[3] = '';
    xlabels = {ind:'Time [s]' for ind in range(4)}
    ylabels[0] = 'W_plus and W_minus ctcs';
    ylabels[1] = 'dual_plus and dual_minus ctcs';
    # ylabels[2] = 'No-fly zone linear \n penalty weights';
    # ylabels[3] = 'Path constraint linear \n penalty weights';
    uselegend = [1]

    #'W_term','W_dyn']; #,'W_plus','W_minus']
    # 'W_ineq' -> path constraints
    # 'W_term' -> terminal condition
    # 'W_dyn', -> dynamics
    # 'W_plus', 'W_minus', -> the weird quadratic 1-norm 
    # 'dual_ineq', 'dual_term', 'dual_dyn', 'dual_plus', 'dual_minus', <- dual versions

    # weight_info = weights = 
    nfz_inds = problem.indices.constraints.nonlinear_inequality['nfz'];
    pth_inds = problem.indices.constraints.nonlinear_inequality['path'];

    # weight_info = [['W_ineq',nfz_inds],
    #             ['dual_ineq',nfz_inds],
    #             ['W_ineq',pth_inds],
    #             ['dual_ineq',pth_inds]];

    # Figure 1:
    # flag_autotune:      "3"       # '0', '1', '2
    # buff_dyn:           "term"    # 'term', 'l1', 'l2', 'quad-1', 'quad-2'
    # buff_dyn_dual:      "none"    # 'l1', 'none'
    # ctcs:               'quad-2'  # 0, 1
    # ctcs_dual:          "l1"   # 'l1', 'none'
    weight_info = ['W_plus_ctcs','W_minus_ctcs','dual_plus_ctcs','dual_minus_ctcs']
    # weight_info = [left: 'W_plus_ctcs','W_minus_ctcs',
    #                right: ];

    # W_dyn[:,ctcs_idx], dual_dyn[:,ctcs_idx]
    
    #,'W_plus_real','W_minus_real']
    # weight_info = ['W_plus_real','W_minus_real']
    # weight_info = ['W_plus_ctcs','W_minus_ctcs']    

    ##########################################
    if 'figsize' in ins: figsize = ins['figsize'];
    if 'grid' in ins: grid = {**grid,**ins['grid']};
    if 'titles' in ins: titles = {**titles,**ins['titles']};
    if 'xlabels' in ins: xlabels = {**xlabels,**ins['xlabels']};
    if 'ylabels' in ins: ylabels = {**ylabels,**ins['ylabels']};
    if 'uselegend' in ins: uselegend = ins['uselegend'];
    
    titleinfo = {}; xlabelinfo = {}; ylabelinfo = {}; ticksinfo = {}; legendinfo = {};
    if 'titleinfo' in ins: titleinfo = {**titleinfo,**ins['titleinfo']}
    if 'xlabelinfo' in ins: xlabelinfo = {**xlabelinfo,**ins['xlabelinfo']}
    if 'ylabelinfo' in ins: ylabelinfo = {**ylabelinfo,**ins['ylabelinfo']}
    if 'ticksinfo' in ins:  ticksinfo = {**ticksinfo,**ins['ticksinfo']}
    if 'legendinfo' in ins: legendinfo = {**legendinfo,**ins['legendinfo']}

    for kk,version in enumerate(versions): 
        scenarios = ['scenario1'];
        methods = ['standard','autotune'];
        runs = itrs_all = list(range(1000))[1:];
        itrs = list(range(1000))[1:];
        if 'methods' in specs[version]: methods = specs[version]['methods']
        if 'runs' in specs[version]: runs = specs[version]['runs']
        if 'itrs' in specs[version]: itrs = specs[version]['itrs']
        ############################################

        # grid = PLTS1.specGrid(typ='2x2'); 
        fig = plt.figure(figsize=figsize);
        axs = PLTS1.createGrid(fig,grid = grid);

        lgnd = 'Fig9b'; PLTS1.dumpLegend(lgnd)

        for jj,info in enumerate(weight_info):

            weight = info; #[0]; winds = info[1]
            if weight in ['W_plus_ctcs','W_minus_ctcs']: j = 0; 
            if weight in ['dual_plus_ctcs','dual_minus_ctcs']: j = 1;
            ax = axs[j]


            for method in methods: 
                
                
                PLTS1.setCurrent({'scenarios':scenarios,'methods':[method],'runs':runs})

                t_opt_len = problem.method.N; t_nl_len = int(t_opt_len * 20);
                # t_nl_len = problem.method.Ndense
                ttag = ['t_opt',list(range(t_opt_len))]; #[:-1]];
                if version in ['standalone']: 
                    # params1 = {'label':'Initial guess','x':'t_opt','y':(weight,winds),'iters':[1],'legend':lgnd,'dataloc':'weights'};
                    params1 = {'label':'Initial guess','x':ttag,'y':weight,'iters':[1],'legend':lgnd,'dataloc':'weights'};
                    params2 = {'label':'Iterations','x':ttag,'y':weight,'iters':itrs,'legend':lgnd,'dataloc':'weights'};
                    params4 = {'label':'Optimal Solution','x':ttag,'y':weight,'iters':[-1],'legend':lgnd,'dataloc':'weights'};
                    PLTS1.addPlot2D(ax,pen=PENS['init'],ins=params1);
                    # PLTS1.addPlot2D(ax,pen=PENS['itr_opt'] ,ins=params2);
                    PLTS1.addPlot2D(ax,pen=PENS['opt2'] ,ins=params4);

                if version in ['sa_iters']: 
                    # params1 = {'label':'Initial guess','x':'t_opt','y':(weight,winds),'iters':[1],'legend':lgnd,'dataloc':'weights'};
                    params1 = {'label':'Initial guess','x':ttag,'y':weight,'iters':[1],'legend':lgnd,'dataloc':'weights'};
                    params2 = {'label':'Iterations','x':ttag,'y':weight,'iters':itrs,'legend':lgnd,'dataloc':'weights'};
                    params4 = {'label':'Optimal Solution','x':ttag,'y':weight,'iters':[-1],'legend':lgnd,'dataloc':'weights'};
                    PLTS1.addPlot2D(ax,pen=PENS['init'],ins=params1);
                    PLTS1.addPlot2D(ax,pen=PENS['itr_opt'] ,ins=params2);
                    PLTS1.addPlot2D(ax,pen=PENS['fitr_opt2'] ,ins=params4);


                if version in ['methodvar','mvmc']:
                    params4 = {'label':method_labels[method],'x':ttag,'y':weight,'iters':[-1],'legend':lgnd,'dataloc':'weights'};
                    PLTS1.addPlot2D(ax,pen=PENS[method + '_opt'] ,ins=params4);
                if version == 'montecarlo':
                    params4 = {'label':method_labels[method],'x':ttag,'y':weight,'iters':[-1],'color_vars':COLORVARS[method],'legend':lgnd,'dataloc':'weights'};
                    PLTS1.addPlot2D(ax,pen=PENS[method + '_opt'] ,ins=params4);


            # fig.suptitle(figtitles[kk], fontsize=24, fontweight='bold')
            params = {};
            params['title'] = {'text':titles[j],'fontsize':20,**titleinfo}
            params['xlabel'] = {'label':xlabels[j],'fontsize':16,**xlabelinfo}
            params['ylabel'] = {'label':ylabels[j],'fontsize':16,**ylabelinfo}
            params['ticks'] = {'labelsize':20,'width':2,**ticksinfo};
            PLTS1.setParams(ax,params);
            if j in uselegend: PLTS1.addLegend(ax,lgnd,ins={'fontsize':14,'loc':'best',**legendinfo});


        if printfigs: 
            figadd = '';
            if version in ['standalone','sa_iters']: figadd = '_sa';
            if version == 'sa_iters': figadd = '_sa_iters';
            if version in ['methodvar','mvmc']: figadd = '_mv';
            if version == 'montecarlo': figadd = '_mc';
            if version == 'mvmc': figadd = '_mvmc';
            figname = figpaths[kk] + 'weights2' + figadd + '.pdf'; #'bankangle1.pdf'
            plt.savefig(figname,bbox_inches='tight',pad_inches = 0,transparent=transparentfigs);
        if not(displayfigs): plt.clf();            


# makePlot6(PLTS1,ins=plotparams);
def makePlotWghts3(PLTS1,ins={}):
    problem = ins['problem'];
    data = ins['data'];
    versions = ins['versions'];
    NEWPENS = ins['PENS'];
    PENS = {**DPENS,**NEWPENS}
    figpaths = ins['figpaths']
    specs = ins['specs'];
    printfigs = True; displayfigs = True; transparentfigs = True;
    if 'printfigs' in ins: printfigs = ins['printfigs'];
    if 'displayfigs' in ins: displayfigs = ins['displayfigs'];
    if 'transparentfigs' in ins: transparentfigs = ins['transparentfigs']


    #########################################
    ######  DEFAULTS FIG INFORMATION ########
    figsize = (10,4);
    grid = {};
    # grid[3] = [0.55,0.05,0.4,0.3];
    
    # grid[2] = [0.55,0.5,0.4,0.3];
    grid[0] = [0.05,0.05,0.4,0.9];    
    grid[1] = [0.6,0.05,0.4,0.9];
    titles = {}; ylabels = {};
    titles[0] = ''; titles[1] = ''; titles[2] = ''; titles[3] = '';
    xlabels = {ind:'Time [s]' for ind in range(4)}
    ylabels[0] = 'Quadratic weights: \n CTCS dynamics';
    ylabels[1] = 'Dual weights: \n CTCS dynamics';
    # ylabels[2] = 'No-fly zone linear \n penalty weights';
    # ylabels[3] = 'Path constraint linear \n penalty weights';
    uselegend = [1]

    #'W_term','W_dyn']; #,'W_plus','W_minus']
    # 'W_ineq' -> path constraints
    # 'W_term' -> terminal condition
    # 'W_dyn', -> dynamics
    # 'W_plus', 'W_minus', -> the weird quadratic 1-norm 
    # 'dual_ineq', 'dual_term', 'dual_dyn', 'dual_plus', 'dual_minus', <- dual versions

    # weight_info = weights = 
    nfz_inds = problem.indices.constraints.nonlinear_inequality['nfz'];
    pth_inds = problem.indices.constraints.nonlinear_inequality['path'];

    # weight_info = [['W_ineq',nfz_inds],
    #             ['dual_ineq',nfz_inds],
    #             ['W_ineq',pth_inds],
    #             ['dual_ineq',pth_inds]];

    # # Figure 2:
    # flag_autotune:      "3"       # '0', '1', '2
    # buff_dyn:           "term"    # 'term', 'l1', 'l2', 'quad-1', 'quad-2'
    # buff_dyn_dual:      "none"    # 'l1', 'none'
    # ctcs:               'l2'  # 0, 1
    # ctcs_dual:          "none"   # 'l1', 'none'    
    ctcs_idx = problem.indices.z['ctcs']
    weight_info = [('W_dyn',ctcs_idx),('dual_dyn',ctcs_idx)];#'W_minus_ctcs','dual_plus_ctcs','dual_minus_ctcs'];



    # W_dyn[:,ctcs_idx], dual_dyn[:,ctcs_idx]
    
    #,'W_plus_real','W_minus_real']
    # weight_info = ['W_plus_real','W_minus_real']
    # weight_info = ['W_plus_ctcs','W_minus_ctcs']    

    ##########################################
    if 'figsize' in ins: figsize = ins['figsize'];
    if 'grid' in ins: grid = {**grid,**ins['grid']};
    if 'titles' in ins: titles = {**titles,**ins['titles']};
    if 'xlabels' in ins: xlabels = {**xlabels,**ins['xlabels']};
    if 'ylabels' in ins: ylabels = {**ylabels,**ins['ylabels']};
    if 'uselegend' in ins: uselegend = ins['uselegend'];
    
    titleinfo = {}; xlabelinfo = {}; ylabelinfo = {}; ticksinfo = {}; legendinfo = {};
    if 'titleinfo' in ins: titleinfo = {**titleinfo,**ins['titleinfo']}
    if 'xlabelinfo' in ins: xlabelinfo = {**xlabelinfo,**ins['xlabelinfo']}
    if 'ylabelinfo' in ins: ylabelinfo = {**ylabelinfo,**ins['ylabelinfo']}
    if 'ticksinfo' in ins:  ticksinfo = {**ticksinfo,**ins['ticksinfo']}
    if 'legendinfo' in ins: legendinfo = {**legendinfo,**ins['legendinfo']}

    for kk,version in enumerate(versions): 
        scenarios = ['scenario1'];
        methods = ['standard','autotune'];
        runs = itrs_all = list(range(1000))[1:];
        itrs = list(range(1000))[1:];
        if 'methods' in specs[version]: methods = specs[version]['methods']
        if 'runs' in specs[version]: runs = specs[version]['runs']
        if 'itrs' in specs[version]: itrs = specs[version]['itrs']
        ############################################

        # grid = PLTS1.specGrid(typ='2x2'); 
        fig = plt.figure(figsize=figsize);
        axs = PLTS1.createGrid(fig,grid = grid);

        lgnd = 'Fig9b'; PLTS1.dumpLegend(lgnd)

        for j,info in enumerate(weight_info):
            ax = axs[j];
            weight = info[0];
            ctcs_inds = info[1];

            for method in methods: 
                # weight = info; #[0]; winds = info[1]
                
                PLTS1.setCurrent({'scenarios':scenarios,'methods':[method],'runs':runs})

                t_opt_len = problem.method.N; t_nl_len = int(t_opt_len * 20);
                # t_nl_len = problem.method.Ndense
                if version in ['standalone']: 
                    ttag = ['t_opt',list(range(t_opt_len))]; #[:-1]];
                    # if j == 1: ttag = ['t_opt',list(range(t_opt_len))[:-1]]; #[:-1]];
                    # params1 = {'label':'Initial guess','x':'t_opt','y':(weight,winds),'iters':[1],'legend':lgnd,'dataloc':'weights'};
                    params1 = {'label':'Initial guess','x':ttag,'y':(weight,ctcs_inds),'iters':[1],'legend':lgnd,'dataloc':'weights'};
                    params2 = {'label':'Iterations','x':ttag,'y':(weight,ctcs_inds),'iters':itrs,'legend':lgnd,'dataloc':'weights'};
                    params4 = {'label':'Optimal Solution','x':ttag,'y':(weight,ctcs_inds),'iters':[-1],'legend':lgnd,'dataloc':'weights'};

                    PLTS1.addPlot2D(ax,pen=PENS['init'],ins=params1);
                    # PLTS1.addPlot2D(ax,pen=PENS['itr_opt'] ,ins=params2);
                    PLTS1.addPlot2D(ax,pen=PENS['opt2'] ,ins=params4);
                if version in ['sa_iters']: 
                    ttag = ['t_opt',list(range(t_opt_len))]; #[:-1]];
                    # if j == 1: ttag = ['t_opt',list(range(t_opt_len))[:-1]]; #[:-1]];
                    # params1 = {'label':'Initial guess','x':'t_opt','y':(weight,winds),'iters':[1],'legend':lgnd,'dataloc':'weights'};
                    params1 = {'label':'Initial guess','x':ttag,'y':(weight,ctcs_inds),'iters':[1],'legend':lgnd,'dataloc':'weights'};
                    params2 = {'label':'Iterations','x':ttag,'y':(weight,ctcs_inds),'iters':itrs,'legend':lgnd,'dataloc':'weights'};
                    params4 = {'label':'Optimal Solution','x':ttag,'y':(weight,ctcs_inds),'iters':[-1],'legend':lgnd,'dataloc':'weights'};

                    PLTS1.addPlot2D(ax,pen=PENS['init'],ins=params1);
                    PLTS1.addPlot2D(ax,pen=PENS['itr_opt'] ,ins=params2);
                    PLTS1.addPlot2D(ax,pen=PENS['fitr_opt2'] ,ins=params4);

                if version in ['methodvar','mvmc']:
                    # if j == 0: ttag = ['t_opt',list(range(t_opt_len))]; #[:-1]];
                    # if j == 1: 
                    ttag = ['t_opt',list(range(t_opt_len))]; #[:-1]];
                    params4 = {'label':method_labels[method],'x':ttag,'y':(weight,ctcs_inds),'iters':[-1],'legend':lgnd,'dataloc':'weights'};
                    PLTS1.addPlot2D(ax,pen=PENS[method + '_opt2'] ,ins=params4);
                if version == 'montecarlo':
                    ttag = ['t_opt',list(range(t_opt_len))]; #[:-1]];
                    params4 = {'label':method_labels[method],'x':ttag,'y':(weight,ctcs_inds),'iters':[-1],'color_vars':COLORVARS[method],'legend':lgnd,'dataloc':'weights'};
                    PLTS1.addPlot2D(ax,pen=PENS[method + '_opt2'] ,ins=params4);


            # fig.suptitle(figtitles[kk], fontsize=24, fontweight='bold')
            params = {};
            params['title'] = {'text':titles[j],'fontsize':20,**titleinfo}
            params['xlabel'] = {'label':xlabels[j],'fontsize':16,**xlabelinfo}
            params['ylabel'] = {'label':ylabels[j],'fontsize':16,**ylabelinfo}
            params['ticks'] = {'labelsize':20,'width':2,**ticksinfo};
            PLTS1.setParams(ax,params);
            if j in uselegend: PLTS1.addLegend(ax,lgnd,ins={'fontsize':14,'loc':'best',**legendinfo});


        if printfigs: 
            figadd = '';
            if version in ['standalone','sa_iters']: figadd = '_sa';
            if version == 'sa_iters': figadd = '_sa_iters';
            if version in ['methodvar','mvmc']: figadd = '_mv';
            if version == 'montecarlo': figadd = '_mc';
            if version == 'mvmc': figadd = '_mvmc';
            figname = figpaths[kk] + 'weights2' + figadd + '.pdf'; #'bankangle1.pdf'
            plt.savefig(figname,bbox_inches='tight',pad_inches = 0,transparent=transparentfigs);
        if not(displayfigs): plt.clf();            



# makePlot6(PLTS1,ins=plotparams);
def makePlotConvs(PLTS1,ins={}):
    problem = ins['problem'];
    data = ins['data'];
    versions = ins['versions'];
    NEWPENS = ins['PENS'];
    PENS = {**DPENS,**NEWPENS}
    figpaths = ins['figpaths']
    specs = ins['specs'];
    printfigs = True; displayfigs = True; transparentfigs = True;
    if 'printfigs' in ins: printfigs = ins['printfigs'];
    if 'displayfigs' in ins: displayfigs = ins['displayfigs'];
    if 'transparentfigs' in ins: transparentfigs = ins['transparentfigs']

    #########################################
    ######  DEFAULTS FIG INFORMATION ########
    figsize = (10,2);
    grid = {};
    grid[0] = [0.05,0.05,0.4,0.9];
    grid[1] = [0.55,0.05,0.4,0.9];
    tags = ['chk_feas_term','chk_feas_dyn'];
    titles = {}; ylabels = {};
    titles[0] = ''; titles[1] = ''; titles[2] = ''; titles[3] = '';
    ylabels[0] = 'Peak Constraint \n Violation';
    ylabels[1] = 'Peak trajectory \n  residual [km]';    
    xlabels = {ind:'Iterations' for ind in range(2)}
    uselegend = [1];

    ##########################################
    if 'figsize' in ins: figsize = ins['figsize'];
    if 'grid' in ins: grid = {**grid,**ins['grid']};
    if 'titles' in ins: titles = {**titles,**ins['titles']};
    if 'xlabels' in ins: xlabels = {**xlabels,**ins['xlabels']};
    if 'ylabels' in ins: ylabels = {**ylabels,**ins['ylabels']};
    if 'uselegend' in ins: uselegend = ins['uselegend'];
    
    titleinfo = {}; xlabelinfo = {}; ylabelinfo = {}; ticksinfo = {}; legendinfo = {};
    if 'titleinfo' in ins: titleinfo = {**titleinfo,**ins['titleinfo']}
    if 'xlabelinfo' in ins: xlabelinfo = {**xlabelinfo,**ins['xlabelinfo']}
    if 'ylabelinfo' in ins: ylabelinfo = {**ylabelinfo,**ins['ylabelinfo']}
    if 'ticksinfo' in ins:  ticksinfo = {**ticksinfo,**ins['ticksinfo']}
    if 'legendinfo' in ins: legendinfo = {**legendinfo,**ins['legendinfo']}

    for kk,version in enumerate(versions): 
        scenarios = ['scenario1'];
        methods = ['standard','autotune'];
        runs = itrs_all = list(range(1000))[1:];
        itrs = list(range(1000))[1:];
        if 'methods' in specs[version]: methods = specs[version]['methods']
        if 'runs' in specs[version]: runs = specs[version]['runs']
        if 'itrs' in specs[version]: itrs = specs[version]['itrs']
        ###########################################

        # grid = PLTS1.specGrid(typ='2x2'); 
        fig = plt.figure(figsize=figsize);
        axs = PLTS1.createGrid(fig,grid = grid);
        lgnd = 'Fig9b'; PLTS1.dumpLegend(lgnd)

        for j,tag in enumerate(tags):
            ax = axs[j];#state_plot_inds[j]];
            for method in methods:             
                
                PLTS1.setCurrent({'scenarios':scenarios,'methods':[method],'runs':runs})

                PLTS1.setCurrent({'scenarios':scenarios,'methods':[method],'runs':runs})

                if version in ['standalone','sa_iters']:
                    params1 = {'label':'Initial guess','tinds':[None],'y':tag,'iters':[1],'legend':lgnd,'dataloc':'conv_data'};
                    params2 = {'label':'Iterations','tinds':[None],'y':tag,'iters':itrs,'legend':lgnd,'dataloc':'conv_data'};
                    params3 = {'label':'Propogated','tinds':[None],'y':tag,'iters':[-1],'legend':lgnd,'dataloc':'conv_data'};
                    params4 = {'label':'Optimal Solution','tinds':[None],'y':tag,'iters':itrs_all,'legend':lgnd,'dataloc':'conv_data'};
                    PLTS1.addPlot2DIter(ax,pen=PENS['init'] ,ins=params1); 
                    PLTS1.addPlot2DIter(ax,pen=PENS['itr_opt'] ,ins=params2); 
                    PLTS1.addPlot2DIter(ax,pen=PENS['nl'] ,ins=params3); 
                    PLTS1.addPlot2DIter(ax,pen=PENS['opt2'] ,ins=params4); 
                
                if version in ['methodvar','mvmc']:
                    params1 = {'label':method_labels[method],'tinds':[None],'y':tag,'iters':itrs,'legend':lgnd,'dataloc':'conv_data'};
                    PLTS1.addPlot2DIter(ax,pen=PENS[method + '_opt2'] ,ins=params1); 
                                
                if version == 'montecarlo':
                    params1 = {'label':method_labels[method],'tinds':[None],'y':tag,'iters':itrs,'color_vars':COLORVARS[method],'legend':lgnd,'dataloc':'conv_data'};
                    PLTS1.addPlot2DIter(ax,pen=PENS[method + '_opt2'] ,ins=params1); 

                    if False: 
                        # temp = data['scenario1']['autotune']['mc_data'][0]['iters'][2]['weights'];#params']['method']['weights']['W_dyn'];
                        temp = data['scenario1']['autotune']['mc_data'][0]['iters'][2]['conv_data'];#['chk_feas_ineq'];#params']['method']['weights']['W_dyn'];
                        # ['w_cost', 'alpha_z', 'alpha_u', 'beta', 'gamma', 'eps_nonzero1',
                        #  'eps_nonzero2', 'wbuff', 'w_path_scale', 'w_custom_scale', 'w_nfz_scale',
                        lenval = len(problem.method.subprob.iter_data)
                        ydat = [problem.method.subprob.iter_data[ii]['conv_data'][tag] for ii in range(lenval)[1:]]; 
                    #     params1 = {'label':method_labels[method],'tinds':[-1],'y':(tag,sind),'iters':[1],'legend':lgnd,'dataloc':'convergence'};
                #     PLTS1.addPlot2DIter(ax,pen=PENS['opt'] ,ins=params1); 
                # if version == 'methods': pass
                # if version == 'mc': pass


            params = {};
            params['title'] = {'text':titles[j],'fontsize':20,**titleinfo}
            params['xlabel'] = {'label':xlabels[j],'fontsize':16,**xlabelinfo}
            params['ylabel'] = {'label':ylabels[j],'fontsize':16,**ylabelinfo}
            params['ticks'] = {'labelsize':20,'width':2,**ticksinfo};
            PLTS1.setParams(ax,params);
            if j in uselegend: PLTS1.addLegend(ax,lgnd,ins={'fontsize':14,'loc':'best',**legendinfo});

        if printfigs: 
            figadd = '';
            if version in ['standalone','sa_iters']: figadd = '_sa';
            if version == 'sa_iters': figadd = '_sa_iters';
            if version in ['methodvar','mvmc']: figadd = '_mv';
            if version == 'montecarlo': figadd = '_mc';
            if version == 'mvmc': figadd = '_mvmc';
            figname = figpaths[kk] + 'convergence' + figadd + '.pdf'; #'bankangle1.pdf'
            plt.savefig(figname,bbox_inches='tight',pad_inches = 0,transparent=transparentfigs);
        if not(displayfigs): plt.clf();            


# makePlot6(PLTS1,ins=plotparams);
def makePlotConvs2(PLTS1,ins={}):
    problem = ins['problem'];
    data = ins['data'];
    versions = ins['versions'];
    NEWPENS = ins['PENS'];
    PENS = {**DPENS,**NEWPENS}
    figpaths = ins['figpaths']
    specs = ins['specs'];
    printfigs = True; displayfigs = True; transparentfigs = True;
    if 'printfigs' in ins: printfigs = ins['printfigs'];
    if 'displayfigs' in ins: displayfigs = ins['displayfigs'];
    if 'transparentfigs' in ins: transparentfigs = ins['transparentfigs']


    #########################################
    ######  DEFAULTS FIG INFORMATION ########
    figsize = (10,2);
    grid = {};
    grid[0] = [0.05,0.05,0.9,0.9];
    sinds = [3] ;#tags = ['max_q','max_Q','max_load']
    titles = {}; ylabels = {};
    xlabels = {ind:'Iterations' for ind in range(1)}
    titles[0] = 'Cost convergence';
    ylabels[0] = 'Cost \n (Terminal Velocity [m/s])'#Terminal state constraint, \n quadratic penalty weights';    
    uselegend = [1]

    ##########################################
    if 'figsize' in ins: figsize = ins['figsize'];
    if 'grid' in ins: grid = {**grid,**ins['grid']};
    if 'titles' in ins: titles = {**titles,**ins['titles']};
    if 'xlabels' in ins: xlabels = {**xlabels,**ins['xlabels']};
    if 'ylabels' in ins: ylabels = {**ylabels,**ins['ylabels']};
    if 'uselegend' in ins: uselegend = ins['uselegend'];
    
    titleinfo = {}; xlabelinfo = {}; ylabelinfo = {}; ticksinfo = {}; legendinfo = {};
    if 'titleinfo' in ins: titleinfo = {**titleinfo,**ins['titleinfo']}
    if 'xlabelinfo' in ins: xlabelinfo = {**xlabelinfo,**ins['xlabelinfo']}
    if 'ylabelinfo' in ins: ylabelinfo = {**ylabelinfo,**ins['ylabelinfo']}
    if 'ticksinfo' in ins:  ticksinfo = {**ticksinfo,**ins['ticksinfo']}
    if 'legendinfo' in ins: legendinfo = {**legendinfo,**ins['legendinfo']}

    for kk,version in enumerate(versions): 
        scenarios = ['scenario1'];
        methods = ['standard','autotune'];
        runs = itrs_all = list(range(1000))[1:];
        itrs = list(range(1000))[1:];
        if 'methods' in specs[version]: methods = specs[version]['methods']
        if 'runs' in specs[version]: runs = specs[version]['runs']
        if 'itrs' in specs[version]: itrs = specs[version]['itrs']
        ###########################################

        # grid = PLTS1.specGrid(typ='2x2'); 
        fig = plt.figure(figsize=figsize);
        axs = PLTS1.createGrid(fig,grid = grid);

        lgnd = 'Fig11'; PLTS1.dumpLegend(lgnd)

        for j,sind in enumerate(sinds):
            ax = axs[j]; #state_plot_inds[0]];
            for method in methods: 
                
                PLTS1.setCurrent({'scenarios':scenarios,'methods':[method],'runs':runs})

                if version in ['standalone','sa_iters']:
                    params1 = {'label':'Initial guess','tinds':[-1],'y':('z_opt',sind),'iters':[1],'legend':lgnd};
                    params2 = {'label':'Iterations','tinds':[-1],'y':('z_opt',sind),'iters':itrs,'legend':lgnd};
                    params3 = {'label':'Propogated','tinds':[-1],'y':('z_nl',sind),'iters':[-1],'legend':lgnd};
                    params4 = {'label':'Optimal Solution','tinds':[-1],'y':('z_opt',sind),'iters':itrs_all,'legend':lgnd};
                    # PLTS1.addPlot2DIter(ax,pen=PENS['init'] ,ins=params1); 
                    # PLTS1.addPlot2DIter(ax,pen=PENS['itr_opt'] ,ins=params2); 
                    # PLTS1.addPlot2DIter(ax,pen=PENS['nl'] ,ins=params3); 
                    PLTS1.addPlot2DIter(ax,pen=PENS['opt2'] ,ins=params4); 
                
                if version in ['methodvar','mvmc']:
                    params1 = {'label':method_labels[method],'tinds':[-1],'y':('z_opt',sind),'iters':itrs,'legend':lgnd};
                    PLTS1.addPlot2DIter(ax,pen=PENS[method + '_opt2'] ,ins=params1); 
                                
                if version == 'montecarlo':
                    params1 = {'label':method_labels[method],'tinds':[-1],'y':('z_opt',sind),'iters':itrs,'color_vars':COLORVARS[method],'legend':lgnd};
                    PLTS1.addPlot2DIter(ax,pen=PENS[method + '_opt2'] ,ins=params1); 

            params = {};
            params['title'] = {'text':titles[j],'fontsize':20,**titleinfo}
            params['xlabel'] = {'label':xlabels[j],'fontsize':16,**xlabelinfo}
            params['ylabel'] = {'label':ylabels[j],'fontsize':16,**ylabelinfo}
            params['ticks'] = {'labelsize':20,'width':2,**ticksinfo};
            PLTS1.setParams(ax,params);
            if j in uselegend: PLTS1.addLegend(ax,lgnd,ins={'fontsize':14,'loc':'best',**legendinfo});

        if printfigs: 
            figadd = '';
            if version in ['standalone','sa_iters']: figadd = '_sa';
            if version == 'sa_iters': figadd = '_sa_iters';
            if version in ['methodvar','mvmc']: figadd = '_mv';
            if version == 'montecarlo': figadd = '_mc';
            if version == 'mvmc': figadd = '_mvmc';
            figname = figpaths[kk] + 'terminal' + figadd + '.pdf'; #'bankangle1.pdf'
            plt.savefig(figname,bbox_inches='tight',pad_inches = 0,transparent=transparentfigs);
        if not(displayfigs): plt.clf();            



# makePlot6(PLTS1,ins=plotparams);
def makePlotConvs3(PLTS1,ins={}):
    problem = ins['problem'];
    data = ins['data'];
    versions = ins['versions'];
    NEWPENS = ins['PENS'];
    PENS = {**DPENS,**NEWPENS}
    figpaths = ins['figpaths']
    specs = ins['specs'];
    printfigs = True; displayfigs = True; transparentfigs = True;
    if 'printfigs' in ins: printfigs = ins['printfigs'];
    if 'displayfigs' in ins: displayfigs = ins['displayfigs'];
    if 'transparentfigs' in ins: transparentfigs = ins['transparentfigs']


    #########################################
    ######  DEFAULTS FIG INFORMATION ########
    figsize = (10,2);
    grid = {};
    grid[0] = [0.05,0.05,0.9,0.9];
    sinds = [3] ;#tags = ['max_q','max_Q','max_load']
    titles = {}; ylabels = {};
    xlabels = {ind:'Iterations' for ind in range(1)}
    titles[0] = 'Penalty weights \n for constraints';
    ylabels[0] = 'Quadratic weights: \n Terminal State Constraint' #Cost (Terminal Velocity [m/s])'#Terminal state constraint, \n quadratic penalty weights';    
    uselegend = [1]

    ##########################################
    if 'figsize' in ins: figsize = ins['figsize'];
    if 'grid' in ins: grid = {**grid,**ins['grid']};
    if 'titles' in ins: titles = {**titles,**ins['titles']};
    if 'xlabels' in ins: xlabels = {**xlabels,**ins['xlabels']};
    if 'ylabels' in ins: ylabels = {**ylabels,**ins['ylabels']};
    if 'uselegend' in ins: uselegend = ins['uselegend'];
    
    titleinfo = {}; xlabelinfo = {}; ylabelinfo = {}; ticksinfo = {}; legendinfo = {};
    if 'titleinfo' in ins: titleinfo = {**titleinfo,**ins['titleinfo']}
    if 'xlabelinfo' in ins: xlabelinfo = {**xlabelinfo,**ins['xlabelinfo']}
    if 'ylabelinfo' in ins: ylabelinfo = {**ylabelinfo,**ins['ylabelinfo']}
    if 'ticksinfo' in ins:  ticksinfo = {**ticksinfo,**ins['ticksinfo']}
    if 'legendinfo' in ins: legendinfo = {**legendinfo,**ins['legendinfo']}

    for kk,version in enumerate(versions): 
        scenarios = ['scenario1'];
        methods = ['standard','autotune'];
        runs = itrs_all = list(range(1000))[1:];
        itrs = list(range(1000))[1:];
        if 'methods' in specs[version]: methods = specs[version]['methods']
        if 'runs' in specs[version]: runs = specs[version]['runs']
        if 'itrs' in specs[version]: itrs = specs[version]['itrs']
        ###########################################

        # grid = PLTS1.specGrid(typ='2x2'); 
        fig = plt.figure(figsize=figsize);
        axs = PLTS1.createGrid(fig,grid = grid);

        lgnd = 'Fig11'; PLTS1.dumpLegend(lgnd)


        Wtag = 'W_term'
        for j,sind in enumerate(sinds):
            ax = axs[j]; #state_plot_inds[0]];
            for method in methods: 
                
                PLTS1.setCurrent({'scenarios':scenarios,'methods':[method],'runs':runs})

                if version in ['standalone','sa_iters']:
                    params1 = {'label':'Initial guess','tinds':[-1],'y':Wtag,'iters':[1],'legend':lgnd,'dataloc':'weights'};
                    params2 = {'label':'Iterations','tinds':[-1],'y':Wtag,'iters':itrs,'legend':lgnd,'dataloc':'weights'};
                    # params3 = {'label':'Propogated','tinds':[-1],'y':('z_nl',sind),'iters':[-1],'legend':lgnd,'dataloc':'weights'};
                    params4 = {'label':'Optimal Solution','tinds':[-1],'y':Wtag,'iters':itrs_all,'legend':lgnd,'dataloc':'weights'};
                    PLTS1.addPlot2DIter(ax,pen=PENS['init'] ,ins=params1); 
                    PLTS1.addPlot2DIter(ax,pen=PENS['itr_opt'] ,ins=params2); 
                    # PLTS1.addPlot2DIter(ax,pen=PENS['nl'] ,ins=params3); 
                    PLTS1.addPlot2DIter(ax,pen=PENS['opt2'] ,ins=params4); 
                
                if version in ['methodvar','mvmc']:
                    params1 = {'label':method_labels[method],'tinds':[-1],'y':Wtag,'iters':itrs,'legend':lgnd,'dataloc':'weights'};
                    PLTS1.addPlot2DIter(ax,pen=PENS[method + '_opt2'] ,ins=params1); 
                                
                if version == 'montecarlo':
                    params1 = {'label':method_labels[method],'tinds':[-1],'y':Wtag,'iters':itrs,'color_vars':COLORVARS[method],'legend':lgnd,'dataloc':'weights'};
                    PLTS1.addPlot2DIter(ax,pen=PENS[method + '_opt2'] ,ins=params1); 

            params = {};
            params['title'] = {'text':titles[j],'fontsize':20,**titleinfo}
            params['xlabel'] = {'label':xlabels[j],'fontsize':16,**xlabelinfo}
            params['ylabel'] = {'label':ylabels[j],'fontsize':16,**ylabelinfo}
            params['ticks'] = {'labelsize':20,'width':2,**ticksinfo};
            PLTS1.setParams(ax,params);
            if j in uselegend: PLTS1.addLegend(ax,lgnd,ins={'fontsize':14,'loc':'best',**legendinfo});

        if printfigs: 
            figadd = '';
            if version in ['standalone','sa_iters']: figadd = '_sa';
            if version == 'sa_iters': figadd = '_sa_iters';
            if version in ['methodvar','mvmc']: figadd = '_mv';
            if version == 'montecarlo': figadd = '_mc';
            if version == 'mvmc': figadd = '_mvmc';
            figname = figpaths[kk] + 'terminal' + figadd + '.pdf'; #'bankangle1.pdf'
            plt.savefig(figname,bbox_inches='tight',pad_inches = 0,transparent=transparentfigs);
        if not(displayfigs): plt.clf();  