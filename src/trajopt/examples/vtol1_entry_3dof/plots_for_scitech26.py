import numpy as np
import jax 
import jax.numpy as jnp
import trajopt.utils.tools as tools
jax.config.update("jax_enable_x64", True)
import trajopt.core.modules.model.obstacles     as obstacles
from custom_functions_dan import max_q_nonjax, max_Q_nonjax, max_load_nonjax, terminal_cost
from trajopt.analysis.trajplots import *



DPENS = {};

# DPENS['z_opt'] = {'frgba':[0,0,0,0.1],'lrgba':[0,0,0,0.1],'lw':2,'ls':'-','msty':'','msz':4};
DPENS['init'] = {'frgba':[.0,.0,.0,.1],'lrgba':[.0,.0,.0,1.],'lw':1,'ls':'--','msty':'' ,'msz':3};
DPENS['itr']  = {'frgba':[.0,.0,.0,.1],'lrgba':[.0,.0,1.,.2],'lw':1,'ls':'--','msty':'' ,'msz':3};
DPENS['opt']  = {'frgba':[.0,.0,.0,.1],'lrgba':[.0,.0,1.,1.],'lw':1,'ls':''  ,'msty':'o','msz':3};
DPENS['prop'] = {'frgba':[.0,.0,.0,.1],'lrgba':[1.,.0,.0,1.],'lw':1,'ls':'-' ,'msty':'' ,'msz':3};
DPENS['ref']  = {'frgba':[.0,.0,.0,.1],'lrgba':[.0,.0,.0,1.],'lw':1,'ls':'--','msty':'*','msz':3};
DPENS['standard']  = {'frgba':[.0,.0,.0,.1],'lrgba':[.0,.0,1.,1.],'lw':1,'ls':'-','msty':'o','msz':3};
DPENS['autotune']  = {'frgba':[.0,.0,.0,.1],'lrgba':[1.,.0,1.,1.],'lw':1,'ls':'-','msty':'o','msz':3};

DPENS['opt_weight']      = {'frgba':[.0,.0,.0,.1],'lrgba':[.0,.0,1.,1.],'lw':1,'ls':'-'  ,'msty':'o','msz':3};
DPENS['opt_weight_0']    = {'frgba':[.0,.0,.0,.1],'lrgba':[.0,.0,1.,1.],'lw':1,'ls':'-'  ,'msty':'o','msz':3};
DPENS['opt_weight_1']    = {'frgba':[.0,.0,.0,.1],'lrgba':[.2,.0,1.,1.],'lw':1,'ls':'-'  ,'msty':'o','msz':3};
DPENS['opt_weight_2']    = {'frgba':[.0,.0,.0,.1],'lrgba':[.4,.0,1.,1.],'lw':1,'ls':'-'  ,'msty':'o','msz':3};
DPENS['opt_weight_3']    = {'frgba':[.0,.0,.0,.1],'lrgba':[.6,.0,1.,1.],'lw':1,'ls':'-'  ,'msty':'o','msz':3};
DPENS['opt_weight_4']    = {'frgba':[.0,.0,.0,.1],'lrgba':[.8,.0,1.,1.],'lw':1,'ls':'-'  ,'msty':'o','msz':3};
DPENS['opt_weight_5']    = {'frgba':[.0,.0,.0,.1],'lrgba':[1.,.0,1.,1.],'lw':1,'ls':'-'  ,'msty':'o','msz':3};


def preProcess(PLTS1,problem):
    PLTS1.setCurrent({'scenarios':['scenario1'],'methods':['standard','autotune'],
                      'runs':list(range(1000)),'iters':list(range(1000))[1:]})
    tags = ['max_q','max_Q','max_load','terminal_cost'];
    for tag in tags:
        tag1 = tag + '_sub'; tag2 = tag + '_nl';
        func_args1 = ['t_opt','z_opt',None,problem];
        
        func_args2 = ['t_nl','z_nl',None,problem];
        if tag == 'max_q': func = max_q_nonjax
        if tag == 'max_Q': func = max_Q_nonjax
        if tag == 'max_load': func = max_load_nonjax
        if tag == 'terminal_cost': func = terminal_cost;

        PLTS1.calcField(tag1,func,func_args = func_args1)
        PLTS1.calcField(tag2,func,func_args = func_args2)

def makePlotCtrls(PLTS1,ins={}):
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

    figsize = (10,4);
    grid = {};
    grid[0] = [0.5,0.5,0.9,0.9];
    titles = {}; ylabels = {}; xlabels = {};
    titles[0] = 'Bank Angle vs. Time';
    xlabels[0] = 'Time [s]';
    ylabels[0] = 'Bank Angle, $\sigma$ [deg]';
    uselegend = [0]
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

    state_inds = [0];

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
        lgnd = 'Fig5'; PLTS1.dumpLegend(lgnd);

        
        for j,sind in enumerate(state_inds):
            for method in methods:
                aind = sind;
                ax = axs[aind];
                PLTS1.setCurrent({'scenarios':scenarios,'methods':[method],'runs':runs})
                if version == 'standalone':
                    params1 = {'label':'Initial guess','x':'t_init','y':('nu_init',sind),'iters':[1],'legend':lgnd};
                    params2 = {'label':'Iterations','x':'t_opt','y':('nu_opt',sind),'iters':itrs,'legend':lgnd};
                    params3 = {'label':'Propogated','x':'t_nl','y':('nu_nl',sind),'iters':[-1],'legend':lgnd};
                    params4 = {'label':'Optimal Solution','x':'t_opt','y':('nu_opt',sind),'iters':[-1],'legend':lgnd};

                    PLTS1.addPlot2D(ax,pen=PENS['init'],ins=params1);
                    # PLTS1.addPlot2D(ax,pen=PENS['itr'] ,ins=params2);
                    PLTS1.addPlot2D(ax,pen=PENS['prop'],ins=params3); 
                    PLTS1.addPlot2D(ax,pen=PENS['opt'] ,ins=params4); 
                if version == 'methodvar':
                    params4 = {'label':method,'x':'t_opt','y':('nu_opt',sind),'iters':[-1],'legend':lgnd};
                    PLTS1.addPlot2D(ax,pen=PENS[method] ,ins=params4); 

                if version == 'montecarlo':
                    params4 = {'label':method,'x':'t_opt','y':('nu_opt',sind),'iters':[-1],'legend':lgnd};
                    PLTS1.addPlot2D(ax,pen=PENS[method] ,ins=params4); 

            params = {};
            params['title'] = {'text':titles[j],'fontsize':20,**titleinfo}
            params['xlabel'] = {'label':xlabels[j],'fontsize':16,**xlabelinfo}
            params['ylabel'] = {'label':ylabels[j],'fontsize':16,**ylabelinfo}
            params['ticks'] = {'labelsize':20,'width':2,**ticksinfo};
            PLTS1.setParams(ax,params);
            if j in uselegend: PLTS1.addLegend(ax,lgnd,ins={'fontsize':14,'loc':'best',**legendinfo});


        if printfigs: 
            figadd = '';
            if version == 'standalone': figadd = '_sa';
            if version == 'methodvar': figadd = '_mv';
            if version == 'montecarlo': figadd = '_mc';
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


    figsize=(10,4);
    grid = {};
    grid[0] = [0.05,0.05,0.4,0.9];
    grid[1] = [0.50,0.05,0.4,0.9];
    titles = {}; ylabels = {}; xlabels = {};
    titles[0] = 'Bank Angle vs. Time';
    titles[1] = 'Angle-of-attack vs. Time';
    ylabels[0] = 'Bank Angle, $\sigma$ [deg]';
    ylabels[1] = 'Angle-of-attack $\\alpha$ [deg]';
    xlabels[0] = 'Time [s]';
    xlabels[1] = 'Time [s]'
    uselegend = [0,1]

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
            j = 0;
            ax = axs[j];
            params = {'label':method,'x':'t_opt','y':('nu_opt',0),'iters':[-1],'legend':lgnd};
            if j in uselegend: PLTS1.addPlot2D(ax,pen=PENS[method] ,ins=params); 
            # ax = axs[(0,1)];
            # params = {'label':method,'x':'t_opt','y':('nu_opt',1),'iters':[-1],'legend':lgnd};
            # PLTS1.addPlot2D(ax,pen=PENS[method] ,ins=params); 

        for j in sinds:
            ax = axs[j]; #state_plot_inds[j]];
            params = {};
            params['title'] = {'text':titles[j],'fontsize':20,**titleinfo}
            params['xlabel'] = {'label':xlabels[j],'fontsize':16,**xlabelinfo}
            params['ylabel'] = {'label':ylabels[j],'fontsize':16,**ylabelinfo}
            params['ticks'] = {'labelsize':20,'width':2,**ticksinfo};
            PLTS1.setParams(ax,params);
            if j in uselegend: PLTS1.addLegend(ax,lgnd,ins={'fontsize':14,'loc':'best'},**legendinfo);

        if printfigs: 
            figadd = '';
            if version == 'standalone': figadd = '_sa';
            if version == 'methodvar': figadd = '_mv';
            if version == 'montecarlo': figadd = '_mc';
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


    figsize = (10,4);
    grid = {}; grid2D = {}; grid3D = {};
    grid[0] = [0.05,0.05,0.5,0.9];
    grid[1] = [0.70,0.05,0.4,0.9];
    titles = {}; ylabels = {}; xlabels = {}; zlabels = {};

    titles[0] = 'Position(3D) vs Time';
    xlabels[0] = 'Latitude $\phi$ [deg]';
    ylabels[0] = 'Longitude $\\theta$ [deg]';
    zlabels[0] = 'Altitude $h$ [km]';

    titles[1] = 'Position(2D) vs Time';
    ylabels[1] = 'Latitude $\phi$ [deg]';
    xlabels[1] = 'Longitude $\\theta$ [deg]';
    uselegend = [0,1]

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
            if version == 'standalone':
                params1 = {'label':'Initial guess','x':('z_opt',sindx),'y':('z_opt',sindy),'z':('z_opt',sindz),'iters':[1],'legend':lgnd,};
                params2 = {'label':'iterations','x':('z_opt',sindx),'y':('z_opt',sindy),'z':('z_opt',sindz),'iters':itrs_all,'legend':lgnd};
                params3 = {'label':'Propogated','x':('z_nl',sindx),'y':('z_nl',sindy),'z':('z_nl',sindz),'iters':[-1],'legend':lgnd};
                params4 = {'label':'Optimal Solution','x':('z_opt',sindx),'y':('z_opt',sindy),'z':('z_opt',sindz),'iters':[-1],'legend':lgnd};

                PLTS1.addPlot3D(ax,pen=PENS['init'],ins=params1);
                PLTS1.addPlot3D(ax,pen=PENS['itr'] ,ins=params2);
                PLTS1.addPlot3D(ax,pen=PENS['prop'],ins=params3); 
                PLTS1.addPlot3D(ax,pen=PENS['opt'] ,ins=params4);     

            if version == 'methodvar':
                params1 = {'label':method,'x':('z_opt',sindx),'y':('z_opt',sindy),'z':('z_opt',sindz),'iters':[-1],'legend':lgnd};
                PLTS1.addPlot3D(ax,pen=PENS[method],ins=params1);

            if version == 'montecarlo':
                params1 = {'label':method,'x':('z_opt',sindx),'y':('z_opt',sindy),'z':('z_opt',sindz),'iters':[-1],'legend':lgnd};
                PLTS1.addPlot3D(ax,pen=PENS[method],ins=params1);


            j = 1;
            ax = axs[j]; #state_plot_inds[j]];
            sindx = 1; sindy = 2;
            if version == 'standalone':
                params1 = {'label':'Initial guess','x':('z_init',sindx),'y':('z_init',sindy),'iters':[1],'legend':lgnd,};
                params2 = {'label':'iterations','x':('z_opt',sindx),'y':('z_opt',sindy),'iters':itrs,'legend':lgnd};
                params3 = {'label':'Propogated','x':('z_nl',sindx),'y':('z_nl',sindy),'iters':[-1],'legend':lgnd};
                params4 = {'label':'Optimal Solution','x':('z_opt',sindx),'y':('z_opt',sindy),'iters':[-1],'legend':lgnd};

                PLTS1.addPlot2D(ax,pen=PENS['init'],ins=params1);
                PLTS1.addPlot2D(ax,pen=PENS['itr'] ,ins=params2);
                PLTS1.addPlot2D(ax,pen=PENS['prop'],ins=params3); 
                PLTS1.addPlot2D(ax,pen=PENS['opt'] ,ins=params4);

            if version == 'methodvar':
                params1 = {'label':method,'x':('z_opt',sindx),'y':('z_opt',sindy),'iters':[-1],'legend':lgnd};
                PLTS1.addPlot2D(ax,pen=PENS[method],ins=params1);

            if version == 'montecarlo':
                params1 = {'label':method,'x':('z_opt',sindx),'y':('z_opt',sindy),'iters':[-1],'legend':lgnd};
                PLTS1.addPlot2D(ax,pen=PENS[method],ins=params1);


        
        for j in [0,1]:
            ax = axs[j];
            params = {};
            params['title'] = {'text':titles[j],'fontsize':20,**titleinfo}
            params['xlabel'] = {'label':xlabels[j],'fontsize':16,**xlabelinfo}
            params['ylabel'] = {'label':ylabels[j],'fontsize':16,**ylabelinfo}
            if j==0: params['zlabel'] = {'label':zlabels[j],'fontsize':16,**zlabelinfo}
            params['ticks'] = {'labelsize':20,'width':2,**ticksinfo};
            PLTS1.setParams(ax,params);
            if j in uselegend: PLTS1.addLegend(ax,lgnd,ins={'fontsize':14,'loc':'best',**legendinfo});
        
        if printfigs: 
            figadd = '';
            if version == 'standalone': figadd = '_sa';
            if version == 'methodvar': figadd = '_mv';
            if version == 'montecarlo': figadd = '_mc';
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


    figsize = (10,6);
    grid = {};
    grid[3] = [0.55,0.05,0.4,0.35];
    grid[1] = [0.55,0.6,0.4,0.35];
    grid[2] = [0.05,0.05,0.4,0.35];
    grid[0] = [0.05,0.6,0.4,0.35];    

    titles = {}; ylabels = {}; xlabels = {ind:'Time [s]' for ind in range(4)};
    titles[2] = 'Flight Path Angle vs Time';
    titles[3] = 'Heading vs Time';
    titles[0] = 'Attitude vs Time';
    titles[1] = 'Velocity vs Time';

    ylabels[2] = 'Flight Path Angle [deg]';
    ylabels[3] = 'Heading $\psi$ [deg]';
    ylabels[0] = 'Attitude [km]';
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

                if version == 'standalone':
                    #TODO (CARLOS / SKYE): init should be in a better place, currently need ot index into iter k >= 1
                    params1 = {'label':'Initial guess','x':'t_init','y':('z_init',sind),'iters':[1],'legend':lgnd,};
                    params2 = {'label':'Iterations','x':'t_opt','y':('z_opt',sind),'iters':itrs,'legend':lgnd};
                    params3 = {'label':'Propogated','x':'t_nl','y':('z_nl',sind),'iters':[-1],'legend':lgnd};
                    params4 = {'label':'Optimal Solution','x':'t_opt','y':('z_opt',sind),'iters':[-1],'legend':lgnd};

                    PLTS1.addPlot2D(ax,pen=PENS['init'],ins=params1);
                    PLTS1.addPlot2D(ax,pen=PENS['itr'] ,ins=params2);
                    PLTS1.addPlot2D(ax,pen=PENS['prop'],ins=params3); 
                    PLTS1.addPlot2D(ax,pen=PENS['opt'] ,ins=params4); 
                
                if version == 'methodvar':
                    params4 = {'label':method,'x':'t_opt','y':('z_opt',sind),'iters':[-1],'legend':lgnd};
                    PLTS1.addPlot2D(ax,pen=PENS[method] ,ins=params4); 
        
                if version == 'montecarlo':
                    params4 = {'label':method,'x':'t_opt','y':('z_opt',sind),'iters':[-1],'legend':lgnd};
                    PLTS1.addPlot2D(ax,pen=PENS[method] ,ins=params4); 
        

            params = {};
            params['title'] = {'text':titles[j],'fontsize':20,**titleinfo}
            params['xlabel'] = {'label':'Time [s]','fontsize':16,**xlabelinfo}
            params['ylabel'] = {'label':ylabels[j],'fontsize':16,**ylabelinfo}
            params['ticks'] = {'labelsize':20,'width':2,**ticksinfo};
            PLTS1.setParams(ax,params);
            if j in uselegend: PLTS1.addLegend(ax,lgnd,ins={'fontsize':14,'loc':'best',**legendinfo});


        if printfigs: 
            figadd = '';
            if version == 'standalone': figadd = '_sa';
            if version == 'methodvar': figadd = '_mv';
            if version == 'montecarlo': figadd = '_mc';
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

    figsize = (10,4)
    grid = {};
    grid[0] = [0.05,0.05,0.25,0.9];
    grid[1] = [0.4,0.05,0.25,0.9];
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
        

        # grid = PLTS1.specGrid(typ='2x2'); 
        fig = plt.figure(figsize=figsize);
        axs = PLTS1.createGrid(fig,grid = grid);

        lgnd = 'Fig8'; PLTS1.dumpLegend(lgnd);

        for j,tag in enumerate(tags):            
            for method in methods: 
                ax = axs[j];
                PLTS1.setCurrent({'scenarios':scenarios,'methods':[method],'runs':runs})

                if version == 'standalone':
                    params1 = {'label':'Initial guess','x':'t_opt','y':tag + '_sub','iters':[1],'legend':lgnd,};
                    params2 = {'label':'Iterations','x':'t_opt','y':tag + '_sub','iters':itrs,'legend':lgnd};
                    params3 = {'label':'Propogated','x':'t_nl','y':tag + '_nl','iters':[-1],'legend':lgnd};
                    params4 = {'label':'Optimal Solution','x':'t_opt','y':tag + '_sub','iters':[-1],'legend':lgnd};

                    PLTS1.addPlot2D(ax,pen=PENS['init'],ins=params1);
                    PLTS1.addPlot2D(ax,pen=PENS['itr'] ,ins=params2);
                    PLTS1.addPlot2D(ax,pen=PENS['prop'],ins=params3); 
                    PLTS1.addPlot2D(ax,pen=PENS['opt'] ,ins=params4);
                if version == 'methodvar':
                    # params1 = {'label':'Initial guess','x':'t_nl','y':tag,'iters':[1],'legend':lgnd,};
                    # params2 = {'label':'iterations','x':'t_nl','y':tag,'iters':itrs_all,'legend':lgnd};
                    params3 = {'label':'Propogated','x':'t_nl','y':tag + '_nl','iters':[-1],'legend':lgnd};
                    params4 = {'label':'Optimal Solution','x':'t_opt','y':tag + '_sub','iters':[-1],'legend':lgnd};
                    # PLTS1.addPlot2D(ax,pen=PENS[method] ,ins=params3);         
                    PLTS1.addPlot2D(ax,pen=PENS[method] ,ins=params4); 
        
                if version == 'montecarlo':
                    # params1 = {'label':'Initial guess','x':'t_nl','y':tag,'iters':[1],'legend':lgnd,};
                    # params2 = {'label':'iterations','x':'t_nl','y':tag,'iters':itrs_all,'legend':lgnd};
                    params3 = {'label':'Propogated','x':'t_nl','y':tag + '_nl','iters':[-1],'legend':lgnd};
                    params4 = {'label':'Optimal Solution','x':'t_opt','y':tag + '_sub','iters':[-1],'legend':lgnd};
                    # PLTS1.addPlot2D(ax,pen=PENS[method] ,ins=params3);         
                    PLTS1.addPlot2D(ax,pen=PENS[method] ,ins=params4); 

            #### hack for adding max value line... not that hacky anyway
            line_tag = 'Max-Value'
            maxval = problem.mission.path_limits[tag];
            if tag == 'max_load': maxval = maxval/problem.mission.planet['g']
            line_handle = ax.axhline(y=maxval, color=[0,0,0,0.7], linestyle='--', linewidth=2, label=line_tag)
            PLTS1.legends[lgnd][line_tag] = line_handle;

            
        params = {};
        params['title'] = {'text':titles[j],'fontsize':20,**titleinfo}
        params['xlabel'] = {'label':xlabels[j],'fontsize':16,**xlabelinfo}
        params['ylabel'] = {'label':ylabels[j],'fontsize':16,**ylabelinfo}
        params['ticks'] = {'labelsize':20,'width':2,**ticksinfo};
        PLTS1.setParams(ax,params);
        if j in uselegend: PLTS1.addLegend(ax,lgnd,ins={'fontsize':14,'loc':'best',**legendinfo});

        if printfigs: 
            figadd = '';
            if version == 'standalone': figadd = '_sa';
            if version == 'methodvar': figadd = '_mv';
            if version == 'montecarlo': figadd = '_mc';
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

    figsize = (10,4);
    grid = {};
    grid[3] = [0.55,0.05,0.4,0.35];
    grid[2] = [0.55,0.6,0.4,0.35];
    grid[1] = [0.05,0.05,0.4,0.35];
    grid[0] = [0.05,0.6,0.4,0.35];

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

                if version == 'standalone': 
                    params1 = {'label':'Initial guess','x':'t_opt','y':(weight,winds),'iters':[1],'legend':lgnd,'dataloc':'weights'};
                    params2 = {'label':'iterations','x':'t_opt','y':(weight,winds),'iters':itrs,'legend':lgnd,'dataloc':'weights'};
                    # params3 = {'label':'Propogated','x':'t_nl','y':weight,'iters':[-1],'legend':lgnd,'dataloc':'weights'};
                    params4 = {'label':'Optimal Solution','x':'t_opt','y':(weight,winds),'iters':[-1],'legend':lgnd,'dataloc':'weights'};

                    # try: 
                    PLTS1.addPlot2D(ax,pen=PENS['init'],ins=params1);
                    PLTS1.addPlot2D(ax,pen=PENS['itr'] ,ins=params2);
                    # PLTS1.addPlot2D(ax,pen=PENS['prop'],ins=params3); 
                    PLTS1.addPlot2D(ax,pen=PENS['opt'] ,ins=params4);
                    # except: pass
                if version == 'methodvar':
                    params4 = {'label':method,'x':'t_opt','y':(weight,winds),'iters':[-1],'legend':lgnd,'dataloc':'weights'};
                    PLTS1.addPlot2D(ax,pen=PENS[method] ,ins=params4);

                if version == 'montecarlo':
                    params4 = {'label':method,'x':'t_opt','y':(weight,winds),'iters':[-1],'legend':lgnd,'dataloc':'weights'};
                    PLTS1.addPlot2D(ax,pen=PENS[method] ,ins=params4);

                
            params = {};
            params['title'] = {'text':titles[j],'fontsize':20,**titleinfo}
            params['xlabel'] = {'label':xlabels[j],'fontsize':16,**xlabelinfo}
            params['ylabel'] = {'label':ylabels[j],'fontsize':16,**ylabelinfo}
            params['ticks'] = {'labelsize':20,'width':2,**ticksinfo};
            PLTS1.setParams(ax,params);
            if j in uselegend: PLTS1.addLegend(ax,lgnd,ins={'fontsize':14,'loc':'best',**legendinfo});

        if printfigs: 
            figadd = '';
            if version == 'standalone': figadd = '_sa';
            if version == 'methodvar': figadd = '_mv';
            if version == 'montecarlo': figadd = '_mc';
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

    figsize = (10,4);
    grid = {};
    grid[3] = [0.55,0.05,0.4,0.3];
    grid[1] = [0.05,0.05,0.4,0.3];
    grid[2] = [0.55,0.5,0.4,0.3];
    grid[0] = [0.05,0.5,0.4,0.3];    
    titles = {}; ylabels = {};
    titles[0] = ''; titles[1] = ''; titles[2] = ''; titles[3] = '';
    xlabels = {ind:'Time [s]' for ind in range(4)}
    ylabels[0] = 'No-fly zone quadratic \n penalty weights';
    ylabels[1] = 'Path constraint quadratic \n penalty weights';
    ylabels[2] = 'No-fly zone linear \n penalty weights';
    ylabels[3] = 'Path constraint linear \n penalty weights';
    uselegend = [1]

    #'W_term','W_dyn']; #,'W_plus','W_minus']
    # 'W_ineq' -> path constraints
    # 'W_term' -> terminal condition
    # 'W_dyn', -> dynamics
    # 'W_plus', 'W_minus', -> the weird quadratic 1-norm 
    # 'dual_ineq', 'dual_term', 'dual_dyn', 'dual_plus', 'dual_minus', <- dual versions

    # weight_info = weights = 
    nfz_inds = problem.indices.constraints.nonlinear_inequality['nfz'];
    pth_inds = winds = problem.indices.constraints.nonlinear_inequality['path'];

    # weight_info = [['W_ineq',nfz_inds],
    #             ['dual_ineq',nfz_inds],
    #             ['W_ineq',pth_inds],
    #             ['dual_ineq',pth_inds]];
    # weight_info = ['W_plus_real','W_minus_real','W_plus_ctcs','W_minus_ctcs']
    weight_info = ['W_plus_ctcs','W_minus_ctcs']    

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
            for method in methods: 
                weight = info; #[0]; winds = info[1]
                
                PLTS1.setCurrent({'scenarios':scenarios,'methods':[method],'runs':runs})

                t_opt_len = problem.method.N; t_nl_len = int(t_opt_len * 20);
                # t_nl_len = problem.method.Ndense
                ttag = ['t_opt',list(range(t_opt_len))[:-1]];
                if version == 'standalone': 
                    # params1 = {'label':'Initial guess','x':'t_opt','y':(weight,winds),'iters':[1],'legend':lgnd,'dataloc':'weights'};
                    params1 = {'label':'Initial guess','x':ttag,'y':weight,'iters':[1],'legend':lgnd,'dataloc':'weights'};
                    params2 = {'label':'iterations','x':ttag,'y':weight,'iters':itrs,'legend':lgnd,'dataloc':'weights'};
                    params4 = {'label':'Optimal Solution','x':ttag,'y':weight,'iters':[-1],'legend':lgnd,'dataloc':'weights'};

                    PLTS1.addPlot2D(ax,pen=PENS['init'],ins=params1);
                    PLTS1.addPlot2D(ax,pen=PENS['itr'] ,ins=params2);
                    PLTS1.addPlot2D(ax,pen=PENS['opt'] ,ins=params4);
                if version == 'methodvar':
                    params4 = {'label':method,'x':ttag,'y':weight,'iters':[-1],'legend':lgnd,'dataloc':'weights'};
                    PLTS1.addPlot2D(ax,pen=PENS[method] ,ins=params4);
                if version == 'montecarlo':
                    params4 = {'label':method,'x':ttag,'y':weight,'iters':[-1],'legend':lgnd,'dataloc':'weights'};
                    PLTS1.addPlot2D(ax,pen=PENS[method] ,ins=params4);


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
            if version == 'standalone': figadd = '_sa';
            if version == 'methodvar': figadd = '_mv';
            if version == 'montecarlo': figadd = '_mc';
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

    figsize = (10,2);
    grid = {};
    grid[0] = [0.05,0.05,0.4,0.9];
    grid[1] = [0.6,0.05,0.4,0.9];
    
    tags = ['chk_feas_term','chk_feas_dyn'];
    titles = {}; ylabels = {};
    titles[0] = ''; titles[1] = ''; titles[2] = ''; titles[3] = '';
    ylabels[0] = 'Peak Constraint Violation';
    ylabels[1] = 'Peak trajectory residual [km]';    
    xlabels = {ind:'Iterations [k]' for ind in range(2)}
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

                if version == 'standalone':
                    lenval = len(problem.method.subprob.iter_data)
                    ydat = [problem.method.subprob.iter_data[ii]['conv_data'][tag] for ii in range(lenval)[1:]]; 

                #     params1 = {'label':method,'tinds':[-1],'y':(tag,sind),'iters':[1],'legend':lgnd,'dataloc':'convergence'};
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
            if version == 'standalone': figadd = '_sa';
            if version == 'methodvar': figadd = '_mv';
            if version == 'montecarlo': figadd = '_mc';
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

    figsize = (10,2);
    grid = {};
    grid[0] = [0.05,0.05,0.9,0.9];
    sinds = [3] ;#tags = ['max_q','max_Q','max_load']
    titles = {}; ylabels = {}; xlabels = {ind:'Iterations [k]' for ind in range(1)}
    titles[0] = 'Penalty weights \n for constraints';
    ylabels[0] = 'Terminal state constraint, \n quadratic penalty weights';    
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

                if version == 'standalone':
                    params1 = {'label':'Initial guess','tinds':[-1],'y':('z_opt',sind),'iters':[1],'legend':lgnd};
                    params2 = {'label':'Iterations','tinds':[-1],'y':('z_opt',sind),'iters':itrs,'legend':lgnd};
                    params3 = {'label':'Propogated','tinds':[-1],'y':('z_nl',sind),'iters':[-1],'legend':lgnd};
                    params4 = {'label':'Optimal Solution','tinds':[-1],'y':('z_opt',sind),'iters':itrs_all,'legend':lgnd};
                    PLTS1.addPlot2DIter(ax,pen=PENS['init'] ,ins=params1); 
                    PLTS1.addPlot2DIter(ax,pen=PENS['itr'] ,ins=params2); 
                    PLTS1.addPlot2DIter(ax,pen=PENS['prop'] ,ins=params3); 
                    PLTS1.addPlot2DIter(ax,pen=PENS['opt'] ,ins=params4); 
                
                if version == 'methodvar':
                    params1 = {'label':method,'tinds':[-1],'y':('z_opt',sind),'iters':itrs,'legend':lgnd};
                    PLTS1.addPlot2DIter(ax,pen=PENS[method] ,ins=params1); 
                                
                if version == 'montecarlo':
                    params1 = {'label':method,'tinds':[-1],'y':('z_opt',sind),'iters':itrs,'legend':lgnd};
                    PLTS1.addPlot2DIter(ax,pen=PENS[method] ,ins=params1); 

            params = {};
            params['title'] = {'text':titles[j],'fontsize':20,**titleinfo}
            params['xlabel'] = {'label':xlabels[j],'fontsize':16,**xlabelinfo}
            params['ylabel'] = {'label':ylabels[j],'fontsize':16,**ylabelinfo}
            params['ticks'] = {'labelsize':20,'width':2,**ticksinfo};
            PLTS1.setParams(ax,params);
            if j in uselegend: PLTS1.addLegend(ax,lgnd,ins={'fontsize':14,'loc':'best',**legendinfo});

        if printfigs: 
            figadd = '';
            if version == 'standalone': figadd = '_sa';
            if version == 'methodvar': figadd = '_mv';
            if version == 'montecarlo': figadd = '_mc';
            figname = figpaths[kk] + 'terminal' + figadd + '.pdf'; #'bankangle1.pdf'
            plt.savefig(figname,bbox_inches='tight',pad_inches = 0,transparent=transparentfigs);
        if not(displayfigs): plt.clf();            
