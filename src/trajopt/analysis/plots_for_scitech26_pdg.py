import numpy as np
import jax 
import jax.numpy as jnp
import trajopt.core.modules.utils.tools as tools
jax.config.update("jax_enable_x64", True)
import trajopt.core.modules.model.obstacles     as obstacles
from trajopt.analysis.custom_functions_dan_pdg import DCM, calc_DCMs, calc_rt_I
from trajopt.analysis.custom_functions_dan_pdg import thrust_mag, compute_tilt, ang_rate, omega_degrees
from trajopt.analysis.custom_functions_dan_pdg import calc_u_vecs_scale1, calc_u_vecs_scale2
from trajopt.analysis.custom_functions_dan_pdg import calc_body_vecs_scale1, calc_body_vecs_scale2

from trajopt.analysis.trajplots import *

matplotlib.rcParams['axes3d.mouserotationstyle'] = 'azel'


import matplotlib


DPENS = {};

# DPENS['z_opt'] = {'frgba':[0,0,0,0.1],'lrgba':[0,0,0,0.1],'lw':2,'ls':'-','msty':'','msz':4};
# standalone 
DPENS['init'] = {'frgba':[.0,.0,.0,.1],'lrgba':[.0,.0,.0,1.],'lw':1,'ls':'--','msty':'' ,'msz':3};
DPENS['nl'] = {'frgba':[.0,.0,.0,.1],'lrgba':[1.,.0,.0,1.],'lw':2,'ls':'-' ,'msty':'' ,'msz':3};
DPENS['opt']  = {'frgba':[.0,.0,.0,.1],'lrgba':[.0,.0,1.,1.],'lw':1,'ls':''  ,'msty':'o','msz':3};


# iteration values
DPENS['itr_opt']  = {'frgba':[.0,.0,.0,.1],'lrgba':[0.7,.0,0.3,.2],'lw':1,'ls':'','msty':'o' ,'msz':3};
DPENS['itr_nl']   = {'frgba':[.0,.0,.0,.1],'lrgba':[0.7,.0,0.3,.4],'lw':1,'ls':'-','msty':'' ,'msz':3};
# final iteration values
DPENS['fitr_opt']  = {'frgba':[.0,.0,.0,.1],'lrgba':[.0,.0,1.,1.],'lw':2,'ls':'','msty':'o' ,'msz':3};
DPENS['fitr_nl']   = {'frgba':[.0,.0,.0,.1],'lrgba':[0.,.0,1.,1.],'lw':2,'ls':'-','msty':'' ,'msz':3};

# # iteration values
# DPENS['itr_opt']  = {'frgba':[.0,.0,.0,.1],'lrgba':[0.,.0,0.3,.2],'lw':1,'ls':'','msty':'o' ,'msz':3};
# DPENS['itr_nl']   = {'frgba':[.0,.0,.0,.1],'lrgba':[0.7,.0,0.3,.4],'lw':1,'ls':'-','msty':'' ,'msz':3};
# # final iteration values
# DPENS['fitr_opt']  = {'frgba':[.0,.0,.0,.1],'lrgba':[.0,.0,1.,1.],'lw':2,'ls':'','msty':'o' ,'msz':5};
# DPENS['fitr_nl']   = {'frgba':[.0,.0,.0,.1],'lrgba':[0.7,.0,0.3,1.],'lw':2,'ls':'-','msty':'' ,'msz':5};

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

# for 3D plot
DPENS['u_vec']  = {'frgba':[.0,.0,.0,.1],'lrgba':[1.,0.25,0.,1.],'lw':2,'ls':'-','msty':'','msz':0};
DPENS['body_vec'] = {'frgba':[.0,.0,.0,.1],'lrgba':[0.,0.,0.,1.],'lw':2,'ls':'-','msty':'','msz':0};
# for 2D plots...
DPENS['u_vec2']  = {'frgba':[.0,.0,.0,.1],'lrgba':[1.,0.25,0.,1.],'lw':4,'ls':'-','msty':'','msz':0};
DPENS['body_vec2'] = {'frgba':[.0,.0,.0,.1],'lrgba':[0.,0.,0.,1.],'lw':4,'ls':'-','msty':'','msz':0};

#             color=(1, 60/255, 0),

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


t_init_tag = 't_init'
nu_init_tag = 'nu_init'
z_init_tag = 'z_init'


def preProcess(PLTS1,trajopt_obj,cases={}):

    newcases = {'scenarios':['scenario1'],'methods':['standard','autotune'],'runs':list(range(1000)),'iters':list(range(1000))[1:]}
    if len(cases)>0: newcases = {**newcases,**cases}
    PLTS1.setCurrent(newcases)

    tags = ['DCM','u_vec1','u_vec2','rt_I',
            'body_vec1','body_vec2','thrust_mag','tilt','ang_rate','omega_deg'];    
    for tag in tags:
        tag1 = tag + '_opt';
        tag2 = tag + '_nl';
        tag3 = tag + '_init';
        func_args1 = ['t_opt','z_opt','nu_opt',trajopt_obj];
        func_args2 = ['t_nl','z_nl','nu_nl',trajopt_obj];
        func_args3 = ['t_init','z_init','nu_init',trajopt_obj];

        if tag == 'DCM': func = calc_DCMs
        if tag == 'u_vec1': func = calc_u_vecs_scale1
        if tag == 'u_vec2': func = calc_u_vecs_scale2
        if tag == 'rt_I': func = calc_rt_I
        if tag == 'body_vec1': func = calc_body_vecs_scale1;
        if tag == 'body_vec2': func = calc_body_vecs_scale2;
        if tag == 'thrust_mag': func = thrust_mag
        if tag == 'tilt': func = compute_tilt
        if tag == 'ang_rate': func = ang_rate
        if tag == 'omega_deg': func = omega_degrees
        
        PLTS1.calcField(tag1,func,func_args = func_args1)
        PLTS1.calcField(tag2,func,func_args = func_args2)
        PLTS1.calcField(tag3,func,func_args = func_args3)



def makePlotTrajs(PLTS1,ins={}):
    trajopt_obj = ins['trajopt_obj'];
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

    show_nfzs = False;
    if 'show_nfzs' in ins: show_nfzs = ins['show_nfzs']

    #########################################
    ######  DEFAULTS FIG INFORMATION ########



    figsize = (9,9);

    sideviews = True; 
    if 'sideviews' in ins: sideviews = ins['sideviews'];
    if sideviews: plot_inds = [0,1,2,3]
    else: plot_inds = [0];



    grid = {};
    temp = 0.30
    grid[1] = [0.65,0.65,temp,temp]
    grid[2] = [0.15,0.65,temp,temp]
    grid[3] = [0.65,0.15,temp,temp]

    if sideviews: grid[0] = [0.05,0.05,0.5,0.5];
    else: grid[0] = [0.15,0.15,0.7,0.7];


    grid2D = {}; grid3D = {};
    # grid[0] = [0.05,0.05,0.5,0.9];
    # grid[1] = [0.70,0.05,0.4,0.9];
    titles = {}; ylabels = {}; xlabels = {}; zlabels = {};

    titles[0] = '';#Position(3D) vs Time';

    xlabels[0] = 'East';
    ylabels[0] = 'North';
    zlabels[0] = 'Up';

    titles[1] = '';#Position(2D) vs Time';
    ylabels[1] = 'North';
    xlabels[1] = 'East';

    titles[2] = '';#Position(2D) vs Time';
    ylabels[2] = 'Up';
    xlabels[2] = 'East';

    titles[3] = '';#Position(2D) vs Time';
    ylabels[3] = 'Up';
    xlabels[3] = 'North';


    uselegend = [0]
    usequiver = True;



    ##########################################
    if 'figsize' in ins: figsize = ins['figsize'];
    if 'grid' in ins: grid = {**grid,**ins['grid']};
    if 'titles' in ins: titles = {**titles,**ins['titles']};
    if 'xlabels' in ins: xlabels = {**xlabels,**ins['xlabels']};
    if 'ylabels' in ins: ylabels = {**ylabels,**ins['ylabels']};
    if 'uselegend' in ins: uselegend = ins['uselegend'];
    if 'usequiver' in ins: usequiver = ins['usequiver'];

    grid3D[0] = grid[0];
    ####################
    grid2D[1] = grid[1];
    grid2D[2] = grid[2];
    grid2D[3] = grid[3];



    titleinfo = {}; xlabelinfo = {}; ylabelinfo = {}; zlabelinfo = {};
    ticksinfo = {}; legendinfo = {};
    if 'titleinfo' in ins: titleinfo = {**titleinfo,**ins['titleinfo']}
    if 'xlabelinfo' in ins: xlabelinfo = {**xlabelinfo,**ins['xlabelinfo']}
    if 'ylabelinfo' in ins: ylabelinfo = {**ylabelinfo,**ins['ylabelinfo']}
    if 'zlabelinfo' in ins: zlabelinfo = {**zlabelinfo,**ins['zlabelinfo']}
    if 'ticksinfo' in ins:  ticksinfo = {**ticksinfo,**ins['ticksinfo']}
    if 'legendinfo' in ins: legendinfo = {**legendinfo,**ins['legendinfo']}    

    skip = 1;
    if 'skip' in ins: skip = ins['skip'];

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

        if sideviews: 
            axs1 = PLTS1.createGrid(fig,grid = grid2D);
            axs2 = {0: fig.add_axes(grid3D[0],projection='3d')} # colorbar axis]
            axs = {**axs1,**axs2};
        else:
            axs = {0: fig.add_axes(grid3D[0],projection='3d')} # colorbar axis]
        lgnd = 'Fig6'; PLTS1.dumpLegend(lgnd)
        
        for method in methods:
            PLTS1.setCurrent({'scenarios':scenarios,'methods':[method],'runs':runs})
            for j in plot_inds:
                ax = axs[j];
                
                if j == 0: sindx = 2; sindy = 3; sindz = 1; qinds = (1,2,0)
                if j == 1: sindx = 2; sindy = 3; qinds = (1,2)
                if j == 2: sindx = 2; sindy = 1; qinds = (1,0)
                if j == 3: sindx = 3; sindy = 1; qinds = (2,0)


                if j == 0:

                    ### ADD CODE FOR 3D CONE...
                    # ax.DRAWCONE 
                    ################


                    if usequiver: 
                        params5 = {'skip':skip,'label':'u quiver','quiver':('u_vec1_opt',qinds),'iters':[-1],'x':('z_opt',sindx),'y':('z_opt',sindy),'z':('z_opt',sindz)};
                        params6 = {'skip':skip,'label':'body quiver','quiver':('body_vec1_opt',qinds),'iters':[-1],'x':('z_opt',sindx),'y':('z_opt',sindy),'z':('z_opt',sindz)};
                        PLTS1.addPlot3D(ax,pen=PENS['u_vec'],ins=params5)
                        PLTS1.addPlot3D(ax,pen=PENS['body_vec'],ins=params6)


                    if version in ['standalone']:
                        params1 = {'label':'Initial guess','x':('z_opt',sindx),'y':('z_opt',sindy),'z':('z_opt',sindz),'iters':[1],'legend':lgnd,};
                        # params2 = {'label':'Iterations','x':('z_opt',sindx),'y':('z_opt',sindy),'z':('z_opt',sindz),'iters':itrs}; #,'legend':lgnd};
                        # params2b = {'label':'Iterations','x':('z_nl',sindx),'y':('z_nl',sindy),'z':('z_nl',sindz),'iters':itrs,'legend':lgnd};
                        params3 = {'label':'Propogated','x':('z_nl',sindx),'y':('z_nl',sindy),'z':('z_nl',sindz),'iters':[-1],'legend':lgnd};
                        params4 = {'label':'Optimal Solution','x':('z_opt',sindx),'y':('z_opt',sindy),'z':('z_opt',sindz),'iters':[-1],'legend':lgnd};
                        PLTS1.addPlot3D(ax,pen=PENS['init'],ins=params1);
                        # PLTS1.addPlot3D(ax,pen=PENS['itr_opt'] ,ins=params2);
                        # PLTS1.addPlot3D(ax,pen=PENS['itr_nl'] ,ins=params2b);
                        PLTS1.addPlot3D(ax,pen=PENS['nl'],ins=params3); 
                        PLTS1.addPlot3D(ax,pen=PENS['opt'] ,ins=params4);


                    if version in ['sa_iters']:
                        params1 = {'label':'Initial guess','x':('z_opt',sindx),'y':('z_opt',sindy),'z':('z_opt',sindz),'iters':[1],'legend':lgnd,};
                        params2 = {'label':'Iterations','x':('z_opt',sindx),'y':('z_opt',sindy),'z':('z_opt',sindz),'iters':itrs}; #,'legend':lgnd};
                        params2b = {'label':'Iterations','x':('z_nl',sindx),'y':('z_nl',sindy),'z':('z_nl',sindz),'iters':itrs,'legend':lgnd};
                        params3 = {'label':'Propogated','x':('z_nl',sindx),'y':('z_nl',sindy),'z':('z_nl',sindz),'iters':[-1],'legend':lgnd};
                        params4 = {'label':'Optimal Solution','x':('z_opt',sindx),'y':('z_opt',sindy),'z':('z_opt',sindz),'iters':[-1],'legend':lgnd};

                        PLTS1.addPlot3D(ax,pen=PENS['init'],ins=params1);
                        PLTS1.addPlot3D(ax,pen=PENS['itr_opt'] ,ins=params2);
                        PLTS1.addPlot3D(ax,pen=PENS['itr_nl'] ,ins=params2b);
                        PLTS1.addPlot3D(ax,pen=PENS['fitr_nl'],ins=params3); 
                        PLTS1.addPlot3D(ax,pen=PENS['fitr_opt'] ,ins=params4);

                    if version in ['methodvar','mvmc']:
                        params1 = {'label':method_labels[method],'x':('z_opt',sindx),'y':('z_opt',sindy),'z':('z_opt',sindz),'iters':[-1]};
                        params2 = {'label':method_labels[method],'x':('z_nl',sindx),'y':('z_nl',sindy),'z':('z_nl',sindz),'iters':[-1],'legend':lgnd};
                        PLTS1.addPlot3D(ax,pen=PENS[method + '_opt'],ins=params1);
                        PLTS1.addPlot3D(ax,pen=PENS[method + '_nl'],ins=params2);

                    if version == 'montecarlo':
                        params1 = {'label':method_labels[method],'x':('z_opt',sindx),'y':('z_opt',sindy),'z':('z_opt',sindz),'iters':[-1],'color_vars':COLORVARS[method]};
                        params2 = {'label':method_labels[method],'x':('z_nl',sindx),'y':('z_nl',sindy),'z':('z_nl',sindz),'iters':[-1],'color_vars':COLORVARS[method],'legend':lgnd};
                        PLTS1.addPlot3D(ax,pen=PENS[method + '_opt'],ins=params1);
                        PLTS1.addPlot3D(ax,pen=PENS[method + '_nl'],ins=params2);


                if j in [1,2,3]:
                    ### ADD CODE FOR 2D CONE...
                    # ax.DRAWCONE 
                    ################


                    if usequiver: 
                        params5 = {'skip':skip,'label':'u quiver','quiver':('u_vec2_opt',qinds),'iters':[-1],'x':('z_opt',sindx),'y':('z_opt',sindy)};
                        params6 = {'skip':skip,'label':'body quiver','quiver':('body_vec2_opt',qinds),'iters':[-1],'x':('z_opt',sindx),'y':('z_opt',sindy)};
                        PLTS1.addPlot2D(ax,pen=PENS['u_vec2'],ins=params5)
                        PLTS1.addPlot2D(ax,pen=PENS['body_vec2'],ins=params6)

                    
                    if version in ['standalone']:
                        params1 = {'label':'Initial guess','x':('z_opt',sindx),'y':('z_opt',sindy),'iters':[1],'legend':lgnd,};
                        params3 = {'label':'Propogated','x':('z_nl',sindx),'y':('z_nl',sindy),'iters':[-1],'legend':lgnd};
                        params4 = {'label':'Optimal Solution','x':('z_opt',sindx),'y':('z_opt',sindy),'iters':[-1],'legend':lgnd};
                        PLTS1.addPlot2D(ax,pen=PENS['init'],ins=params1);
                        PLTS1.addPlot2D(ax,pen=PENS['nl'],ins=params3); 
                        PLTS1.addPlot2D(ax,pen=PENS['opt'] ,ins=params4);


                    if version in ['sa_iters']:
                        params1 = {'label':'Initial guess','x':('z_opt',sindx),'y':('z_opt',sindy),'iters':[1],'legend':lgnd,};
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


            # j = 1;
            # ax = axs[j]; #state_plot_inds[j]];
            # sindx = 1; sindy = 2;
            # if version in ['standalone']:
            #     params1 = {'label':'Initial guess','x':(z_init_tag,sindx),'y':(z_init_tag,sindy),'iters':[1],'legend':lgnd,};
            #     # params2 = {'label':'Iterations','x':('z_opt',sindx),'y':('z_opt',sindy),'iters':itrs}; #,'legend':lgnd};
            #     # params2b = {'label':'Iterations','x':('z_nl',sindx),'y':('z_nl',sindy),'iters':itrs,'legend':lgnd};
            #     params3 = {'label':'Propogated','x':('z_nl',sindx),'y':('z_nl',sindy),'iters':[-1],'legend':lgnd};
            #     params4 = {'label':'Optimal Solution','x':('z_opt',sindx),'y':('z_opt',sindy),'iters':[-1],'legend':lgnd};

            #     PLTS1.addPlot2D(ax,pen=PENS['init'],ins=params1);
            #     # PLTS1.addPlot2D(ax,pen=PENS['itr_opt'] ,ins=params2);
            #     # PLTS1.addPlot2D(ax,pen=PENS['itr_nl'] ,ins=params2b);
            #     PLTS1.addPlot2D(ax,pen=PENS['nl'],ins=params3); 
            #     PLTS1.addPlot2D(ax,pen=PENS['opt'] ,ins=params4);

            # if version in ['sa_iters']:
            #     params1 = {'label':'Initial guess','x':(z_init_tag,sindx),'y':(z_init_tag,sindy),'iters':[1],'legend':lgnd,};
            #     params2 = {'label':'Iterations','x':('z_opt',sindx),'y':('z_opt',sindy),'iters':itrs}; #,'legend':lgnd};
            #     params2b = {'label':'Iterations','x':('z_nl',sindx),'y':('z_nl',sindy),'iters':itrs,'legend':lgnd};
            #     params3 = {'label':'Propogated','x':('z_nl',sindx),'y':('z_nl',sindy),'iters':[-1],'legend':lgnd};
            #     params4 = {'label':'Optimal Solution','x':('z_opt',sindx),'y':('z_opt',sindy),'iters':[-1],'legend':lgnd};

            #     PLTS1.addPlot2D(ax,pen=PENS['init'],ins=params1);
            #     PLTS1.addPlot2D(ax,pen=PENS['itr_opt'] ,ins=params2);
            #     PLTS1.addPlot2D(ax,pen=PENS['itr_nl'] ,ins=params2b);
            #     PLTS1.addPlot2D(ax,pen=PENS['fitr_nl'],ins=params3); 
            #     PLTS1.addPlot2D(ax,pen=PENS['fitr_opt'] ,ins=params4);                

            # if version in ['methodvar','mvmc']:
            #     params1 = {'label':method_labels[method],'x':('z_opt',sindx),'y':('z_opt',sindy),'iters':[-1]};
            #     params2 = {'label':method_labels[method],'x':('z_nl',sindx),'y':('z_nl',sindy),'iters':[-1],'legend':lgnd};
            #     PLTS1.addPlot2D(ax,pen=PENS[method + '_opt'],ins=params1);
            #     PLTS1.addPlot2D(ax,pen=PENS[method + '_nl'],ins=params2);

            # if version == 'montecarlo':
            #     params1 = {'label':method_labels[method],'x':('z_opt',sindx),'y':('z_opt',sindy),'iters':[-1],'color_vars':COLORVARS[method]};
            #     params2 = {'label':method_labels[method],'x':('z_nl',sindx),'y':('z_nl',sindy),'iters':[-1],'color_vars':COLORVARS[method],'legend':lgnd};
            #     PLTS1.addPlot2D(ax,pen=PENS[method + '_opt'],ins=params1);
            #     PLTS1.addPlot2D(ax,pen=PENS[method + '_nl'],ins=params2);



        # ============================================================
        # Cylindrical Keepout Zones
        # ============================================================
        if show_nfzs:
            ax = axs[0]
            mission = trajopt_obj.mission; method = trajopt_obj.method
            n_nfz = mission.n_nfz;
            temp = trajopt_obj.mission.zi[0] - mission.planet['r']
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



        # axs[0].set_aspect('equal')
        # axs[1].set_aspect('equal')

        # axs[0].view_init(elev=50,azim=-20); #, azim=45)
        

        
        for j in plot_inds:
         
            ax = axs[j];
            ax.grid('True')

            # xlabelpad = 0; ylabelpad = 0; zlabelpad = 0;
            # labelpads = 
            # if 'xlabelpad' in xlabelpad = ins['xlabelpad']
            # if 'ylabelpad' in ylabelpad = ins['ylabelpad']
            # if 'zlabelpad' in zlabelpad = ins['zlabelpad']

            params = {};
            params['title'] = {'text':titles[j],'fontsize':20,**titleinfo}
            params['xlabel'] = {'label':xlabels[j],'fontsize':16,**xlabelinfo}
            params['ylabel'] = {'label':ylabels[j],'fontsize':16,**ylabelinfo}
            if j==0: params['zlabel'] = {'label':zlabels[j],'fontsize':16,**zlabelinfo}
            params['ticks'] = {'labelsize':20,'width':2,**ticksinfo};
            PLTS1.setParams(ax,params);
            if j in uselegend: PLTS1.addLegend(ax,lgnd,ins={'fontsize':12,'loc':'best',**legendinfo});



            if version == 'sa_iters':
                if j == 0: 
                    # # PLOT ASPECT RATIO FIXING (NEEDS TO BE DONE MANUALLY FOR 3D PLOTS :( )
                    ax.set_xlim3d([-2,4])
                    ax.set_ylim3d([-2,4])
                    ax.set_zlim3d([0,5])
                    ax.view_init(elev=20, azim=160)
                    # axs[0].view_init(elev=25,azim=30); #, azim=45)        
                else: 
                    ax.set_xlim([-2.5,5])
                    ax.set_ylim([-2.5,5])

            else: 
                if j == 0: 
                    # # PLOT ASPECT RATIO FIXING (NEEDS TO BE DONE MANUALLY FOR 3D PLOTS :( )
                    x_lim = ax.get_xlim3d()
                    y_lim = ax.get_ylim3d()
                    z_lim = ax.get_zlim3d()
                    max_lim = max(abs(x_lim[1] - x_lim[0]), abs(y_lim[1] - y_lim[0]), abs(z_lim[1] - z_lim[0]))
                    x_mid = sum(x_lim) * 0.5
                    y_mid = sum(y_lim) * 0.5
                    ax.set_xlim3d([x_mid - max_lim * 0.5, x_mid + max_lim * 0.5])
                    ax.set_ylim3d([y_mid - max_lim * 0.5, y_mid + max_lim * 0.5])
                    ax.set_zlim3d([0, max_lim])
                    ax.view_init(elev=20, azim=160)
                    # axs[0].view_init(elev=25,azim=30); #, azim=45)        
                else: 
                    x_lim = ax.get_xlim()
                    y_lim = ax.get_ylim()
                    max_lim = max(abs(x_lim[1] - x_lim[0]), abs(y_lim[1] - y_lim[0]))
                    x_mid = sum(x_lim) * 0.5
                    y_mid = sum(y_lim) * 0.5
                    temp = 1.; 
                    ax.set_xlim([x_mid - max_lim * temp, x_mid + max_lim * temp])
                    ax.set_ylim([y_mid - max_lim * temp, y_mid + max_lim * temp])


        
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
    trajopt_obj = ins['trajopt_obj'];
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

    titles = {}; ylabels = {}; xlabels = {ind:'Time [$U_T$]' for ind in range(4)};
    titles[2] = ' ';# 'Flight Path Angle vs Time';
    titles[3] = ' ';# 'Heading vs Time';
    titles[0] = ' ';# 'Altitude vs Time';
    titles[1] = ' ';# 'Velocity vs Time';

    ylabels[0] = 'Thrust Mag. [$U_M U_L/U^2_T$]';
    ylabels[1] = '$\omega_{B,x},\omega_{B,y},\omega_{B,z}$ [deg/$U_T$]';    
    ylabels[2] = 'Tilt [deg]';
    ylabels[3] = 'Angular Rate [deg/$U_T$]';
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

        plot_inds = [0,1,2,3] # replace with appropriate state indices

        # 0: 'thrust_mag'
        # 1: states, (12,13,14)

        
        lgnd = 'Fig7'; PLTS1.dumpLegend(lgnd)

        
        for j,sind in enumerate(plot_inds):
            ax = axs[j];
            for method in methods: 
                PLTS1.setCurrent({'scenarios':scenarios,'methods':[method],'runs':runs})

                if j == 0: 
                    ytag_init = 'thrust_mag_init';
                    ytag_nl = 'thrust_mag_nl'
                    ytag_opt = 'thrust_mag_opt'
                if j == 1:
                    sinds = (11,12,13)
                    ytag_init = (z_init_tag,sinds)
                    ytag_nl = ('z_nl',sinds)
                    ytag_opt = ('z_opt',sinds)
                if j == 2:
                    ytag_init = 'tilt_init'
                    ytag_nl = 'tilt_nl'
                    ytag_opt = 'tilt_opt'
                if j == 3:
                    ytag_init = 'ang_rate_init'
                    ytag_nl = 'ang_rate_nl'
                    ytag_opt ='ang_rate_opt'

                ###############################################################
                ### NEEDS TO BE UPDATED 
                if False: 
                    if j == 0: #thrust mag
                        # #### hack for adding max value line... not that hacky anyway
                        penn = PENS['max-value'];
                        valmin = trajopt_obj.mission.u_min[0]*(180/np.pi) # CHANGE 
                        valmax = trajopt_obj.mission.u_max[0]*(180/np.pi) # CHANGE 
                        lrgba = penn['lrgba']; ls = penn['ls']; lw = penn['lw']
                        line_handle = ax.axhline(y=valmin, color=lrgba, linestyle=ls, linewidth=lw); # label=line_tag)
                        line_handle = ax.axhline(y=valmax, color=lrgba, linestyle=ls, linewidth=lw); #, label=line_tag)                    
                    if j == 1: # angular vector
                        penn = PENS['max-value'];
                        valmin = trajopt_obj.mission.u_min[0]*(180/np.pi) # CHANGE 
                        valmax = trajopt_obj.mission.u_max[0]*(180/np.pi) # CHANGE 
                        lrgba = penn['lrgba']; ls = penn['ls']; lw = penn['lw']
                        line_handle = ax.axhline(y=valmin, color=lrgba, linestyle=ls, linewidth=lw); # label=line_tag)
                        line_handle = ax.axhline(y=valmax, color=lrgba, linestyle=ls, linewidth=lw); #, label=line_tag)                    
                    if j == 2: # tilt
                        penn = PENS['max-value'];
                        valmin = trajopt_obj.mission.u_min[0]*(180/np.pi) # CHANGE 
                        valmax = trajopt_obj.mission.u_max[0]*(180/np.pi) # CHANGE 
                        lrgba = penn['lrgba']; ls = penn['ls']; lw = penn['lw']
                        line_handle = ax.axhline(y=valmin, color=lrgba, linestyle=ls, linewidth=lw); # label=line_tag)
                        line_handle = ax.axhline(y=valmax, color=lrgba, linestyle=ls, linewidth=lw); #, label=line_tag)                    
                    if j == 3: # angular rate
                        penn = PENS['max-value'];
                        valmin = trajopt_obj.mission.u_min[0]*(180/np.pi) # CHANGE 
                        valmax = trajopt_obj.mission.u_max[0]*(180/np.pi) # CHANGE 
                        lrgba = penn['lrgba']; ls = penn['ls']; lw = penn['lw']
                        line_handle = ax.axhline(y=valmin, color=lrgba, linestyle=ls, linewidth=lw); # label=line_tag)
                        line_handle = ax.axhline(y=valmax, color=lrgba, linestyle=ls, linewidth=lw); #, label=line_tag)                    



                ###############################################################

                if version in ['standalone']:
                    #TODO (CARLOS / SKYE): init should be in a better place, currently need ot index into iter k >= 1
                    params1 = {'label':'Initial guess','x':t_init_tag,'y':ytag_init,'iters':[1],'legend':lgnd,};
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
                    params1 = {'label':'Initial guess','x':t_init_tag,'y':ytag_init,'iters':[1],'legend':lgnd,};
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


        #### PLOTTING LIMITS>...
        ## HAND CODED HACK
        for j,sind in enumerate(plot_inds):
            ax = axs[j]; #state_plot_inds[j]];
            line_tag = 'Max/Min Value';
            penn = PENS['max-value']; lrgba = penn['lrgba']; ls = penn['ls']; lw = penn['lw']
            if j == 0:  # thrust_mag 
                mmin = 0.3; mmax = 5.0;
                line_handle = ax.axhline(y=mmin, xmin = 0, color=lrgba, linestyle=ls, linewidth=lw,label=line_tag); # label=line_tag)
                PLTS1.legends[lgnd][line_tag] = line_handle;
                line_handle = ax.axhline(y=mmax, xmin = 0, color=lrgba, linestyle=ls, linewidth=lw,label=line_tag); #, label=line_tag)
            # if j == 2: # tilt mag
            #     line_handle = ax.axhline(y=mmin, xmin = 0, color=lrgba, linestyle=ls, linewidth=lw,label=line_tag); # label=line_tag)
            #     PLTS1.legends[lgnd][line_tag] = line_handle;
            #     line_handle = ax.axhline(y=mmax, xmin = 0, color=lrgba, linestyle=ls, linewidth=lw,label=line_tag); #, label=line_tag)            
            if j == 3: # angular rate
                mmax = 1.5707*180/np.pi
                line_handle = ax.axhline(y=mmax, xmin = 0, color=lrgba, linestyle=ls, linewidth=lw,label=line_tag); # label=line_tag)
                PLTS1.legends[lgnd][line_tag] = line_handle;
                # line_handle = ax.axhline(y=mmax, xmin = 0, color=lrgba, linestyle=ls, linewidth=lw,label=line_tag); #, label=line_tag)


            if False: 
                if sind in trajopt_obj.mission.z_min_idx:
                    iind = np.where(trajopt_obj.mission.z_min_idx == sind)[0][0]; zmin = trajopt_obj.mission.z_min[iind]
                    line_handle = ax.axhline(y=zmin, xmin = 0, color=lrgba, linestyle=ls, linewidth=lw,label=line_tag); # label=line_tag)
                    PLTS1.legends[lgnd][line_tag] = line_handle;

                if sind in trajopt_obj.mission.z_max_idx:
                    iind = np.where(trajopt_obj.mission.z_max_idx == sind)[0][0];
                    zmax = ((trajopt_obj.mission.z_max[iind]-1)*trajopt_obj.mission.planet['r'])/1000; 
                     # - trajopt_obj.mission.planet['r']
                    line_handle = ax.axhline(y=zmax, xmin = 0, color=lrgba, linestyle=ls, linewidth=lw,label=line_tag); #, label=line_tag)
                    # PLTS1.legends[lgnd][line_tag] = line_handle;


        for j,sind in enumerate(plot_inds):
            ax = axs[j]; #state_plot_inds[j]];
            params = {};
            params['title'] = {'text':titles[j],'fontsize':20,**titleinfo}
            params['xlabel'] = {'label':xlabels[j],'fontsize':16,**xlabelinfo}
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
    # ax.tight_layout()

def makePlotCtrls(PLTS1,ins={}):

    ### LOADING DATA
    trajopt_obj = ins['trajopt_obj'];
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
                    params1 = {'label':'Initial guess','x':t_init_tag,'y':(nu_init_tag,sind),'iters':[1],'legend':lgnd};
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
                    params1 = {'label':'Initial guess','x':t_init_tag,'y':(nu_init_tag,sind),'iters':[1],'legend':lgnd};
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
            umin = trajopt_obj.mission.u_min[0]*(180/np.pi)
            umax = trajopt_obj.mission.u_max[0]*(180/np.pi)   
            # line_tag = 'Max-Value'
            # maxval = trajopt_obj.mission.path_limits[tag];
            # if tag == 'max_load': maxval = maxval/trajopt_obj.mission.planet['g']
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
    trajopt_obj = ins['trajopt_obj'];
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
                    params1 = {'label':'Initial guess','x':t_init_tag,'y':(nu_init_tag,j),'iters':[1],'legend':lgnd};
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
                    params1 = {'label':'Initial guess','x':t_init_tag,'y':(nu_init_tag,j),'iters':[1],'legend':lgnd};
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
            umin = trajopt_obj.mission.u_min[j]*(180/np.pi)
            umax = trajopt_obj.mission.u_max[j]*(180/np.pi)   
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





def makePlotLoads(PLTS1,ins={}):
    trajopt_obj = ins['trajopt_obj'];
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
            maxval = trajopt_obj.mission.path_limits[tag];

            penn = PENS['max-value'];
            lrgba = penn['lrgba']; ls = penn['ls']; lw = penn['lw']
            if tag == 'max_load': maxval = maxval/trajopt_obj.mission.planet['g']
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
    trajopt_obj = ins['trajopt_obj'];
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
    nfz_inds = trajopt_obj.indices.constraints.nonlinear_inequality['nfz'];
    pth_inds = winds = trajopt_obj.indices.constraints.nonlinear_inequality['path'];
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
    trajopt_obj = ins['trajopt_obj'];
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
    nfz_inds = trajopt_obj.indices.constraints.nonlinear_inequality['nfz'];
    pth_inds = trajopt_obj.indices.constraints.nonlinear_inequality['path'];

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

                t_opt_len = trajopt_obj.method.N; t_nl_len = int(t_opt_len * 20);
                # t_nl_len = trajopt_obj.method.Ndense
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
    trajopt_obj = ins['trajopt_obj'];
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
    nfz_inds = trajopt_obj.indices.constraints.nonlinear_inequality['nfz'];
    pth_inds = trajopt_obj.indices.constraints.nonlinear_inequality['path'];

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
    ctcs_idx = trajopt_obj.indices.z['ctcs']
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

                t_opt_len = trajopt_obj.method.N; t_nl_len = int(t_opt_len * 20);
                # t_nl_len = trajopt_obj.method.Ndense
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
def makePlotWghtsFlex(PLTS1,ins={}):
    trajopt_obj = ins['trajopt_obj'];
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

    weights_info = ['W_dyn','dual_dyn'];
    if 'weights_info' in ins: weights_info = ins['weights_info']

    #########################################
    ######  DEFAULTS FIG INFORMATION ########
    figsize = (10,4);
    grid = {};
    if len(weights_info) == 1:
        grid[0] = [0.05,0.05,0.9,0.9];    
        titles = {}; ylabels = {};
        titles[0] = ''; titles[1] = ''; titles[2] = ''; titles[3] = '';
        xlabels = {ind:'Time [s]' for ind in range(1)}
        # ylabels[0] = 'Quadratic Weights: \n CTCS dynamics';
        ylabels[0] = 'Quadratic Weights: \n Dynamics';
        uselegend = [0]

    if len(weights_info)==2:
        grid[0] = [0.05,0.05,0.4,0.9];    
        grid[1] = [0.6,0.05,0.4,0.9];        
        titles = {}; ylabels = {};
        titles[0] = ''; titles[1] = ''; titles[2] = ''; titles[3] = '';
        xlabels = {ind:'Time [s]' for ind in range(2)}
        # ylabels[0] = 'Quadratic Weights: \n CTCS Dynamics';
        # ylabels[1] = 'Dual Weights: \n CTCS Dynamics';
        ylabels[0] = 'Quadratic Weights: \n Dynamics';
        ylabels[1] = 'Dual Weights: \n Dynamics';
        uselegend = [1]

    if len(weights_info)==3:
        figsize = (10,3);
        grid[0] = [0.05,0.05,0.17,0.9];    
        grid[1] = [0.38,0.05,0.17,0.9];        
        grid[2] = [0.7,0.05,0.17,0.9];        
        titles = {}; ylabels = {};
        titles[0] = ''; titles[1] = ''; titles[2] = ''; titles[3] = '';
        xlabels = {ind:'Time [s]' for ind in range(3)}
        # ylabels[0] = 'Quadratic Weights: \n CTCS Dynamics';
        # ylabels[1] = 'Dual Weights: \n CTCS Dynamics';
        # ylabels[2] = 'Dual Weights: \n CTCS Dynamics';

        ylabels[0] = 'Quadratic Weights: \n Dynamics';
        ylabels[1] = 'Dual Weights: \n Dynamics';
        ylabels[2] = 'Dual Weights: \n Dynamics';
        uselegend = [0]

    if len(weights_info)==4:
        grid[0] = [0.05,0.60,0.3,0.35];
        grid[1] = [0.05,0.05,0.3,0.35];
        grid[2] = [0.55,0.60,0.3,0.35];
        grid[3] = [0.55,0.05,0.3,0.35];        
        state_inds = [0,3,4,5] # replace with appropriate state indices
        titles = {}; ylabels = {}; xlabels = {ind:'Time [s]' for ind in range(4)}
        titles[0] = ''; titles[1] = ''; titles[2] = ''; titles[3] = '';
        ylabels[0] = 'No-fly zone \n quadratic \n penalty weights';
        ylabels[1] = 'Path constraint \n quadratic \n penalty weights';
        ylabels[2] = 'No-fly zone linear \n penalty weights';
        ylabels[3] = 'Path constraint \n linear \n penalty weights';
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

        lgnd = 'Fig9wflex'; PLTS1.dumpLegend(lgnd)

        for j,info in enumerate(weights_info):
            ax = axs[j];

            for method in methods: 
                PLTS1.setCurrent({'scenarios':scenarios,'methods':[method],'runs':runs})
                t_opt_len = trajopt_obj.method.N;
                t_nl_len = int(t_opt_len * 20); # Hack
                if version in ['standalone']: 
                    ttag = ['t_opt',list(range(t_opt_len))]; #[:-1]];
                    params1 = {'label':'Initial guess','x':ttag,'y':info,'iters':[1],'legend':lgnd,'dataloc':'weights'};
                    params2 = {'label':'Iterations','x':ttag,'y':info,'iters':itrs,'legend':lgnd,'dataloc':'weights'};
                    params4 = {'label':'Optimal Solution','x':ttag,'y':info,'iters':[-1],'legend':lgnd,'dataloc':'weights'};
                    # PLTS1.addPlot2D(ax,pen=PENS['init'],ins=params1);
                    PLTS1.addPlot2D(ax,pen=PENS['opt2'] ,ins=params4);
                if version in ['sa_iters']: 
                    ttag = ['t_opt',list(range(t_opt_len))];
                    params1 = {'label':'Initial guess','x':ttag,'y':info,'iters':[1],'legend':lgnd,'dataloc':'weights'};
                    params2 = {'label':'Iterations','x':ttag,'y':info,'iters':itrs,'legend':lgnd,'dataloc':'weights'};
                    params4 = {'label':'Optimal Solution','x':ttag,'y':info,'iters':[-1],'legend':lgnd,'dataloc':'weights'};

                    # PLTS1.addPlot2D(ax,pen=PENS['init'],ins=params1);
                    PLTS1.addPlot2D(ax,pen=PENS['itr_opt'] ,ins=params2);
                    PLTS1.addPlot2D(ax,pen=PENS['fitr_opt2'] ,ins=params4);

                if version in ['methodvar','mvmc']:
                    ttag = ['t_opt',list(range(t_opt_len))];
                    params4 = {'label':method_labels[method],'x':ttag,'y':info,'iters':[-1],'legend':lgnd,'dataloc':'weights'};
                    PLTS1.addPlot2D(ax,pen=PENS[method + '_opt2'] ,ins=params4);
                if version == 'montecarlo':
                    ttag = ['t_opt',list(range(t_opt_len))]; #[:-1]];
                    params4 = {'label':method_labels[method],'x':ttag,'y':info,'iters':[-1],'color_vars':COLORVARS[method],'legend':lgnd,'dataloc':'weights'};
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
            figname = figpaths[kk] + 'weights2' + figadd + '.pdf'; #'bankangle1.pdf'
            plt.savefig(figname,bbox_inches='tight',pad_inches = 0,transparent=transparentfigs);
        if not(displayfigs): plt.clf();                    



# makePlot6(PLTS1,ins=plotparams);
def makePlotConvs(PLTS1,ins={}):
    trajopt_obj = ins['trajopt_obj'];
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
    ylabels[1] = 'Peak trajectory \n  residual';    
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
                        lenval = len(trajopt_obj.method.subprob.iter_data)
                        ydat = [trajopt_obj.method.subprob.iter_data[ii]['conv_data'][tag] for ii in range(lenval)[1:]]; 
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
    trajopt_obj = ins['trajopt_obj'];
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
    trajopt_obj = ins['trajopt_obj'];
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
        runs = list(range(1000))[1:];
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
                    # params1 = {'label':'Initial guess','tinds':[None],'y':Wtag,'iters':[1],'legend':lgnd,'dataloc':'weights'};
                    # params2 = {'label':'Iterations','tinds':[-1],'y':Wtag,'iters':itrs,'legend':lgnd,'dataloc':'weights'};
                    # params3 = {'label':'Propogated','tinds':[-1],'y':('z_nl',sind),'iters':[-1],'legend':lgnd,'dataloc':'weights'};
                    params4 = {'label':'Optimal Solution','tinds':[None],'y':Wtag,'iters':itrs,'legend':lgnd,'dataloc':'weights'};
                    # PLTS1.addPlot2DIter(ax,pen=PENS['init'] ,ins=params1); 
                    # PLTS1.addPlot2DIter(ax,pen=PENS['itr_opt'] ,ins=params2); 
                    # PLTS1.addPlot2DIter(ax,pen=PENS['nl'] ,ins=params3); 
                    PLTS1.addPlot2DIter(ax,pen=PENS['opt2'] ,ins=params4); 
                
                if version in ['methodvar','mvmc']:
                    params1 = {'label':method_labels[method],'tinds':[None],'y':Wtag,'iters':itrs,'legend':lgnd,'dataloc':'weights'};
                    PLTS1.addPlot2DIter(ax,pen=PENS[method + '_opt2'] ,ins=params1); 
                                
                if version == 'montecarlo':
                    params1 = {'label':method_labels[method],'tinds':[None],'y':Wtag,'iters':itrs,'color_vars':COLORVARS[method],'legend':lgnd,'dataloc':'weights'};
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
def makePlotConvsFlex(PLTS1,ins={}):
    trajopt_obj = ins['trajopt_obj'];
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
    
    converge_info = ['chk_feas_term','chk_feas_dyn'];
    if 'converge_info' in ins: converge_info = ins['converge_info']

    #########################################
    ######  DEFAULTS FIG INFORMATION ########
    figsize = (10,4);
    grid = {};
    if len(converge_info) == 1:
        grid[0] = [0.05,0.05,0.9,0.9];    
        titles = {}; ylabels = {};
        titles[0] = ''; titles[1] = ''; titles[2] = ''; titles[3] = '';
        xlabels = {ind:'Iterations' for ind in range(1)}
        ylabels[0] = 'Peak Constraint \n Violation';
        uselegend = [0]

    if len(converge_info)==2:
        grid[0] = [0.05,0.05,0.4,0.9];    
        grid[1] = [0.6,0.05,0.4,0.9];        
        titles = {}; ylabels = {};
        titles[0] = ''; titles[1] = ''; titles[2] = ''; titles[3] = '';
        xlabels = {ind:'Iterations' for ind in range(2)}
        ylabels[0] = 'Peak Constraint \n Violation';
        ylabels[1] = 'Peak trajectory \n  residual';    
        uselegend = [1]

    if len(converge_info)==3:
        figsize = (10,3);
        grid[0] = [0.05,0.05,0.17,0.9];    
        grid[1] = [0.38,0.05,0.17,0.9];        
        grid[2] = [0.7,0.05,0.17,0.9];        
        titles = {}; ylabels = {};
        titles[0] = ''; titles[1] = ''; titles[2] = ''; titles[3] = '';
        xlabels = {ind:'Iterations' for ind in range(3)}
        ylabels[0] = 'Peak Constraint \n Violation';
        ylabels[1] = 'Peak trajectory \n  residual';    
        ylabels[2] = 'Dual weights: \n CTCS dynamics';
        uselegend = [0]

    if len(converge_info)==4:
        grid[0] = [0.05,0.60,0.3,0.35];
        grid[1] = [0.05,0.05,0.3,0.35];
        grid[2] = [0.55,0.60,0.3,0.35];
        grid[3] = [0.55,0.05,0.3,0.35];        
        state_inds = [0,3,4,5] # replace with appropriate state indices
        titles = {}; ylabels = {};
        xlabels = {ind:'Iterations' for ind in range(4)}
        titles[0] = ''; titles[1] = ''; titles[2] = ''; titles[3] = '';
        ylabels[0] = 'Peak Constraint \n Violation';
        ylabels[1] = 'Peak trajectory \n  residual';    
        ylabels[2] = 'No-fly zone linear \n penalty weights';
        ylabels[3] = 'Path constraint \n linear \n penalty weights';
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
        ###########################################

        # grid = PLTS1.specGrid(typ='2x2'); 
        fig = plt.figure(figsize=figsize);
        axs = PLTS1.createGrid(fig,grid = grid);
        lgnd = 'Fig9cflex'; PLTS1.dumpLegend(lgnd)

        for j,tag in enumerate(converge_info):
            ax = axs[j];#state_plot_inds[j]];
            for method in methods:             
                
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
