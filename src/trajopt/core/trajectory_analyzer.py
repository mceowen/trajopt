from trajopt.core.problem import Problem
from trajopt.core.solution_method import SolutionMethod
import trajopt.utils.config_loader as cfg
import trajopt.library.methods.scp as scp
from trajopt.core.analysis.standalone import run_standalone_analysis

from trajopt.core.analysis.trajplots import *

import importlib

class TrajectoryAnalyzer:
    def __init__(self, trajopt_config_path):

        # load configs
        trajopt_config = cfg.load_trajopt_config(trajopt_config_path)
        problem_config = trajopt_config['problem']
        method_config = trajopt_config['method']

        # build optimal control problem and solution method from configs
        self.problem = Problem(problem_config)
        self.method  = SolutionMethod(self.problem, method_config)

        self.solution = None
        self.scenario_data = None

    def solve(self):
        """
        Solve the optimal control problem using configured method.
        """
        self.solution = scp.run_scp(self)

    def analyze(self, temporary_plotting_name=None):
        """
        Perform analysis of the solution
        """

        # store scenario data struct for plotting
        self.scenario_data = run_standalone_analysis(self)


        if temporary_plotting_name == "lander_6dof":
            plot_module = importlib.import_module("trajopt.core.analysis.plots_for_scitech26_pdg")
            
            method_name = "autotune"

            data = {'scenario1':self.scenario_data}
            PLTS1 = SCVXPLOTS(data);
            cases = {'scenarios':['scenario1'],'methods':['standard','autotune'],'runs':list(range(1000)),'iters':list(range(1000))[1:]}
            plot_module.preProcess(PLTS1,self,cases=cases);

            versions = ['standalone','sa_iters'];
            figpaths = ['figs/standalone/','figs/standalone/'];

            displayfigs = True;
            printfigs = True; 
            transparentfigs = True; 

            specs = {}

            specs['standalone'] = {'methods':[method_name],'runs':[0],'itrs':[]};
            specs['sa_iters'] = {'methods':[method_name],'runs':[0],'itrs':list(range(1000))[1:]};
            specs['methodvar'] = {'methods':['standard',method_name],'runs':[0]}; #,'itrs':list(range(1000))[1:]};
            specs['mvmc'] = {'methods':['standard',method_name],'runs':list(range(10))}; #,'itrs':list(range(1000))[1:]};
            specs['montecarlo'] = {'methods':['standard'],'runs':list(range(1000))}; #'itrs':list(range(1000))[1:]};

            ######## --- ######## --- ######## --- ######## --- ######## --- ######## --- ######## --- ######## --- 
            ######## --- ######## --- ######## --- ######## --- ######## --- ######## --- ######## --- ######## --- 
            ######## --- ######## --- ######## --- ######## --- ######## --- ######## --- ######## --- ######## --- 

            ## default pens are set in the function plots_for_scitech26.py
            ## set new pens here. 
            PENS = {};  

            plotparams = {};
            plotparams['trajopt_obj'] = self
            plotparams['data'] = data;
            plotparams['versions'] = versions;
            plotparams['specs'] = specs;
            plotparams['PENS'] = PENS;
            plotparams['figpaths'] = figpaths;
            plotparams['transparentfigs'] = transparentfigs;
            plotparams['printfigs'] = printfigs;
            plotparams['displayfigs'] = displayfigs;

            ALL_PLOTS = True
            if ALL_PLOTS:

                newparams = {'usequiver':True,'sideviews':True,'skip':1}
                plot_module.makePlotTrajs(PLTS1,ins={**plotparams,**newparams});
                
                plot_module.makePlotStates(PLTS1,ins=plotparams);
                
                ### looks in 'weights' plots over time 
                newparams = {'weights_info':['W_dyn','dual_dyn']}
                # # # newparams = {'weights_info':[('W_dyn',(0)),('dual_dyn',(0,1))]}
                plot_module.makePlotWghtsFlex(PLTS1,ins={**plotparams,**newparams});
                
                # ### looks in 'conv_data' plots over iterations
                # # newparams = {'converge_info':['chk_feas_term','chk_feas_dyn']}
                plot_module.makePlotConvsFlex(PLTS1,ins={**plotparams,**newparams});

        if temporary_plotting_name == "vtol1_entry_3dof":
            pass

        if temporary_plotting_name == "cobra_entry_3dof":

            plot_module = importlib.import_module("trajopt.core.analysis.plots_for_scitech26")
            method_name = "autotune"
            data = {'scenario1':self.scenario_data}
            PLTS1 = plot_module.SCVXPLOTS(data);
            cases = {'scenarios':['scenario1'],'methods':['standard','autotune'],'runs':list(range(1000)),'iters':list(range(1000))[1:]}
            plot_module.preProcess(PLTS1,self,cases=cases);

            versions = ['standalone','sa_iters'];
            figpaths = ['figs/standalone/','figs/standalone/'];

            displayfigs = True;
            printfigs = True; 
            transparentfigs = True; 

            specs = {}

            specs['standalone'] = {'methods':[method_name],'runs':[0],'itrs':[]};
            specs['sa_iters'] = {'methods':[method_name],'runs':[0],'itrs':list(range(1000))[1:]};
            specs['methodvar'] = {'methods':['standard','autotune'],'runs':[0]}; #,'itrs':list(range(1000))[1:]};
            specs['mvmc'] = {'methods':['standard','autotune'],'runs':list(range(10))}; #,'itrs':list(range(1000))[1:]};
            specs['montecarlo'] = {'methods':['standard'],'runs':list(range(1000))}; #'itrs':list(range(1000))[1:]};

            ######## --- ######## --- ######## --- ######## --- ######## --- ######## --- ######## --- ######## --- 
            ######## --- ######## --- ######## --- ######## --- ######## --- ######## --- ######## --- ######## --- 
            ######## --- ######## --- ######## --- ######## --- ######## --- ######## --- ######## --- ######## --- 

            ## default pens are set in the function plots_for_scitech26.py
            ## set new pens here. 
            PENS = {};  
            PENS['newpen'] = {'frgba':[.0,.0,.0,.1],'lrgba':[.0,.0,.0,1.],'lw':1,'ls':'--','msty':'' ,'msz':3};

            ## common pens to change
            PENS['init'] = {'frgba':[.0,.0,.0,.1],'lrgba':[.0,.0,.0,1.],'lw':1,'ls':'--','msty':'' ,'msz':3};
            PENS['itr']  = {'frgba':[.0,.0,.0,.1],'lrgba':[.0,.0,1.,.2],'lw':1,'ls':'--','msty':'' ,'msz':3};
            PENS['opt']  = {'frgba':[.0,.0,.0,.1],'lrgba':[.0,.0,1.,1.],'lw':1,'ls':''  ,'msty':'o','msz':3};
            PENS['prop'] = {'frgba':[.0,.0,.0,.1],'lrgba':[1.,.0,.0,1.],'lw':1,'ls':'-' ,'msty':'' ,'msz':3};
            PENS['ref']  = {'frgba':[.0,.0,.0,.1],'lrgba':[.0,.0,.0,1.],'lw':1,'ls':'--','msty':'*','msz':3};
            PENS['standard']  = {'frgba':[.0,.0,.0,.1],'lrgba':[.0,.0,1.,1.],'lw':1,'ls':'-','msty':'o','msz':3};
            PENS[method_name]  = {'frgba':[.0,.0,.0,.1],'lrgba':[1.,.0,1.,1.],'lw':1,'ls':'-','msty':'o','msz':3};

            ## weight pens (not currently in use)
            # PENS['opt_weight']      = {'frgba':[.0,.0,.0,.1],'lrgba':[.0,.0,1.,1.],'lw':1,'ls':'-'  ,'msty':'o','msz':3};
            # PENS['opt_weight_0']    = {'frgba':[.0,.0,.0,.1],'lrgba':[.0,.0,1.,1.],'lw':1,'ls':'-'  ,'msty':'o','msz':3};
            # PENS['opt_weight_1']    = {'frgba':[.0,.0,.0,.1],'lrgba':[.2,.0,1.,1.],'lw':1,'ls':'-'  ,'msty':'o','msz':3};
            # PENS['opt_weight_2']    = {'frgba':[.0,.0,.0,.1],'lrgba':[.4,.0,1.,1.],'lw':1,'ls':'-'  ,'msty':'o','msz':3};
            # PENS['opt_weight_3']    = {'frgba':[.0,.0,.0,.1],'lrgba':[.6,.0,1.,1.],'lw':1,'ls':'-'  ,'msty':'o','msz':3};
            # PENS['opt_weight_4']    = {'frgba':[.0,.0,.0,.1],'lrgba':[.8,.0,1.,1.],'lw':1,'ls':'-'  ,'msty':'o','msz':3};
            # PENS['opt_weight_5']    = {'frgba':[.0,.0,.0,.1],'lrgba':[1.,.0,1.,1.],'lw':1,'ls':'-'  ,'msty':'o','msz':3};

            plotparams = {};
            plotparams['trajopt_obj'] = self
            plotparams['data'] = data;
            plotparams['versions'] = versions;
            plotparams['specs'] = specs;
            plotparams['PENS'] = PENS;
            plotparams['figpaths'] = figpaths;
            plotparams['transparentfigs'] = transparentfigs;
            plotparams['printfigs'] = printfigs;
            plotparams['displayfigs'] = displayfigs;

            ALL_PLOTS = False

            if ALL_PLOTS:
                plot_module.makePlotCtrls(PLTS1,ins=plotparams);
                plot_module.makePlotCtrls2(PLTS1,ins=plotparams);
                plot_module.makePlotTrajs(PLTS1,ins=plotparams);
                plot_module.makePlotStates(PLTS1,ins=plotparams);
                plot_module.makePlotLoads(PLTS1,ins=plotparams);
                plot_module.makePlotWghts(PLTS1,ins=plotparams);
                plot_module.makePlotWghts2(PLTS1,ins=plotparams);
                plot_module.makePlotWghts3(PLTS1,ins=plotparams);
                plot_module.makePlotConvs(PLTS1,ins=plotparams);
                plot_module.makePlotConvs2(PLTS1,ins=plotparams);

            params = {}

            grid = {};
            grid[0] = [0.05,0.05,0.37,0.9]; grid[1] = [0.50,0.05,0.37,0.9];
            params['grid'] = grid;

            params['uselegend'] = [1];

            plot_module.makePlotCtrls2(PLTS1,ins={**plotparams,**params});

            params = {}
            params['show_nfzs'] = False

            grid = {};
            grid[0] = [0.05,0.05,0.6,0.9]; grid[1] = [0.70,0.05,0.4,0.9];

            grid[0] = [0.05, 0.05, 0.58, 0.9]; 
            grid[1] = [0.65, 0.05, 0.23, 0.9];  # Increased left margin from 0.05 to 0.10

            params['grid'] = grid;

            params['legendinfo'] = {'fontsize':12}

            plot_module.makePlotTrajs(PLTS1,ins={**plotparams,**params});


            plot_module.makePlotStates(PLTS1,ins=plotparams);

            plot_module.makePlotLoads(PLTS1,ins=plotparams);

            # UNCOMMENT FOR NODAL INEQUALITY WEIGHTS
            # plot_module.makePlotWghts(PLTS1,ins=plotparams);

            # plot_module.makePlotWghts2(PLTS1,ins=plotparams);

            W_dyn = self.scenario_data["autotune"]["mc_data"][0]["iters"][-1]["weights"]["W_dyn"]

            # print(W_dyn)

            # UNCOMMENT FOR CTCS
            plot_module.makePlotWghts3(PLTS1,ins=plotparams);

            plot_module.makePlotConvs(PLTS1,ins=plotparams);

            plot_module.makePlotConvs2(PLTS1,ins=plotparams);

            plot_module.makePlotConvs3(PLTS1,ins=plotparams);

        return self.scenario_data