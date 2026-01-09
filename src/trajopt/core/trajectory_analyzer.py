from pathlib import Path

from trajopt.core.problem import Problem
from trajopt.core.solution_method import SolutionMethod
import trajopt.utils.config_loader as cfg
import trajopt.library.methods.scp as scp
from trajopt.core.analysis.standalone import run_standalone_analysis

from trajopt.core.analysis.trajplots import *
from trajopt.core.analysis.plots_for_scitech26_pdg import *

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
            method_name = "autotune"

            data = {'scenario1':self.scenario_data}
            PLTS1 = SCVXPLOTS(data);
            cases = {'scenarios':['scenario1'],'methods':['standard','autotune'],'runs':list(range(1000)),'iters':list(range(1000))[1:]}
            preProcess(PLTS1,self,cases=cases);


            
            ### WOULD RUN ALL OPTIONS ####
            # versions = ['standalone','sa_iters','methodvar','mvmc','montecarlo'];
            # figpaths = ['figs/standalone/','figs/sa_iters/','figs/methodvar/','figs/mvmc/','figs/montecarlo/'];

            #### CHANGE 
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
                makePlotTrajs(PLTS1,ins={**plotparams,**newparams});
                
                makePlotStates(PLTS1,ins=plotparams);
                
                ### looks in 'weights' plots over time 
                # # # newparams = {'weights_info':[('W_dyn',(0)),('dual_dyn',(0,1))]}
                newparams = {'weights_info':['W_dyn','dual_dyn']}
                makePlotWghtsFlex(PLTS1,ins={**plotparams,**newparams});
                
                # ### looks in 'conv_data' plots over iterations
                # # newparams = {'converge_info':['chk_feas_term','chk_feas_dyn']}
                makePlotConvsFlex(PLTS1,ins={**plotparams,**newparams});

            if True: 
                newparams = {'weights_info':['W_ineq','dual_ineq']}
                makePlotWghtsFlex(PLTS1,ins={**plotparams,**newparams});


        if temporary_plotting_name == "vtol1_entry_3dof":
            pass

        return self.scenario_data