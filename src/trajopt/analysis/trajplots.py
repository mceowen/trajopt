import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.rcParams['text.usetex'] = True
plt.rcParams['text.usetex'] = True
import numpy.linalg as mat
import scipy.linalg as smat

class SCVXPLOTS:
    def __init__(self,data):
        self.data = data;
        self.scenarios = list(self.data); self.methods = {};
        for tag in self.scenarios: self.methods[tag] = list(self.data[tag]);
        self.base_pen = {'frgba':[0,0,0,0.1],'lrgba':[0,0,0,0.1]}
        self.base_pen = {**self.base_pen,'lw':2,'ls':'-'}
        self.base_pen = {**self.base_pen,'msty':'','msz':1}
        self.legends = {};

        self.use_current = False;
        self.current_scenarios = [];
        self.current_methods = [];
        self.current_runs = []; 
        self.current_iters = [];

    def setCurrent(self,ins={}):
        self.use_current = True
        if 'methods' in ins: self.current_methods = ins['methods'];
        if 'scenarios' in ins: self.current_scenarios = ins['scenarios'];
        if 'runs' in ins: self.current_runs = ins['runs'];
        if 'iters' in ins: self.current_iters = ins['iters'];


    def calcField(self,tag,func,func_args=[],ins={}):
        # scenarios = [self.scenarios[0]];
        # methods = [self.methods[scenarios[0]][0]];
        # runs = [0];
        # iters = [0];
        if self.use_current:
            scenarios = self.current_scenarios;
            methods = self.current_methods;
            runs = self.current_runs;
            iters = self.current_iters;
        if 'scenarios' in ins: scenarios = ins['scenarios'];
        if 'methods' in ins: methods = ins['methods'];
        if 'runs' in ins: runs = ins['runs'];
        if 'iters' in ins: iters = ins['iters'];
        for scenario in scenarios:
            for method in methods:
                if method in self.data[scenario]:
                    RUNS = self.data[scenario][method]['mc_data'];
                    for r in runs:
                        if r < len(RUNS):
                            RUN = RUNS[r];
                            for i in iters:
                                if i < len(RUN['iters']):
                                    data = RUN['iters'][i];
                                    #############################
                                    lens = []; args = []
                                    for arg in func_args:
                                        newarg = arg; 
                                        if isinstance(arg,str): newarg = data[arg].copy();
                                        if isinstance(newarg,(list,np.ndarray)): lens.append(len(newarg))
                                        args.append(newarg);                                        
                                    totlen = 1; 
                                    if len(lens)>0: totlen = np.min(lens);
                                    fxvals = [];
                                    for t in range(totlen):
                                        currargs = []
                                        for arg in args:
                                            if isinstance(arg,(list,np.ndarray)): currargs.append(arg[t]);
                                            else: currargs.append(arg);
                                        fxvals.append(func(*currargs))
                                    fxvals = np.array(fxvals);
                                    self.data[scenario][method]['mc_data'][r]['iters'][i][tag] = fxvals;
                                    # lens = [];
                                    # for arg in func_args:
                                    #     given = False;
                                    #     if 'given' in arg: newarg = arg['val']; given = arg['given'];
                                    #     if not(given): newarg = data[arg['val']]
                                    #     if isinstance(newarg,(list,np.ndarray)): lens.append(len(newarg))
                                    # totlen = 1; 
                                    # if len(lens)>0: totlen = np.min(lens);
                                    # fxvals = [];                                    
                                    # for t in range(totlen):
                                    #     args = [];
                                    #     for arg in func_args:
                                    #         given = False;
                                    #         if 'given' in arg: newarg = arg['val']; given = arg['given'];
                                    #         if not(given): newarg = data[arg['val']]
                                    #         if 'inds' in arg: newarg = newarg[:,arg['inds']];
                                    #         if isinstance(newarg,(list,np.ndarray)): newarg[t]
                                    #         args.append(newarg);
                                    #     fxvals.append(func(*args))
                                    # fxvals = np.array(fxvals);
                                    # self.data[scenario][method]['mc_data'][r]['iters'][i][tag] = fxvals;


    ########### BASIC 2D-PLOTTING ###############
    def addPlot2D(self,ax,pen={},typ='line',ins={}):
        if len(pen)==0: penn = self.base_pen.copy();
        else: penn = {**self.base_pen,**pen}

        scenarios = [self.scenarios[0]];
        methods = [self.methods[scenarios[0]][0]];
        runs = [0];
        iters = [0];
        if self.use_current:
            scenarios = self.current_scenarios;
            methods = self.current_methods;
            runs = self.current_runs;
            iters = self.current_iters;
        if 'scenarios' in ins: scenarios = ins['scenarios'];
        if 'methods' in ins: methods = ins['methods'];
        if 'runs' in ins: runs = ins['runs'];
        if 'iters' in ins: iters = ins['iters'];


        xtag = None; ytag = None;
        if 'x' in ins: xtag = ins['x'];
        if 'y' in ins: ytag = ins['y'];
        leg = None;
        if 'legend' in ins: leg = ins['legend'];
        if not(leg in self.legends):  self.legends[leg] = {};
        label = 'blarg';
        if 'label' in ins: label = ins['label']

        for scenario in scenarios:
            for method in methods:
                if method in self.data[scenario]:
                    RUNS = self.data[scenario][method]['mc_data'];
                    for r in runs:
                        if r < len(RUNS):
                            RUN = RUNS[r];
                            for i in iters:
                                if i < len(RUN['iters']):
                                    data = RUN['iters'][i];
                                    if not(ytag == None):
                                        if isinstance(ytag,tuple): ydata = data[ytag[0]][:,ytag[1]];
                                        else: ydata = data[ytag];
                                        if xtag == None: xdata = list(range(len(ydata)));
                                        else:
                                            if isinstance(xtag,tuple): xdata = data[xtag[0]][:,xtag[1]];
                                            else: xdata = data[xtag];

                                        frgba = penn['frgba']; lrgba = penn['lrgba'];
                                        lw = penn['lw']; ls = penn['ls']
                                        msty = penn['msty']; msz = penn['msz']
                                        if typ == 'line':
                                            if leg == None: 
                                                ax.plot(xdata,ydata.T,color=lrgba[:3],alpha=lrgba[3],linewidth=lw,linestyle = ls,marker=msty,markersize=msz)
                                            else: 
                                                self.legends[leg][label] = ax.plot(xdata,ydata.T,
                                                    label=label,color=lrgba[:3],alpha=lrgba[3],
                                                    linewidth=lw,linestyle = ls,marker=msty,markersize=msz)[0]


    def setParams(self,ax,ins={}):
        if 'ticks' in ins: ax.tick_params(**ins['ticks'])
        if 'xticks' in ins: ax.tick_params(**ins['xticks'])
        if 'yticks' in ins: ax.tick_params(**ins['yticks'])
        if 'xlabel' in ins: temp = ins['xlabel']; label = temp['label']; temp.pop('label'); ax.set_xlabel(label,**temp)
        if 'ylabel' in ins: temp = ins['ylabel']; label = temp['label']; temp.pop('label'); ax.set_ylabel(label,**temp)
        if 'title' in ins: temp = ins['title']; title = temp['text']; temp.pop('text'); ax.set_title(title,**temp);


    ######## LABELS AND LEGENDS ############
    def setTicks(self,ax,x=False,y=False,ins={}):
        if x==True:  ax.tick_params(**ins);
        if y==True:  ax.tick_params(**ins);

    def setLabels(self,ax,xlabel='',ylabel='',ins={}):
        ax.set_xlabel(xlabel,**ins)
        ax.set_ylabel(ylabel,**ins);
    def setTitle(self,ax,title = '',ins={}):
        ax.set_title(title,**ins)
    def addLegend(self,ax,leg,labels=[],ins={}):
        if len(labels) == 0: labels = list(self.legends[leg]);
        handles = [self.legends[leg][label] for label in labels];
        ax.legend(handles,labels,**ins)

    ########## CONSTRUCT SUBPLOTS ##########
    def genGridTags(self,fig,typ=None,params={}):
        return self.createGrid(fig,typ=typ,grid=self.specGrid(typ=typ,params=params));
    def specGrid(self,typ=None,params={}):
        grid = {};
        if typ=='2x2':
            grid[(0,0)] = [0.05,0.05,0.45,0.4];
            grid[(0,1)] = [0.05,0.6,0.45,0.4];
            grid[(1,0)] = [0.55,0.05,0.45,0.4];
            grid[(1,1)] = [0.55,0.6,0.45,0.4];
        return grid; 
    def createGrid(self,fig,typ='manual',grid={}):
        if typ=='manual':
            axs = {};
            for tag in grid: axs[tag] = fig.add_axes(grid[tag])
        return axs;

            
