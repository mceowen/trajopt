import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True
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



    ########### BASIC 2D-PLOTTING ###############
    def addPlot2D(self,ax,pen={},typ='line',ins={}):
        if len(pen)==0: penn = self.base_pen.copy();
        else: penn = {**self.base_pen,**pen}
        scenario = self.scenarios[0];
        if 'scenario' in ins: scenario = ins['scenario'];
        method = self.methods[scenario][0];
        if 'method' in ins: method = ins['method'];
        runs = [0]; iters = [0];
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

        for r in runs:
            RUNS = self.data[scenario][method]['runs'];
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


    ######## LABELS AND LEGENDS ############
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

            
