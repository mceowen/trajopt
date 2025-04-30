import numpy as np
import numpy.linalg as mat, scipy.linalg as smat
import matplotlib.pyplot as plt
import matplotlib.patches as patches, matplotlib as mpl
from matplotlib.patches import Circle
from scipy.spatial import ConvexHull
import pandas as pd
from itertools import permutations as perm
from itertools import combinations
import itertools
from IPython.core.debugger import set_trace
import time
from matplotlib import animation
from matplotlib.patches import Polygon
import matplotlib.gridspec as gridspec, cvxpy as cvx
#from scipy.interpolate import spline, interp2dfd
from matplotlib.widgets import CheckButtons
from matplotlib.widgets import Slider
from ipywidgets import *
from io import StringIO
import pickle, os
from functools import *

# import sys
# sys.path.append('/Users/dan/Documents/code');
# from drawing import * 
# from mathz import *

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}'
import pygifsicle #import optimize
import imageio.v3 as iio
import imageio


#### CLASS/FUNCTION DEFINITIONS #### CLASS/FUNCTION DEFINITIONS #### CLASS/FUNCTION DEFINITIONS #### CLASS/FUNCTION DEFINITIONS 
#### CLASS/FUNCTION DEFINITIONS #### CLASS/FUNCTION DEFINITIONS #### CLASS/FUNCTION DEFINITIONS #### CLASS/FUNCTION DEFINITIONS 
#### CLASS/FUNCTION DEFINITIONS #### CLASS/FUNCTION DEFINITIONS #### CLASS/FUNCTION DEFINITIONS #### CLASS/FUNCTION DEFINITIONS 

class SCVXPLOT:
    def __init__(self,ins={}):
        ### DISCUSSION SECTION ### DISCUSSION SECTION ### DISCUSSION SECTION ### DISCUSSION SECTION 
        ### DISCUSSION SECTION ### DISCUSSION SECTION ### DISCUSSION SECTION ### DISCUSSION SECTION 
        ### MAIN FIELDS 


        self.default_prob = 'default_prob';

        self.data = {self.default_prob:1} 
        self.meta = {self.default_prob:1}

        self.plts = {}
        self.legs = {};

        data = {}; meta = {}; figs = {}; legs = {};
        if 'data' in ins:
            for prob in ins['data']: self.data[prob] = ins['data'][prob]
        # if 'meta' in ins:
        #     for prob in ins['meta']: self.meta[prob] = ins['meta'][prob]
        # if 'plts' in ins:
        #     for prob in ins['plts']: self.plts[prob] = ins['plts'][prob]

        self.default_style = {'lw': 1,'lc': [0,0,0,1],'fc':[0,0,0,0.2],'mc': [0,0,0,1],'ms': 2,'msty': 'o'};
        # self.default_meta = {'spatials':[0,1]};
        #class PEN

        self.probstyles = {}
        self.plotstyles = {}
        self.layerstyles = {}
        self.istyles = {}

        if 'plotstyles' in ins: self.addPlotStyles(ins['plotstyles'])
        if 'layerstyles' in ins: self.addLayerStyles(ins['layerstyles'])        
        if 'istyles' in ins:
            for prob in ins['istyles']: self.addIStyles(prob,ins['istyles'][prob])

        # if 'probmetas' in ins: self.addProbMetas(ins['probmetas']);
        # if 'plotmetas' in ins: self.addPlotMetas(ins['plotmetas']);
        # if 'layermetas' in ins: self.addLayerMetas(ins['layermetas']);
        # if 'imetas'in ins:
        #     for prob in ins['imetas']: self.addIMetas(prob,ins['imetas'][prob])

        

        # ### STYLE OF instance IN SUBPLOT 
        # instance = (method1,initcond1,sample1,epoch1,itr1,wypt1); #version1);
        # subplot = 'sub1';
        # # self.figs[prob]['defaults'] = default_style;
        # self.plts[prob][subplt] = {}
        # self.figs[prob][subplot]['default'] = default_style
        # self.figs[prob][subplot]['specifics'] = {}
        # self.figs[prob][subplot]['specifics'][instance] = style1;
        # self.figs[prob][subplot]['specifics'][instance] = style2;
        # #### STYLE OF instance IN LEGEND
        # leg = 'sub1';
        # self.legs[prob][leg]['default'] = default_style
        # instance = (method1,initcond1,sample1,epoch1,itr1,wypt1,0); #version1);
        # self.legs[prob][leg][instance] = style_leg1

        #### END OF DISCUSSION #### END OF DISCUSSION #### END OF DISCUSSION #### END OF DISCUSSION #### END OF DISCUSSION
        #### END OF DISCUSSION #### END OF DISCUSSION #### END OF DISCUSSION #### END OF DISCUSSION #### END OF DISCUSSION
        #### END OF DISCUSSION #### END OF DISCUSSION #### END OF DISCUSSION #### END OF DISCUSSION #### END OF DISCUSSION

        #### FIG DETAILS #### FIG DETAILS #### FIG DETAILS #### FIG DETAILS #### FIG DETAILS #### FIG DETAILS 
        #### FIG DETAILS #### FIG DETAILS #### FIG DETAILS #### FIG DETAILS #### FIG DETAILS #### FIG DETAILS 
        #### FIG DETAILS #### FIG DETAILS #### FIG DETAILS #### FIG DETAILS #### FIG DETAILS #### FIG DETAILS 

        self.frame = None;
        self.inputs = None;
        if 'frame' in ins: self.frame = ins['frame'];
        if 'inputs' in ins: self.inputs = ins['inputs'];

        self.folder = './';
        self.filename = 'fig';
        self.gifname = self.filename
        self.type = '.jpeg';
        self.makegif = False;
        if 'folder' in ins: self.folder = ins['folder'];

        self.giffolder = self.folder;
        if 'giffolder' in ins: self.giffolder = ins['giffolder'];
        if 'filename' in ins: self.filename = ins['filename'];
        if 'gifname' in ins: self.gifname = ins['gifname'];
        if 'type' in ins: self.type = ins['type']
        if 'makegif' in ins: self.makegif = ins['makegif'];
    
        self.figpath = self.folder + self.filename + self.type;
        self.gifpath = self.giffolder + self.gifname;

        if self.makegif:
            if not os.path.exists(self.gifpath): os.mkdir(self.gifpath)
            self.gifpath = self.gifpath + '/'

        self.transparent = True;
        if 'transparent' in ins: self.transparent = ins['transparent'];
        self.display_frame = True;
        self.print_frame = True;
        if 'display' in ins: self.display_frame = ins['display'];
        if 'print' in ins: self.print_frame = ins['print'];

        self.axparams = {}
        self.axparams['set_aspect'] = 'equal';
        self.axparams['axisoff'] = 'off';
        if 'set_aspect' in ins: self.axparams['set_aspect'] = ins['set_aspect'];
        if 'axisoff' in ins: self.axparams['axisoff'] = ins['axisoff'];

        self.naxs = 1;
        self.axsparams = {};
        if 'naxs' in ins: self.naxs = ins['naxs'];
        if 'axsparams' in ins: self.naxs = len(ins['axsparams']);
        for i in range(self.naxs):
            self.axsparams[i] = self.axparams.copy();
            if 'axsparams' in ins:
                if i in ins['axsparams']:
                    self.axsparams[i] = {**self.axsparams[i],**ins['axsparams'][i]}

        self.nk = 10;
        if 'nk' in ins: self.nk = ins['nk'];
        self.ks = list(range(self.nk));
        if 'ks' in ins: self.ks = ins['ks'];
        self.nk = len(self.ks);

        self.gifnumshift = 0; 
        if 'gifnumshift' in ins: self.gifnumshift = ins['gifnumshift']; 
        self.gifnumshift = int(self.gifnumshift);

    # file['meta'] = {'structure':(method,sample,epoch,itr,wypt)}
    # file['plotdata'] = {'defaults':{},'specifics':{}}
    # file['plotdata']['specifics'][] = {'ls':'-','lw':1}
    # file['params'] = {}
    # file['data'] = {}

    # file['data']['controls'] = {};
    # file['data']['states'] = {};
    # file['data']['states'][(method1,initcond1,sample1,epoch1,itr1,wypt1)] = np.array([nt,nx])
    # file['data']['states'][(method1,sample1,epoch1,itr1,wypt1)] = np.array([nt,nx])
    # # def setDefaults(self): pass
    # initial guess
    # propagated solution
    # plot over algorithm iterations
    # optimal solutions
    # monte carlo

    ###### TECH FUNCTIONS ###### TECH FUNCTIONS ###### TECH FUNCTIONS ###### TECH FUNCTIONS 
    ###### TECH FUNCTIONS ###### TECH FUNCTIONS ###### TECH FUNCTIONS ###### TECH FUNCTIONS 
    ###### TECH FUNCTIONS ###### TECH FUNCTIONS ###### TECH FUNCTIONS ###### TECH FUNCTIONS 

    def addStateData(self,prob,data,typs):
        for tag in data:
            self.data[prob]['states'][tag] = data[tag];
            self.data[prob]['state_types'][tag] = typs[tag];
    def addControlData(self,prob,data,typs):
        for tag in data:
            self.data[prob]['controls'][tag] = data[tag];
            self.data[prob]['control_types'][tag] = typs[tag];

    ############################################################
    def addProbStyles(self,styles):
        for tag in styles: self.probstyles[tag] = styles[tag];
    def addPlotStyles(self,styles):
        for tag in styles: self.plotstyles[tag] = styles[tag];
    def addLayerStyles(self,styles):
        for tag in styles: self.layerstyles[tag] = styles[tag];
    def addIStyles(self,prob,styles):
        if not(prob in self.istyles): self.istyles[prob] = {};            
        for tag in styles: self.istyles[prob][tag] = styles[tag];
    ############################################################
    # def addProbMetas(self,metas):
    #     for tag in metas: self.probmetas[tag] = metas[tag];
    # def addPlotMetas(self,metas):
    #     for tag in metas: self.plotmetas[tag] = metas[tag];
    # def addLayerMetas(self,metas):
    #     for tag in metas: self.layermetas[tag] = metas[tag];
    # def addIMetas(self,prob,metas):
    #     if not(prob in self.imetas): self.imetas[prob] = {};
    #     for tag in metas: self.imetas[prob][tag] = metas[tag];
    ############################################################        

    def syncDefaultStyles(self,probstyle = {},plotstyle={},layerstyle={},istyle={},ostyle={}):
        out = self.default_style;
        for tag in probstyle: out[tag] = probstyle[tag];
        for tag in plotstyle: out[tag] = plotstyle[tag];
        for tag in layerstyle: out[tag] = layerstyle[tag];
        for tag in istyle: out[tag] = istyle[tag];
        for tag in ostyle: out[tag] = ostyle[tag];
        return out

    def selectStyle(self,prob=None,plot=None,layer=None,inst=None,ostyle={}):
        if prob == None: prob = self.default_prob;
        probstyle = {};
        plotstyle = {};
        layerstyle = {};
        istyle = {};
        if prob in self.probstyles: probstyle = self.probstyles[prob];
        if plot in self.plotstyles: plotstyle = self.plotstyles[plot];
        if layer in self.layerstyles: layerstyle = self.layerstyles[layer];
        if prob in self.istyles:
            if inst in self.istyles[prob]: istyle = self.istyles[prob][inst];
        return self.syncDefaultStyles(probstyle=probstyle,plotstyle=plotstyle,layerstyle=layerstyle,istyle=istyle,ostyle=ostyle)


    ### TYPES OF PLOTS ### TYPES OF PLOTS ### TYPES OF PLOTS ### TYPES OF PLOTS ### TYPES OF PLOTS ### TYPES OF PLOTS 
    ### TYPES OF PLOTS ### TYPES OF PLOTS ### TYPES OF PLOTS ### TYPES OF PLOTS ### TYPES OF PLOTS ### TYPES OF PLOTS 
    ### TYPES OF PLOTS ### TYPES OF PLOTS ### TYPES OF PLOTS ### TYPES OF PLOTS ### TYPES OF PLOTS ### TYPES OF PLOTS 
    ### TYPES OF PLOTS ### TYPES OF PLOTS ### TYPES OF PLOTS ### TYPES OF PLOTS ### TYPES OF PLOTS ### TYPES OF PLOTS 
    ### TYPES OF PLOTS ### TYPES OF PLOTS ### TYPES OF PLOTS ### TYPES OF PLOTS ### TYPES OF PLOTS ### TYPES OF PLOTS 
    ### TYPES OF PLOTS ### TYPES OF PLOTS ### TYPES OF PLOTS ### TYPES OF PLOTS ### TYPES OF PLOTS ### TYPES OF PLOTS 

    ### 1. SPECIFIC PLOTS ### 1. SPECIFIC PLOTS ### 1. SPECIFIC PLOTS ### 1. SPECIFIC PLOTS 
    ### 1. SPECIFIC PLOTS ### 1. SPECIFIC PLOTS ### 1. SPECIFIC PLOTS ### 1. SPECIFIC PLOTS 
    ### 1. SPECIFIC PLOTS ### 1. SPECIFIC PLOTS ### 1. SPECIFIC PLOTS ### 1. SPECIFIC PLOTS 

    def plotStates(self,fig,ax,prob=None,plot=None,layer=None,insts=[],ostyles = {},args={},package='matplotlib'):

        if prob == None: prob = self.default_prob
        for i,inst in enumerate(insts):
            if isinstance(ostyles,dict): ostyle = ostyles;
            if isinstance(ostyles,list): ostyle = ostyles[i];
            style = self.selectStyle(prob=prob,plot=plot,layer=layer,inst=inst,ostyle=ostyle)
            if package == 'matplotlib':
                ### DETAILS ### DETAILS ### DETAILS ### DETAILS 
                data = self.data[prob]['states'][inst];
                xs = data;
                typs = self.data[prob]['state_types'][inst];
                typ = args['state_type'];
                inds = [i for i, val in enumerate(typs) if val == typ]
                if len(inds)>=2:
                    pts = xs[:,inds[:2]];
                    ax.plot(pts[:,0],pts[:,1],color=style['lc']);

    def plotControls(self,fig,ax,prob=None,plot=None,layer=None,insts=[],ostyles = {},args={},package='matplotlib'):
        if prob == None: prob = self.default_prob
        for i,inst in enumerate(insts):
            if isinstance(ostyles,dict): ostyle = ostyles;
            if isinstance(ostyles,list): ostyle = ostyles[i];
            style = self.selectStyle(prob=prob,plot=plot,layer=layer,inst=inst,ostyle=ostyle)
            if package == 'matplotlib':
                ### DETAILS ### DETAILS ### DETAILS ### DETAILS 
                data = self.data[prob]['controls'][inst];
                xs = data;
                typs = self.data[prob]['control_types'][inst];
                typ = args['control_type'];
                inds = [i for i, val in enumerate(typs) if val == typ]
                if len(inds)>=2:
                    pts = xs[:,inds[:2]];
                    ax.plot(pts[:,0],pts[:,1],color=style['lc']);                

    ### 2. GENERAL PLOTS ### 2. GENERAL PLOTS ### 2. GENERAL PLOTS ### 2. GENERAL PLOTS 
    ### 2. GENERAL PLOTS ### 2. GENERAL PLOTS ### 2. GENERAL PLOTS ### 2. GENERAL PLOTS 
    ### 2. GENERAL PLOTS ### 2. GENERAL PLOTS ### 2. GENERAL PLOTS ### 2. GENERAL PLOTS 

    def plotBlah1(self,fig,ax,prob=None,version=None,args={},package='matplotlib'):
        if version == 'typicalplot1': pass

    ### 3. CUSTOM PLOTS ### 3. CUSTOM PLOTS ### 3. CUSTOM PLOTS ### 3. CUSTOM PLOTS ### 3. CUSTOM PLOTS 
    ### 3. CUSTOM PLOTS ### 3. CUSTOM PLOTS ### 3. CUSTOM PLOTS ### 3. CUSTOM PLOTS ### 3. CUSTOM PLOTS 
    ### 3. CUSTOM PLOTS ### 3. CUSTOM PLOTS ### 3. CUSTOM PLOTS ### 3. CUSTOM PLOTS ### 3. CUSTOM PLOTS 

    def drawFrame(self,axs):
        for a,ax in enumerate(axs):
            ax.cla();
            if a in self.axsparams:
                params = self.axsparams[a];
                if 'lims' in params:
                    lims = params['lims'];
                    ax.set_xlim(lims[:2])
                    ax.set_ylim(lims[2:])
                if 'axisoff' in params:
                    axisoff = params['axisoff'];
                    ax.axis(axisoff)
                if 'set_aspect' in params:
                    ax.set_aspect(params['set_aspect'])

        self.inputs['isgif'] = False;
        self.frame(axs,self.inputs);
        if not(self.display_frame):
            for _,ax in enumerate(axs): ax.cla();
        if self.print_frame:
            plt.savefig(self.figpath,bbox_inches='tight',pad_inches = 0,transparent=self.transparent)

    def drawGif(self,axs,vmod=None):
        for k in self.ks:
            if not(vmod==None):
                if np.mod(k,vmod)==0:
                    end_time = time.time()
                    if not(k==0): print('time to plot 10 frames: ', end_time - start_time)
                    print('drawing frame... ',k)
                    start_time = time.time();
            for a,ax in enumerate(axs):
                ax.cla()
                if a in self.axsparams:     
                    params = self.axsparams[a];
                    if 'lims' in params:
                        lims = params['lims'];
                        ax.set_xlim(lims[:2])
                        ax.set_ylim(lims[2:])
                    if 'axisoff' in params:
                        axisoff = params['axisoff'];
                        ax.axis(axisoff)
                    if 'set_aspect' in params:
                        ax.set_aspect(params['set_aspect'])
            self.inputs['isgif'] = True;
            self.inputs['current_ind'] = k;
            self.inputs['all_inds'] = self.ks
            self.frame(axs,self.inputs);      
            plt.savefig(self.gifpath+'/'+str(k+self.gifnumshift).zfill(4)+self.type,bbox_inches='tight',pad_inches = 0,transparent=self.transparent)
            if k<self.nk-1:
              for ax in axs: ax.cla()

        #### CUSTOM PLOTTER #### CUSTOM PLOTTER #### CUSTOM PLOTTER 
        #### CUSTOM PLOTTER #### CUSTOM PLOTTER #### CUSTOM PLOTTER








