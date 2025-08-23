import numpy as np
import numpy.linalg as mat, scipy.linalg as smat
import matplotlib.pyplot as plt
import itertools
# import matplotlib.patches as patches, matplotlib as mpl
# from matplotlib.patches import Circle
# from scipy.spatial import ConvexHull

# from IPython.core.debugger import set_trace
import time
from matplotlib import animation
# from matplotlib.patches import Polygon
# import matplotlib.gridspec as gridspec, cvxpy as cvx

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}'


#### CLASS/FUNCTION DEFINITIONS #### CLASS/FUNCTION DEFINITIONS #### CLASS/FUNCTION DEFINITIONS #### CLASS/FUNCTION DEFINITIONS 
#### CLASS/FUNCTION DEFINITIONS #### CLASS/FUNCTION DEFINITIONS #### CLASS/FUNCTION DEFINITIONS #### CLASS/FUNCTION DEFINITIONS 
#### CLASS/FUNCTION DEFINITIONS #### CLASS/FUNCTION DEFINITIONS #### CLASS/FUNCTION DEFINITIONS #### CLASS/FUNCTION DEFINITIONS 

class SCVXPLOT:
    def __init__(self,ins={}):
        ### DISCUSSION SECTION ### DISCUSSION SECTION ### DISCUSSION SECTION ### DISCUSSION SECTION 
        ### DISCUSSION SECTION ### DISCUSSION SECTION ### DISCUSSION SECTION ### DISCUSSION SECTION 
        ### MAIN FIELDS 


        self.default_prob = 'default_prob';
        self.state_locs = {};

        self.data = {self.default_prob:1} 
        self.meta = {self.default_prob:1}
        self.numiters = {self.default_prob:0}


        self.plts = {}
        self.legs = {};
        data = {}; meta = {}; figs = {}; legs = {};

        # self.default_style = {'lw': 1,'lc': [0,0,0,1],'fc':[0,0,0,0.2],'mc': [0,0,0,1],'ms': 2,'msty': 'o'};
        self.default_style = {'frgba':[0,0,0,0.1],'lrgba':[0,0,0,0.3],
                               'mrgba':[0,0,0,0.3],
                              'lw':1,'msty':'','ms':2,
                              'lsty':'-',
                              }
        # self.default_meta = {'spatials':[0,1]};
        #class PEN

        self.probstyles = {}
        self.plotstyles = {}
        self.layerstyles = {}
        self.istyles = {}

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

        self.loadData(ins)

    def loadData(self,ins={},version='basic'):

        if version=='basic':
            if 'data' in ins:
                for prob in ins['data']:
                    self.data[prob] = ins['data'][prob]
            if 'plotstyles' in ins: self.addPlotStyles(ins['plotstyles'])
            if 'layerstyles' in ins: self.addLayerStyles(ins['layerstyles'])        
            if 'istyles' in ins:
                for prob in ins['istyles']: self.addIStyles(prob,ins['istyles'][prob])
            # if 'meta' in ins:
            #     for prob in ins['meta']: self.meta[prob] = ins['meta'][prob]
            # if 'plts' in ins:
            #     for prob in ins['plts']: self.plts[prob] = ins['plts'][prob]

        if version == 'skye':
            prob = self.default_prob;

            matrix_style = 'basic'
            if 'matrix_style' in ins: matrix_style = ins['matrix_style']

            model=''; tags=[]; state_locs = {};
            if 'prob' in ins: prob = ins['prob'];
            if 'model' in ins: model = ins['model']; 
            if 'tags' in ins: tags = ins['tags'];
            if 'state_locs' in ins: state_locs = ins['state_locs'];
            if len(state_locs)>0:
                self.state_locs = state_locs;
            if 'data' in ins:
                jsondata = ins['data'];
                self.data[prob] = {}
                numiters = len(jsondata['O'])
                self.numiters[prob] = numiters;

                for itr in range(numiters):
                    for tag in tags:
                        pts = jsondata['O'][itr][tag]
                        if isinstance(pts,list): pts = np.array(pts);
                        if len(model)>0: datatag = (model,itr,tag)
                        else: datatag = (itr,tag)
                        if matrix_style == 'matlab': pts = pts.T;
                        self.data[prob][datatag] = pts


    def axParams(self,ax,ins={}):

        package = 'matplotlib'
        if package in ins: package = ins['package'];

        if package == 'matplotlib':
            # aspect = 'auto'
            if 'aspect' in ins: ax.set_aspect(ins['aspect']);
            if 'title' in ins: ax.set_title(ins['title']);
            if 'xlabel' in ins: ax.set_xlabel(ins['xlabel'])
            if 'ylabel' in ins: ax.set_ylabel(ins['ylabel'])
            if 'xlabelparams' in ins: ax.set_xlabel(**ins['xlabelparams'])
            if 'ylabelparams' in ins: ax.set_ylabel(**ins['ylabelparams'])



# # Create a subplot with specific figure size
# fig, ax = plt.subplots(figsize=(8, 6))

# # Plot some example data
# x = [0, 1, 2, 3, 4, 5]
# y = [0, 1, 4, 9, 16, 25]
# ax.plot(x, y, label='y = x^2', color='blue', marker='o')

# # Add labels to the axes
# ax.set_xlabel('X-axis Label', fontsize=12)
# ax.set_ylabel('Y-axis Label', fontsize=12)

# # Add a title to the plot
# ax.set_title('Example Subplot with Customizations', fontsize=14)

# # Customize tick marks and labels
# ax.set_xticks([0, 1, 2, 3, 4, 5])  # Set specific tick positions on the x-axis
# ax.set_xticklabels(['Zero', 'One', 'Two', 'Three', 'Four', 'Five'], rotation=45, fontsize=10)  # Custom labels
# ax.set_yticks([0, 5, 10, 15, 20, 25])  # Set specific tick positions on the y-axis
# ax.tick_params(axis='both', which='major', labelsize=10)  # Adjust tick label size

# # Add a legend
# ax.legend(loc='upper left', fontsize=10)

# # Adjust layout for better spacing
# plt.tight_layout()

# # Show the plot
# plt.show()



    ############################################################
    def addLayerStyles(self,styles):
        for tag in styles: self.layerstyles[tag] = styles[tag];

    ############################################################
    def syncDefaultStyles(self,layerstyle={},style={}):
        out = self.default_style;
        for tag in layerstyle: out[tag] = layerstyle[tag];
        for tag in style: out[tag] = style[tag];
        return out    

    def selectStyle(self,layer=None,style={}):
        layerstyle = {};
        if layer in self.layerstyles: layerstyle = self.layerstyles[layer];
        return self.syncDefaultStyles(layerstyle=layerstyle,style=style)
    ############################################################        

    ### TYPES OF PLOTS ### TYPES OF PLOTS ### TYPES OF PLOTS ### TYPES OF PLOTS ### TYPES OF PLOTS ### TYPES OF PLOTS 
    ### TYPES OF PLOTS ### TYPES OF PLOTS ### TYPES OF PLOTS ### TYPES OF PLOTS ### TYPES OF PLOTS ### TYPES OF PLOTS 
    ### TYPES OF PLOTS ### TYPES OF PLOTS ### TYPES OF PLOTS ### TYPES OF PLOTS ### TYPES OF PLOTS ### TYPES OF PLOTS 
    ### TYPES OF PLOTS ### TYPES OF PLOTS ### TYPES OF PLOTS ### TYPES OF PLOTS ### TYPES OF PLOTS ### TYPES OF PLOTS 
    ### TYPES OF PLOTS ### TYPES OF PLOTS ### TYPES OF PLOTS ### TYPES OF PLOTS ### TYPES OF PLOTS ### TYPES OF PLOTS 
    ### TYPES OF PLOTS ### TYPES OF PLOTS ### TYPES OF PLOTS ### TYPES OF PLOTS ### TYPES OF PLOTS ### TYPES OF PLOTS 

    ### 1. SPECIFIC PLOTS ### 1. SPECIFIC PLOTS ### 1. SPECIFIC PLOTS ### 1. SPECIFIC PLOTS 
    ### 1. SPECIFIC PLOTS ### 1. SPECIFIC PLOTS ### 1. SPECIFIC PLOTS ### 1. SPECIFIC PLOTS 
    ### 1. SPECIFIC PLOTS ### 1. SPECIFIC PLOTS ### 1. SPECIFIC PLOTS ### 1. SPECIFIC PLOTS 


    def plotOpt(self,ax,ins={}):

        prob = self.default_prob
        if 'prob' in ins: prob = ins['prob'];
        showtags = {};
        if 'showtags' in ins: showtags = ins['showtags'];

        show0 = True;
        showiters = True;
        showopt = True;
        if 'init' in showtags: show0 = showtags['init'];
        if 'iters' in showtags: showiters = showtags['iters'];
        if 'opt' in showtags: showopt = showtags['opt'];

        if 'tag' in ins: tag = ins['tag'];
        if 'ttag' in ins: ttag = ins['ttag'];

        paramso = {};
        params0 = {};
        paramsi = {};

        if 'inds' in ins: params0['inds'] = ins['inds']
        if 'inds' in ins: paramsi['inds'] = ins['inds']
        if 'inds' in ins: paramso['inds'] = ins['inds']

        paramso['tags'] = [tag];
        paramso['ttags'] = [ttag];

        paramsi['tags'] = [tag];
        paramsi['ttags'] = [ttag];

        ttag0 = 'ts_ref';
        if tag == 'zs': tag0 = 'zs_ref';
        if tag == 'us': tag0 = 'us_ref';

        params0['tags'] = [tag0];
        params0['ttags'] = [ttag0];

        paramso['layer'] = 'opt';
        params0['layer'] = 'init';
        paramsi['layer'] = 'iters';

        itrs0 = [0];
        itrso = [self.numiters[prob]-1]
        itrsi = list(range(self.numiters[prob]));


        paramso['itrs'] = itrso;
        params0['itrs'] = itrs0;
        paramsi['itrs'] = itrsi;

        params0['curve_type'] = 'by_time';
        paramsi['curve_type'] = 'by_time';
        paramso['curve_type'] = 'by_time';

        if show0: self.plotCurves(ax,ins=params0);
        if showiters: self.plotCurves(ax,ins=paramsi);
        if showopt: self.plotCurves(ax,ins=paramso);



    def plotCurves(self,ax,ins={}):
        package = 'matplotlib'
        curve_type = 'by_time_step';
        verbose = False; 
        if 'package' in ins: package = ins['package'];
        if 'curve_type' in ins: curve_type = ins['curve_type'];
        if 'verbose' in ins: verbose = ins['verbose'];

        prob = self.default_prob
        models = []; tags = []; ttags = []; itrs = []; specs = []; inds = [];
        if 'prob' in ins: prob = ins['prob'];
        if 'models' in ins: models = ins['models']
        if 'itrs' in ins: itrs = ins['itrs'];
        if 'tags'  in ins: tags = ins['tags'];
        if 'ttags' in ins: ttags = ins['ttags'];
        if 'inds' in ins: inds = ins['inds'];

        arggs = [];
        if len(models)>0: arggs.append(models);
        if len(itrs)>0: arggs.append(itrs);
        if len(tags)>0: arggs.append(tags);
        targgs = [];
        if len(models)>0: targgs.append(models);
        if len(itrs)>0: targgs.append(itrs);
        if len(ttags)>0: targgs.append(ttags);


        insts = itertools.product(*arggs)
        tinsts = list(itertools.product(*targgs));

        if 'insts' in ins: insts = ins['insts']
        layer = None; style = {};
        if 'layer' in ins: layer = ins['layer'];
        if 'style' in ins: style = ins['style'];
        for i,inst in enumerate(insts):
            sty = self.selectStyle(layer=layer,style=style)
            if package == 'matplotlib':
                ### DETAILS ### DETAILS ### DETAILS ### DETAILS 
                if curve_type == 'by_time_step':
                    try: 
                        dat = self.data[prob][inst];
                        pts = dat;
                        # if len(inds) == 1: pts = pts[:,inds[0]];
                        if len(inds) > 0: pts = pts[:,inds];
                        pparams = {};
                        pparams['color'] = sty['lrgba'][:3];
                        pparams['alpha'] = sty['lrgba'][3];
                        pparams['linestyle'] = sty['lsty'];
                        pparams['linewidth'] = sty['lw'];                        
                        ax.plot(pts.T,**pparams);
                    except:
                        if verbose: print('plot failed for inst: ',inst)

                if curve_type == 'by_time':
                    try: 
                        tinst = tinsts[i];
                        # print(inst)
                        dat = self.data[prob][inst];
                        times = self.data[prob][tinst];
                        pts = dat
                        if len(inds) == 1: pts = pts[:,inds[0]];
                        elif len(inds) >= 2: pts = pts[:,inds];
                        pparams = {};
                        pparams['color'] = sty['lrgba'][:3];
                        pparams['alpha'] = sty['lrgba'][3];
                        pparams['linestyle'] = sty['lsty'];
                        pparams['linewidth'] = sty['lw'];
                        ax.plot(times,pts,**pparams)
                    except:
                        if verbose: print('plot failed for inst: ',inst)                        

                if curve_type == 'traj2D':
                    try: 
                        dat = self.data[prob][inst];
                        pts = dat
                        if len(inds) > 1: pts[:,inds];
                        ax.plot(pts[:,inds[0]],pts[:,inds[1]],color=sty['lc']);
                    except:
                        if verbose: print('plot failed for inst: ',inst)



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








