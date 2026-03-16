import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import trajopt.utils.tools as tools
# matplotlib.rcParams['text.usetex'] = True
plt.rcParams['text.usetex'] = True

class SCVXPLOTS:
    def __init__(self,data):
        self.data = data;
        
        self.base_pen = {'frgba':[0,0,0,0.1],
                         'lrgba':[0,0,0,0.1],
                         'lw':2,
                         'ls':'-',
                         'msty':'',
                         'msz':1
                         }
        
        self.legends = {}

    ########### BASIC 2D-PLOTTING ###############
    def addPlot2D(self,ax,pen={},ins={}):
        
        # extract mandatory (method, run, and iters) from ins
        method_name = ins['method']
        run         = ins['run']
        iters       = ins['iters']

        # x and y data
        x_path   = ins['x']
        x_idx    = ins.get('x_idx', slice(None))
        
        y_path   = ins['y']
        y_idx    = ins.get('y_idx', slice(None))
        
        # legend and label
        leg      = ins['legend']
        label    = ins['label']
        
        # pen data
        pen      = {**self.base_pen,**pen}
        frgba    = pen['frgba'] 
        lrgba    = pen['lrgba']
        lw       = pen['lw']
        ls       = pen['ls']
        msty     = pen['msty']
        msz      = pen['msz']

        iter_data_list = np.array(self.data[method_name]['runs'][run]['iters'])
            
        for i, iter_data in enumerate(iter_data_list[iters]):
            
            # get x and y data from iter_data
            x_data = tools.get_from_path(iter_data, x_path)
            y_data = tools.get_from_path(iter_data, y_path)

            # force everything to be 2D for plotting, dont change the time axis
            if x_data.ndim == 1: x_data = x_data[:, np.newaxis]
            if y_data.ndim == 1: y_data = y_data[:, np.newaxis]

            # hack to get y_data with N-1 points to match x_data
            if y_data.shape[0] == x_data.shape[0] - 1:
                x_data = x_data[:-1]

            ax.plot(x_data[:, x_idx], y_data[:, y_idx], color=lrgba[:3], alpha=lrgba[3], linewidth=lw, linestyle=ls, marker=msty, markersize=msz)

    ########### BASIC 3D-PLOTTING ###############
    def addPlot3D(self,ax,pen={},ins={}):
        
        method_name = ins.get('method', list(self.data.keys())[0])
        iters    = ins.get('iters', slice(None, -1))
        run      = ins.get('run', 0)
        
        # x, y, and z data
        x_path   = ins['x']
        x_idx    = ins.get('x_idx', slice(None))
        y_path   = ins['y']
        y_idx    = ins.get('y_idx', slice(None))
        z_path   = ins['z']
        z_idx    = ins.get('z_idx', slice(None))

        
        # legend and label
        leg      = ins['legend']
        label    = ins['label']
        
        # pen data
        pen      = {**self.base_pen,**pen}
        frgba    = pen['frgba'] 
        lrgba    = pen['lrgba']
        lw       = pen['lw']
        ls       = pen['ls']
        msty     = pen['msty']
        msz      = pen['msz']

        iter_data_list = np.array(self.data[method_name]['runs'][run]['iters'])
            
        for i, iter_data in enumerate(iter_data_list[iters]):
            
            # get x and y data from iter_data
            x_data = tools.get_from_path(iter_data, x_path)
            y_data = tools.get_from_path(iter_data, y_path)
            z_data = tools.get_from_path(iter_data, z_path)

            # force everything to be 2D for plotting, dont change the time axis
            if x_data.ndim == 1: x_data = x_data[:, np.newaxis]
            if y_data.ndim == 1: y_data = y_data[:, np.newaxis]
            if z_data.ndim == 1: z_data = z_data[:, np.newaxis]

            ax.plot(x_data[:, x_idx], y_data[:, y_idx], z_data[:, z_idx], color=lrgba[:3], alpha=lrgba[3], linewidth=lw, linestyle=ls, marker=msty, markersize=msz)

    ########### BASIC 2D-PLOTTING over iterations ###############
    def addPlot2D_iters(self,ax,pen={},ins={}):
        
        method_name = ins.get('method', list(self.data.keys())[0])
        run      = ins.get('run', 0)
        
        y_path   = ins['y']
        y_idx    = ins.get('y_idx', slice(None))

        
        # legend and label
        leg      = ins['legend']
        label    = ins['label']
        
        # pen data
        pen      = {**self.base_pen,**pen}
        frgba    = pen['frgba'] 
        lrgba    = pen['lrgba']
        lw       = pen['lw']
        ls       = pen['ls']
        msty     = pen['msty']
        msz      = pen['msz']

        iter_data_list = np.array(self.data[method_name]['runs'][run]['iters'][1:])
        last_iter_data = iter_data_list[-1]
            
        # get x and y data from last iter_data
        y_data_last = tools.get_from_path(last_iter_data, y_path)
        if y_data_last.ndim == 1:
            y_data_last = y_data_last[np.newaxis, :]

        y_data = np.zeros((len(iter_data_list), y_data_last.shape[1]))

        for i, iter_data in enumerate(iter_data_list):

            iter_y_data = tools.get_from_path(iter_data, y_path)
            if iter_y_data.ndim == 1:
                y_data[i, :] = iter_y_data
            
            elif iter_y_data.ndim == 2:
                y_data[i, :] = np.max(iter_y_data, axis=0)

        ax.plot(np.arange(1, len(iter_data_list) + 1), y_data[:, y_idx], color=lrgba[:3], alpha=lrgba[3], linewidth=lw, linestyle=ls, marker=msty, markersize=msz)

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

    def dumpLegend(self,leg):
        self.legends[leg] = {};

    ########## CONSTRUCT SUBPLOTS ##########
    def genGridTags(self,fig,typ=None,params={}):
        return self.createGrid(fig,typ=typ,grid=self.specGrid(typ=typ,params=params));


    def createGrid(self,fig,typ='manual',grid={},ins={}):
        plt_typ = '2D';
        if 'plt_typ' in ins: plttyp = ins['plt_typ'];
        if typ=='manual':
            axs = {};
            for tag in grid:
                if plt_typ == '3d': axs[tag] = fig.add_axes(grid[tag],projection='3d')
                else: axs[tag] = fig.add_axes(grid[tag])
        return axs;

    def createGrid2(self,fig,typ='manual',grid={},ins={}):
        plttyps = {};
        if 'plt_typs' in ins: 
            plttyps = ins['plt_typs'];
        
        if typ=='manual':
            axs = {};
            for tag in grid:
                if tag in plttyps and plttyps[tag] == '3D':
                    axs[tag] = fig.add_axes(grid[tag],projection='3d')
                else:
                    axs[tag] = fig.add_axes(grid[tag])
        return axs;