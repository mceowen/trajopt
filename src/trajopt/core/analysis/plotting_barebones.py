
PENS = {};

# PENS['z_opt'] = {'frgba':[0,0,0,0.1],'lrgba':[0,0,0,0.1],'lw':2,'ls':'-','msty':'','msz':4};
# standalone 
PENS['init'] = {'frgba':[.0,.0,.0,.1],'lrgba':[.0,.0,.0,1.],'lw':1,'ls':'--','msty':'' ,'msz':3};
PENS['nl'] = {'frgba':[.0,.0,.0,.1],'lrgba':[1.,.0,.0,1.],'lw':2,'ls':'-' ,'msty':'' ,'msz':3};
PENS['opt']  = {'frgba':[.0,.0,.0,.1],'lrgba':[.0,.0,1.,1.],'lw':1,'ls':''  ,'msty':'o','msz':3};
# iteration values
PENS['itr_opt']  = {'frgba':[.0,.0,.0,.1],'lrgba':[0.7,.0,0.3,.2],'lw':1,'ls':'','msty':'o' ,'msz':3};
PENS['itr_nl']   = {'frgba':[.0,.0,.0,.1],'lrgba':[0.7,.0,0.3,.4],'lw':1,'ls':'-','msty':'' ,'msz':3};
# final iteration values
PENS['fitr_opt']  = {'frgba':[.0,.0,.0,.1],'lrgba':[.0,.0,1.,1.],'lw':2,'ls':'','msty':'o' ,'msz':3};
PENS['fitr_nl']   = {'frgba':[.0,.0,.0,.1],'lrgba':[0.,.0,1.,1.],'lw':2,'ls':'-','msty':'' ,'msz':3};

# convergence
PENS['opt2']  = {'frgba':[.0,.0,.0,.1],'lrgba':[.0,.0,1.,1.],'lw':1,'ls':'-'  ,'msty':'o','msz':3};
PENS['fitr_opt2']  = {'frgba':[.0,.0,.0,.1],'lrgba':[.0,.0,1.,1.],'lw':1,'ls':'-'  ,'msty':'o','msz':3};

PENS['ref']  = {'frgba':[.0,.0,.0,.1],'lrgba':[.0,.0,.0,1.],'lw':1,'ls':'--','msty':'*','msz':3};
PENS['standard']  = {'frgba':[.0,.0,.0,.1],'lrgba':[.0,.0,1.,1.],'lw':1,'ls':'-','msty':'o','msz':3};
PENS['standard_nl']  = {'frgba':[.0,.0,.0,.1],'lrgba':[.0,.0,1.,1.],'lw':1,'ls':'-','msty':'','msz':3};
PENS['standard_opt']  = {'frgba':[.0,.0,.0,.1],'lrgba':[.0,.0,1.,1.],'lw':1,'ls':'','msty':'o','msz':3};
PENS['standard_opt2']  = {'frgba':[.0,.0,.0,.1],'lrgba':[.0,.0,1.,1.],'lw':1,'ls':'-','msty':'','msz':3};

PENS['autotune']  = {'frgba':[.0,.0,.0,.1],'lrgba':[1.,.0,1.,1.],'lw':1,'ls':'-','msty':'o','msz':3};
PENS['autotune_nl']  = {'frgba':[.0,.0,.0,.1],'lrgba':[1.,.0,1.,1.],'lw':1,'ls':'-','msty':'','msz':3};
PENS['autotune_opt']  = {'frgba':[.0,.0,.0,.1],'lrgba':[1.,.0,1.,1.],'lw':1,'ls':'','msty':'o','msz':3};
PENS['autotune_opt2']  = {'frgba':[.0,.0,.0,.1],'lrgba':[1.,.0,1.,1.],'lw':1,'ls':'-','msty':'','msz':3};

# TODO(Skye): Tune this line thickness
PENS['max-value'] = {'frgba':[.0,.0,.0,.1],'lrgba':[0.0,.0,0.,0.7],'lw':2.5,'ls':'-','msty':'','msz':0};

# for 3D plot
PENS['u_vec']  = {'frgba':[.0,.0,.0,.1],'lrgba':[1.,0.25,0.,1.],'lw':2,'ls':'-','msty':'','msz':0};
PENS['body_vec'] = {'frgba':[.0,.0,.0,.1],'lrgba':[0.,0.,0.,1.],'lw':2,'ls':'-','msty':'','msz':0};
# for 2D plots...
PENS['u_vec2']  = {'frgba':[.0,.0,.0,.1],'lrgba':[1.,0.25,0.,1.],'lw':4,'ls':'-','msty':'','msz':0};
PENS['body_vec2'] = {'frgba':[.0,.0,.0,.1],'lrgba':[0.,0.,0.,1.],'lw':4,'ls':'-','msty':'','msz':0};



### potential new data structure
# DATA = {};
# DATA['meta'] = # database information stuff
# DATA['data'] = {('scenario1','auto',run):{'itrs':[]}


## yaml file should have 
# grid info -- how to setup a grid of subplots
# y-axes labels, and subplot titles  - for each subplot
# state name and indices - what to put on each subplot

PLTS1 = SCVXPLOTS(data);

### Manual grid
## make four subplots - [x position,y position, x width , y width]    
grid = {};
grid[0] = [0.05,0.6,0.35,0.35];
grid[1] = [0.51,0.6,0.35,0.35];
grid[2] = [0.05,0.05,0.35,0.35];
grid[3] = [0.51,0.05,0.35,0.35];

### Automatic grid 
def makeSquareGridSpecs(num_states):
  num = num_states;
  colnum = int(np.sqrt(num)) + 1;
  rownum = colnum;
  dx = 0.8/colnum; dy = dx;
  grid = {};
  for i in range(rownum):
    for j in range(colnum):
      x = 0.05 + dx*j; y = 0.05 + dy*i
      grid[i*rownum + colnum] = [x,y,dx,dy];
  return grid
grid = makeSquareGridSpecs(num_states)
########################################


fig = plt.figure(figsize=(10,10));
axs = PLTS1.createGrid(fig,grid = grid); # makes the subplot from the grid info
ax = axs[0] # grab the first subplot
################################################
scenarios = ['scenario1']; methods = ['autotune']; runs = list(range(1000)); iters = list(range(1000))
# set default scenarios, methods, and runs
PLTS1.setCurrent({'scenarios':scenarios,'methods':methods,'runs':runs,'iters':iters}) 

PLTS1.dumpLegend('legend1') # clears info from 'legend1'
#########################################
### all options to pass to addPlot2D and addPlot2DIters
params = {'scenarios': scenarios, # which scenarios to loop through
           'methods': methods, # which methods to loop through
           'runs': runs, # which runs to loop through
           'iters':iters, # which iterations to loop through
           ### only for addPlot2DIter 
           'tinds':[None], # i think this does nothing at the moment... but it does need to be [None]
           #######            
           'x':xtag, # where to get the x data. example: 't_nl', ('z_nl',0), etc
           'y':ytag, # where to get the ydata. 'z_nl',('z_nl',1),('z_nl'(1,2));
           'legend':'legend1', # which legend to put the plot on
           'label': 'label1', # what should the plot be called on the legend.
           'force_lens': True, # if x and y data are different lengths make it work
           'dataloc': 'weights' or 'conv_data', # if ydata not in regular place in iters, where to look for it a level deeper
           'use_quiver': False, #make a quiver plot -- hacky
           'skip': 2, # downsample data points
           }

for j in range(num_states):
  ### add plot of final iterations ie. iters = [-1]
  params = {'label':'state '+str(j),'x':'t_nl','y':('z_nl',j),'iters':[-1],'legend':'legend1'};
  PLTS1.addPLot2D(axs[j],pen=PENS['nl'],ins=params)
  ### add plot of all iterations ie. iters = iters
  params = {'label':'state '+str(j),'x':'t_nl','y':('z_nl',j),'iters':iters,'legend':'legend1'};
  PLTS1.addPLot2D(axs[j],pen=PENS['itr_nl'],ins=params)

params2 = {'label':'Iterations','tinds':[None],'y':tag,'iters':itrs,'legend':'legend1','dataloc':'conv_data'};
PLTS1.addPLot2DIter(ax,pen=PENS['basic_pen2'],ins=params2)

params = {};
params['title'] = {'text':'title','fontsize':20}
params['xlabel'] = {'label':'Time [$U_T$]','fontsize':16}
params['ylabel'] = {'label':'some state','fontsize':16}
params['ticks'] = {'labelsize':20,'width':}
PLTS1.setParams(ax,params)
PLTS1.addLegend(ax,'legend1',ins={'fontsize':12,'loc':'best'});

plt.savefig(filename,bbox_inches='tight',pad_inches = 0,transparent=True);

