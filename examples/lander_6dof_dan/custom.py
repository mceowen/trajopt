
import numpy as np



## for a cube # as defined by the box constraints... 
affine2vertices = [[0,1,2],[3,1,2],[0,4,2],[3,4,2], # bottom face
                   [0,1,5],[3,1,5],[0,4,5],[3,4,5]] # top face
vertices2edges = [[0,1],[0,2],[1,3],[2,3], # bottom face edges
                  [4,5],[4,6],[5,7],[6,7], # top face edges
                  [0,4],[1,5],[2,6],[3,7]] # vertical edges
vertices2faces = [[0,1,3,2], #bottom face
                  [4,5,7,6], #top face,
                  [0,1,5,4], #vertical face 1
                  [0,2,6,4], #vertical face 2
                  [1,3,7,5], #vertical face 3
                  [2,3,7,6]] #vertical face 4
                  

# ['up','east','north']
upper_out = np.array([ 6., 3., 1.5]);
lower_out = np.array([-1., 1., -2.5]);

# # keep in region - not buffered
upper_in_convex = np.array([10., 4.,1.]);
lower_in_convex = np.array([-1.,-1.,-1.]);

upper_in_buffer = np.array([4., 4.,0.0]);
lower_in_buffer = np.array([-1.,-1.,-1.]);