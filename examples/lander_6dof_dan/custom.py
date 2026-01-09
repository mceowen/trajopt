
import numpy as np
from trajopt.library.models.shared.geometry import *



# Define the golden ratio
phi = (1 + np.sqrt(5)) / 2

# Define the 20 vertices of a regular dodecahedron
dodec0 = np.array([
    [1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1],
    [0, 1/phi, phi], [0, -1/phi, phi], [0, -1/phi, -phi], [0, 1/phi, -phi],
    [1/phi, phi, 0], [-1/phi, phi, 0], [-1/phi, -phi, 0], [1/phi, -phi, 0],
    [phi, 0, 1/phi], [phi, 0, -1/phi], [-phi, 0, -1/phi], [-phi, 0, 1/phi],
    [1, -1, 1], [-1, 1, 1], [1, 1, -1], [-1, -1, -1]])


pts = 1.3*dodec0 + 1.*np.array([2,1,0]);
vertices,faces,affine = pts2faces3D(pts,tol=0.001)
##########################################
vertices_dodec = vertices.copy();
vertices2faces_dodec = faces.copy();
A_dodec = -affine[:,:3];
b_dodec = +affine[:,3]


pts = 1.*dodec0 + 1.*np.array([2,1,1]);
vertices,faces,affine = pts2faces3D(pts,tol=0.001)
##########################################
vertices_dodec_out2 = vertices.copy();
vertices2faces_dodec_out2 = faces.copy();
A_dodec_out2 = -affine[:,:3];
b_dodec_out2 = +affine[:,3]


pts = 0.8*dodec0 + 1.*np.array([2,1,1]);
vertices,faces,affine = pts2faces3D(pts,tol=0.001)
##########################################
vertices_dodec_out3 = vertices.copy();
vertices2faces_dodec_out3 = faces.copy();
A_dodec_out3 = -affine[:,:3];
b_dodec_out3 = +affine[:,3]

##### WAYPOINTS ###### 
wayscale = 0.6;
waypoint1 = 1.*np.array([3.5,3,2]);
waypoint2 = 1.*np.array([3.8,2,-1]);
waypoint3 = 1.*np.array([2,2,-1]);
waypoint4 = 1.*np.array([1.5,2,2]);
waypoint5 = 1.*np.array([4.,-0.5,2]);
waypoint6 = 1.*np.array([0,0,0]);
waypoint7 = 1.*np.array([0,0,0]);

time_steps1 = [5]
time_steps2 = [10]
time_steps3 = [10]
time_steps4 = [10]
time_steps5 = [9]
time_steps6 = []
time_steps7 = []


##########################################
pts = wayscale*dodec0 + waypoint1
vertices,faces,affine = pts2faces3D(pts,tol=0.001)
vertices_dodec1 = vertices.copy();
vertices2faces_dodec1 = faces.copy();
A_dodec1 = affine[:,:3];
b_dodec1 = -affine[:,3]
##########################################
pts = wayscale*dodec0 + waypoint2
vertices,faces,affine = pts2faces3D(pts,tol=0.001)
vertices_dodec2 = vertices.copy();
vertices2faces_dodec2 = faces.copy();
A_dodec2 = affine[:,:3];
b_dodec2 = -affine[:,3]
##########################################
pts = wayscale*dodec0 + waypoint3
vertices,faces,affine = pts2faces3D(pts,tol=0.001)
vertices_dodec3 = vertices.copy();
vertices2faces_dodec3 = faces.copy();
A_dodec3 = affine[:,:3];
b_dodec3 = -affine[:,3]


### BOX WAYPOINTS
upper_in0 = np.array([1,1,1])
lower_in0 = np.array([-1,-1,-1])
wayscale0 = 0.6

upper_in1 = wayscale0*upper_in0 + waypoint1
upper_in2 = wayscale0*upper_in0 + waypoint2
upper_in3 = wayscale0*upper_in0 + waypoint3
upper_in4 = wayscale0*upper_in0 + waypoint4
upper_in5 = wayscale0*upper_in0 + waypoint5
upper_in6 = wayscale0*upper_in0 + waypoint6
upper_in7 = wayscale0*upper_in0 + waypoint7

lower_in1 = wayscale0*lower_in0 + waypoint1
lower_in2 = wayscale0*lower_in0 + waypoint2
lower_in3 = wayscale0*lower_in0 + waypoint3
lower_in4 = wayscale0*lower_in0 + waypoint4
lower_in5 = wayscale0*lower_in0 + waypoint5
lower_in6 = wayscale0*lower_in0 + waypoint6
lower_in7 = wayscale0*lower_in0 + waypoint7



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