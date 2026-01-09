import jax 
import jax.numpy as jnp
import cvxpy as cp

import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull


def orderFacePts3D(nodes,tags=[]):
    ### puts points in order assuming they are all on the same face
    newnodes = []; newtags = tags;
    if len(nodes)>0:
      n = len(nodes);
      center = (1./n)*np.ones(n)@nodes;
      diffs = nodes - center;
      normal = np.cross(diffs[0],diffs[1]);
      basis1 = diffs[0]; basis2 = np.cross(normal,diffs[0]);
      basis1 = (1./np.linalg.norm(basis1))*basis1; basis2 = (1./np.linalg.norm(basis2))*basis2;
      basis =  np.array([basis1,basis2]);
      X = basis.T; pts = (np.linalg.inv(X.T@X)@X.T@diffs.T).T;
      angles = [float(np.arctan2(pt[1],pt[0])) for pt in pts];
      ##############################
      inds = np.argsort(angles)
      newtags = [];
      if len(tags)>0: newtags = [tags[ind] for ind in inds];
      newnodes = nodes[inds];
    return newnodes,newtags

def pts2faces3D(pts,tol=0.001):
    # takes pts
    # - selects out vertices of convex hull 
    # - generate faces from simplices and order points
    # - return equations,faces,edges?(need to add )
    # affine = [normals,offsets] # same form as equations from scipy.spatial.ConvexHull
    # compares simplicies to see if they are part of the same face (up to some tolerance)
    # computes the normal directions of simplicies

    # The simplest way to get the face indices for a regular convex polyhedron 
    # in matplotlib is to use the ConvexHull function from SciPy, which automatically 
    # computes the facets (faces) from the vertices.
    hull = ConvexHull(pts)
    affine = hull.equations;
    vertices0 = np.array(hull.vertices); simplices0 = np.array(hull.simplices)

    newpts = pts[vertices0];
    vertices = list(range(len(vertices0)));
    simplices = [];
    for simplex in simplices0: simplices.append([np.where(vertices0==ind)[0][0] for ind in simplex])
    simplices = np.array(simplices);

    normals = affine[:,:3]; offsets = affine[:,3]; m = len(affine);
    # inside is normals@x + offsets <= 0
    # normalizing...
    diag = np.array([np.linalg.norm(normal) for normal in normals])
    normals = np.diag(1./diag)@normals; offsets = np.diag(1./diag)@offsets; 
    #####
    already_taken = []; co_inds = [];
    for j1,normal1 in enumerate(normals):
      if not(j1 in already_taken):
        co_inds.append([j1])
        offset1 = offsets[j1]
        for j2,normal2 in enumerate(normals):
          if j2 > j1 and not(j2 in already_taken):
            offset2 = offsets[j2];
            compare = np.dot(normal1,normal2)
            if np.abs(compare) > 1-tol:
              if compare < 0.0: normal2 = -normal2; offset2 = -offset2;
              if np.isclose(offset1,offset2): co_inds[-1].append(j2); already_taken.append(j2);

    newaffine = []; newfaces = [];
    for simp_inds in co_inds:
      newaffine.append(affine[simp_inds[0]]);
      tags = [];
      for inds in simplices[simp_inds]: tags = tags + list(inds); 
      tags = list(set(tags)); tags = [int(tag) for tag in tags];
      _,newtags = orderFacePts3D(newpts[tags],tags);
      # newtags = tags;
      newfaces.append(newtags.copy());
    newaffine = np.array(newaffine);
    # newpts = pts[vertices]
    return newpts, newfaces, newaffine

    
# def genPolytope_fromIntersects(C,d,intersections):
#     face_inds = list(range(len(C)));
#     face_nodes = [[] for _ in face_inds];
#     nodes = [];
#     for ind,inter in enumerate(intersections):
#         pt = mat.inv(C[inter])@d[inter];
#         nodes.append(pt);
#         for j in inter: face_nodes[j].append(ind);
#     nodes = np.array(nodes);
#     for j,_ in enumerate(face_nodes):
#         face_nodes[j],_ = orderFacePts3D(face_nodes[j],nodes[face_nodes[j]])
#     return nodes,face_nodes

