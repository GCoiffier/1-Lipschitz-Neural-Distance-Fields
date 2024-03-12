import mouette as M
import sys
from igl import fast_winding_number_for_meshes, fast_winding_number_for_points
import numpy as np
from sklearn.neighbors import kneighbors_graph
import triangle
from tqdm import trange

N_SAMPLING = 50_000 # number of samples in the surface point cloud

def point_cloud_from_array(X, D=None):
    pc = M.mesh.PointCloud()
    if X.shape[1]==2:
        X = np.pad(X, ((0,0),(0,1)))
    pc.vertices += list(X)
    if D is not None:
        attr = pc.vertices.create_attribute("dist", float, dense=True)
        attr._data = D.reshape((D.shape[0], 1))
    return pc

def estimate_vertex_areas(V,N, k=20):
    n_pts = V.shape[0]
    KNN_mat = kneighbors_graph(V,k,mode="connectivity")
    KNN = [[] for _ in range(n_pts)]
    rows, cols = KNN_mat.nonzero()
    for r,c in zip(rows,cols):
        KNN[r].append(c)
    KNN = np.array(KNN)
    A = np.zeros(n_pts)
    for i in trange(n_pts):
        ni = M.Vec.normalized(N[i])
        Xi,Yi,Zi = (M.geometry.cross(basis, ni) for basis in (M.Vec.X(), M.Vec.Y(), M.Vec.Z()))
        Xi = [_X for _X in (Xi,Yi,Zi) if M.geometry.norm(_X)>1e-8][0]
        Xi = M.Vec.normalized(Xi)
        Yi = M.geometry.cross(ni,Xi)

        neighbors = [V[j] for j in KNN[i,:]] # coordinates of k neighbors
        neighbors = [M.geometry.project_to_plane(_X,N[i],V[i]) for _X in neighbors] # Project onto normal plane
        neighbors = [M.Vec(Xi.dot(_X), Yi.dot(_X)) for _X in neighbors]

        for (p1,p2,p3) in triangle.triangulate({"vertices" : neighbors})["triangles"]:
            A[i] += M.geometry.triangle_area_2D(neighbors[p1], neighbors[p2], neighbors[p3])
    return A

mesh = M.mesh.load(sys.argv[1])
sampling_surf = M.mesh.load(sys.argv[3])
WN_mesh = fast_winding_number_for_meshes(np.array(mesh.vertices), np.array(mesh.faces, dtype=np.int32), np.array(sampling_surf.vertices))
wnattr = sampling_surf.vertices.register_array_as_attribute("wn", WN_mesh)
uvs = M.attributes.generate_uv_colormap_vertices(sampling_surf, wnattr, 0,1)
M.mesh.save(sampling_surf, "wn_surf.geogram_ascii")
M.mesh.save(sampling_surf, "wn_surf.obj")

pc = M.mesh.load(sys.argv[2])
sampling_surf = M.mesh.load(sys.argv[3])
V = np.array(pc.vertices)
N = pc.vertices.get_attribute("normals").as_array(len(pc.vertices))
A = pc.vertices.get_attribute("area").as_array(len(pc.vertices))
WN_pts = fast_winding_number_for_points(V,N,A, np.array(sampling_surf.vertices))
wnattr = sampling_surf.vertices.register_array_as_attribute("wn", WN_pts)
uvs = M.attributes.generate_uv_colormap_vertices(sampling_surf, wnattr, 0,1)
M.mesh.save(sampling_surf, "wn_pc.geogram_ascii")
M.mesh.save(sampling_surf, "wn_pc.obj")
