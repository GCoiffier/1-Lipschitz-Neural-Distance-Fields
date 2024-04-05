import mouette as M
import sys
from igl import signed_distance
import numpy as np

GTmesh = M.mesh.load(sys.argv[1])
querymesh = M.mesh.load(sys.argv[2])
name = M.utils.get_filename(sys.argv[2])

GT,_,_ = signed_distance(np.array(querymesh.vertices), np.array(GTmesh.vertices), np.array(GTmesh.faces, dtype=np.int32))
GT = abs(GT)
wnattr = querymesh.vertices.register_array_as_attribute("dist", GT)
print(np.max(GT))
uvs = M.attributes.generate_uv_colormap_vertices(querymesh, wnattr, 0, 0.03)
M.mesh.save(querymesh, f"{name}_with_dist.geogram_ascii")
M.mesh.save(querymesh, f"{name}_with_dist.obj")