import math, os
import numpy as np
import trimesh
import yaml


def calc_pts_diameter(pts):
    diameter = -1
    for pt_id in range(pts.shape[0]):
        pt_dup = np.tile(np.array([pts[pt_id, :]]), [pts.shape[0] - pt_id, 1])
        pts_diff = pt_dup - pts[pt_id:, :]
        max_dist = math.sqrt((pts_diff * pts_diff).sum(axis=1).max())
        if max_dist > diameter:
            diameter = max_dist
    return diameter


label_names = ('ape', 'benchvise', 'cam', 'can', 'cat', 'driller', 'duck', 'eggbox', 'glue',
               'holepuncher', 'iron', 'lamp', 'phone')

data = {}

for cls in label_names:
    mesh_base = 'data/LINEMOD_original/models'
    mesh_name = os.path.join(mesh_base, cls + '.ply')
    mesh = trimesh.load(mesh_name)
    vertices = mesh.vertices
    diam = calc_pts_diameter(vertices)
    data[cls] = diam
    print cls, diam

with open('cfgs/diameter_linemod.yml', 'w') as outfile:
    yaml.dump(data, outfile, default_flow_style=False)


