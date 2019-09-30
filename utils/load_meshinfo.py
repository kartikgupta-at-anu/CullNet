import numpy as np
import os
import cfgs.config_yolo6d as cfg
import trimesh


def threed_correspondences(cls):
    return np.load(os.path.join(cfg.DATA_DIR, cfg.args.datadirname) + '/' + cls + '_3dcorres.npy')


def threed_corners(cls):
    if cfg.args.dataset_name == 'LINEMOD':
        mesh_base = cfg.DATA_DIR + '/LINEMOD_original/models'
        mesh_name = os.path.join(mesh_base, cls + '.ply')
        mesh = trimesh.load(mesh_name)
        corners = trimesh.bounds.corners(mesh.bounding_box.bounds)
        corners = np.insert(corners, 0, np.average(corners, axis=0), axis=0)
    return corners


def threed_vertices(cls, eval=False):
    if cfg.args.dataset_name == 'LINEMOD':
        mesh_base = cfg.DATA_DIR + '/LINEMOD_original/models'
        mesh_name = os.path.join(mesh_base, cls + '.ply')
    mesh = trimesh.load(mesh_name)

    if not eval:
        mesh = mesh.subdivide()
    vertices = mesh.vertices

    return vertices
