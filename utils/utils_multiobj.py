import cv2
import os
import numpy as np
import trimesh


def load_gt_pose_brachmann(path):
    R = []
    t = []
    rotation_sec = False
    center_sec = False
    with open(path, 'r') as f:
        for line in f.read().splitlines():
            if 'rotation:' in line:
                rotation_sec = True
            elif rotation_sec:
                R += line.split(' ')
                if len(R) == 9:
                    rotation_sec = False
            elif 'center:' in line:
                center_sec = True
            elif center_sec:
                t = line.split(' ')
                center_sec = False

    assert((len(R) == 0 and len(t) == 0) or
           (len(R) == 9 and len(t) == 3))

    if len(R) == 0:
        pose = {'R': np.array([]), 't': np.array([])}
    else:
        pose = {'R': np.array(map(float, R)).reshape((3, 3)),
                't': np.array(map(float, t)).reshape((3, 1))}

        # Flip Y and Z axis (OpenGL -> OpenCV coordinate system)
        yz_flip = np.eye(3, dtype=np.float32)
        yz_flip[0, 0], yz_flip[1, 1], yz_flip[2, 2] = 1, -1, -1
        pose['R'] = yz_flip.dot(pose['R'])
        pose['t'] = yz_flip.dot(pose['t'])
    return pose


def convert_to_occlusionfmt(corners_3d, corners_cuboid, vertices):
    corners_3d = corners_3d - corners_cuboid[0]
    vertices_transformed = vertices - corners_cuboid[0]
    corners_cuboid = corners_cuboid - corners_cuboid[0]
    corners_3d_transformed = corners_3d
    corners_cuboid_transformed = corners_cuboid
    return corners_3d_transformed, corners_cuboid_transformed, vertices_transformed


def threed_correspondences(cls, dir_path):
    return np.load(dir_path + '/' + cls + '_3dcorres.npy')


def threed_corners(cls):
    mesh_base = '../data/LINEMOD_original/models'

    mesh_name = os.path.join(mesh_base, cls + '.ply')
    mesh = trimesh.load(mesh_name)
    corners = trimesh.bounds.corners(mesh.bounding_box.bounds)
    corners = np.insert(corners, 0, np.average(corners, axis=0), axis=0)

    return corners


def threed_vertices(cls):
    mesh_base = '../data/LINEMOD_original/models'

    mesh_name = os.path.join(mesh_base, cls + '.ply')
    mesh = trimesh.load(mesh_name)
    vertices = mesh.vertices

    return vertices


def compute_projection(points_3D, transformation, internal_calibration):
    projections_2d = np.zeros((2, points_3D.shape[1]), dtype='float32')
    camera_projection = (internal_calibration.dot(transformation)).dot(points_3D)
    projections_2d[0, :] = camera_projection[0, :]/camera_projection[2, :]
    projections_2d[1, :] = camera_projection[1, :]/camera_projection[2, :]
    return projections_2d


def compute_transformation(points_3D, transformation):
    return transformation.dot(points_3D)


def pnp(points_3D, points_2D, cameraMatrix):
    try:
        distCoeffs = pnp.distCoeffs
    except:
        distCoeffs = np.zeros((8, 1), dtype='float32')

    assert points_2D.shape[0] == points_2D.shape[0], 'points 3D and points 2D must have same number of vertices'

    _, R_exp, t = cv2.solvePnP(points_3D,
                               # points_2D,
                               np.ascontiguousarray(points_2D[:, :2]).reshape((-1, 1, 2)),
                               cameraMatrix,
                               distCoeffs)
    R, _ = cv2.Rodrigues(R_exp)
    return R, t


def corners_transform(corners, cam_k, cam_rt):
    cam_p = np.matmul(cam_k, cam_rt)
    corners_homo = np.concatenate((corners, np.ones((corners.shape[0], 1))), axis=1)
    corners_transformed_homo = np.matmul(cam_p, corners_homo.T)
    corners_transformed = np.zeros((2, corners.shape[0]))
    corners_transformed[0, :] = corners_transformed_homo[0, :]/corners_transformed_homo[2, :]
    corners_transformed[1, :] = corners_transformed_homo[1, :]/corners_transformed_homo[2, :]
    return corners_transformed


def create_annotations(aug_annotbase, file_id, corners):
    annotation = []

    for cls in corners:
        gt_instance = list(np.array(np.reshape(corners[cls], (9 * 2), 'F'), dtype=int))
        gt_instance.append(cls)
        annotation.append(gt_instance)
    annotation = np.array(annotation, dtype=str, ndmin=2)
    aug_annotname = os.path.join(aug_annotbase, file_id+'.txt')
    np.savetxt(aug_annotname, annotation, fmt='%s')


def overlap_iou(mask_obj, mask_complete):
    n_ii = np.sum(np.logical_and(mask_obj, mask_complete))
    t_i = np.sum(mask_obj)
    return  n_ii / (t_i)
