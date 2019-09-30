import numpy as np


def compute_projection(points_3D, transformation, internal_calibration=np.identity(3)):
    projections_2d = np.zeros((2, points_3D.shape[1]), dtype='float32')
    camera_projection = (internal_calibration.dot(transformation)).dot(points_3D)
    projections_2d[0, :] = camera_projection[0, :]/camera_projection[2, :]
    projections_2d[1, :] = camera_projection[1, :]/camera_projection[2, :]
    return projections_2d


def compute_transformation(points_3D, transformation):
    return transformation.dot(points_3D)
