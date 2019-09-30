import cv2
import numpy as np
from numpy import matlib as mb


def normalization(nd, x, cam_K):
    '''
    Normalization of coordinates (centroid to the origin and mean distance of sqrt(2 or 3).

    Inputs:
     nd: number of dimensions (2 for 2D; 3 for 3D)
     x: the data to be normalized (directions at different columns and points at rows)
    Outputs:
     Tr: the transformation matrix (translation plus scaling)
     x: the transformed data
    '''
    x = np.asarray(x)
    m, s = np.mean(x, 0), np.std(x)

    if nd==2:
        # Tr = np.array([[s, 0, m[0]], [0, s, m[1]], [0, 0, 1]])
        Tr = cam_K
    else:
        Tr = np.array([[s, 0, 0, m[0]], [0, s, 0, m[1]], [0, 0, s, m[2]], [0, 0, 0, 1]])
        # Tr = np.identity(4)

    Tr = np.linalg.inv(Tr)
    x = np.dot(Tr, np.concatenate((x.T, np.ones((1, x.shape[0])))))
    x = x[0:nd, :].T
    return Tr, x


def calcampose(XXc, XXw):
    # kps, 3
    n = np.shape(XXc)[0]
    K = np.eye(n) - np.ones((n, n))/n

    uw = np.mean(XXw, axis=0)
    uc = np.mean(XXc, axis=0)

    sigmx2 = np.square(np.matmul(XXw.T, K))
    sigmx2 = np.mean(np.sum(sigmx2, axis=0))

    SXY = np.matmul(np.matmul(XXc.T, K), XXw)/n
    U, D, V_t = np.linalg.svd(SXY)
    S = np.eye(3)
    #     if np.linalg.det(SXY) < 0:
    #         S[2,2]=-1
    if np.linalg.det(np.matmul(U, V_t)) < 0:
        S[2,2] = -1
    R2 = np.matmul(U, np.matmul(S,V_t))
    D = np.diag(D)
    c2 = np.trace(np.matmul(D,S))/sigmx2
    t2 = uc - c2 * np.matmul(R2, uw)

    X = R2[:, 0]
    Y = R2[:, 1]
    Z = R2[:, 2]
    if np.linalg.norm(np.cross(X, Y) - Z) > 2e-2:
        R2[:, 2] = -Z
    return R2, t2


## normalized 3d dlt + non normalized 3d procrustes
def DLT(XXw_old, xx, cam_K, mask=None):
    '''
    DLT from EigFree code which does Procustes algo based refinement for Pose to make a valid rotation matrix.

    Inputs:
     XXw: 3d points (kps * 3)
     xx: 2d points in image (kps * 2)
     cam_K: camera intrinsic matrix (3X3)
     mask: weights for weighted DLT (kps)
    Outputs:
     R,t : Estimated Pose
    '''

    Tuv, xx = normalization(2, xx, cam_K)
    Txyz, XXw = normalization(3, XXw_old, cam_K)
    x = np.expand_dims(XXw[:, 0], axis=-1)
    y = np.expand_dims(XXw[:, 1], axis=-1)
    z = np.expand_dims(XXw[:, 2], axis=-1)
    u = np.expand_dims(xx[:, 0], axis=-1)
    v = np.expand_dims(xx[:, 1], axis=-1)

    ones = np.ones(np.shape(x))
    zeros = np.zeros(np.shape(x))
    M1n = np.concatenate([x, y, z, ones, zeros, zeros, zeros, zeros, \
                          -u*x, -u*y, -u*z, -u], axis=1)
    M2n = np.concatenate([zeros, zeros, zeros, zeros, x, y, z, ones, \
                          -v*x, -v*y, -v*z, -v], axis=1)
    M = np.concatenate([M1n, M2n], axis=0) # kps, 12
    M_t = np.transpose(M)
    if mask is not None:
        mask = np.ceil(mask)
        if len(np.shape(mask))is not 2:
            mask = np.expand_dims(mask, axis=-1)
        w2n = np.concatenate([mask, mask], axis=0)
        wM = w2n * M
        MwM = np.matmul(M_t, wM)
    else:
        MwM = np.matmul(M_t, M)
    e, V = np.linalg.eigh(MwM)
    v = np.reshape(V[:, 0], (3, 4))
    v = np.dot(np.reshape(v, (3, 4)), Txyz)
    v /= np.linalg.norm(v[2, :3])
    v *= np.sign(v[2, 3])
    R = v[:, :3]
    t = np.expand_dims(v[:, 3], axis=-1)

    XXc = np.matmul(R, XXw_old.T) + mb.repmat(t, 1, np.shape(XXw_old)[0])
    R, t = calcampose(XXc.T, XXw_old)
    return R, t[:, None]


def pnp(points_3D, points_2D, cameraMatrix):
    distCoeffs = np.zeros((8, 1), dtype='float32')

    assert points_2D.shape[0] == points_2D.shape[0], 'points 3D and points 2D must have same number of vertices'

    _, R_exp, t = cv2.solvePnP(points_3D,
                               # points_2D,
                               np.ascontiguousarray(points_2D[:, :2]).reshape((-1, 1, 2)),
                               cameraMatrix,
                               distCoeffs, flags=cv2.SOLVEPNP_EPNP)

    R, _ = cv2.Rodrigues(R_exp)
    # Rt = np.c_[R, t]
    return R, t


def pnp_iterative(points_3D, points_2D, cameraMatrix):
    try:
        distCoeffs = pnp.distCoeffs
    except:
        distCoeffs = np.zeros((8, 1), dtype='float32')

    assert points_2D.shape[0] == points_2D.shape[0], 'points 3D and points 2D must have same number of vertices'

    _, R_exp, t = cv2.solvePnP(points_3D,
                               # points_2D,
                               np.ascontiguousarray(points_2D[:, :2]).reshape((-1, 1, 2)),
                               cameraMatrix,
                               distCoeffs, flags=cv2.SOLVEPNP_EPNP)

    _, R_exp, t = cv2.solvePnP(points_3D,
                               # points_2D,
                               np.ascontiguousarray(points_2D[:, :2]).reshape((-1, 1, 2)),
                               cameraMatrix,
                               distCoeffs, rvec=R_exp, tvec=t, flags=cv2.SOLVEPNP_ITERATIVE)

    R, _ = cv2.Rodrigues(R_exp)
    # Rt = np.c_[R, t]
    return R, t


def pnp_ransac(points_3D, points_2D, cameraMatrix):
    try:
        distCoeffs = pnp.distCoeffs
    except:
        distCoeffs = np.zeros((8, 1), dtype='float32')

    assert points_2D.shape[0] == points_2D.shape[0], 'points 3D and points 2D must have same number of vertices'

    _, R_exp, t, inliers = cv2.solvePnPRansac(points_3D,
                                              # points_2D,
                                              np.ascontiguousarray(points_2D[:, :2], dtype='float64').reshape((-1, 1, 2)),
                                              cameraMatrix,
                                              None, flags=cv2.SOLVEPNP_EPNP)

    R, _ = cv2.Rodrigues(R_exp)
    # Rt = np.c_[R, t]
    return R, t
