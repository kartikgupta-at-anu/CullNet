import torch
import numpy as np
import math


def rotmat_to_rodrigues(R):
    print 'R: ', R
    theta = torch.acos((torch.trace(R) - 1.0)/2.0)
    # print 'theta: ', theta
    # if theta == 0:
    #     theta = theta + 1e-5
    #     print 'yeah! this is it !'
    w = (theta/(2*torch.sin(theta)))*torch.stack((R[2, 1]-R[1, 2], R[0, 2]-R[2, 0], R[1, 0]-R[0, 1]))
    return w


def rotmat_to_rodrigues_numpy(R):
    c = (np.trace(R) - 1.0)/2.0

    if c > 1.0:
        c = 1.0
    elif c < -1.0:
        c = -1.0

    theta = np.arccos(c)

    w = (theta/(2*np.sin(theta)))*np.stack((R[2, 1]-R[1, 2], R[0, 2]-R[2, 0], R[1, 0] - R[0, 1]))
    return w


def rodrigues_to_rotmat(w):
    theta = w[0]*w[0] + w[1]*w[1] + w[2]*w[2]
    theta = torch.sqrt(theta)
    omega1 = torch.stack((torch.tensor(0, device='cuda', dtype=torch.float), -1*w[2], w[1]))[None]
    omega2 = torch.stack((w[2], torch.tensor(0, device='cuda', dtype=torch.float), -1*w[0]))[None]
    omega3 = torch.stack((-1*w[1], w[0], torch.tensor(0, device='cuda', dtype=torch.float)))[None]
    omega = torch.cat((omega1, omega2, omega3), 0)

    # print "theta: ", theta

    R = torch.eye(3, device='cuda', dtype=torch.float) + (torch.sin(theta)/theta)*omega + ((1 - torch.cos(theta))/(theta*theta))*torch.mm(omega, omega)
    return R


def rodrigues_to_rotmat_numpy(w):
    theta = w[0]*w[0] + w[1]*w[1] + w[2]*w[2]
    theta = np.sqrt(theta)
    omega1 = np.stack((0, -1*w[2], w[1]))[None]
    omega2 = np.stack((w[2], 0, -1*w[0]))[None]
    omega3 = np.stack((-1*w[1], w[0], 0))[None]
    omega = np.concatenate((omega1, omega2, omega3), 0)

    R = np.eye(3) + (np.sin(theta)/theta)*omega + ((1 - np.cos(theta))/(theta*theta))*np.dot(omega, omega)
    return R


def quaternion_matrix(q):
    """Return rotation matrix from quaternion.

    """
    # return torch.eye(3, device='cuda', dtype=torch.float, requires_grad=True)
    epsilon = 1e-5

    n = torch.dot(q, q)
    if n < epsilon:
        return torch.eye(3, device='cuda', dtype=torch.float, requires_grad=True)
    q = torch.sqrt(2.0 / n) * q
    q = torch.ger(q, q)
    rot_mat1 = torch.stack((1.0-q[2, 2]-q[3, 3], q[1, 2]-q[3, 0], q[1, 3]+q[2, 0]))[None]
    rot_mat2 = torch.stack((q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3], q[2, 3]-q[1, 0]))[None]
    rot_mat3 = torch.stack((q[1, 3]-q[2, 0], q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2]))[None]
    rot_mat = torch.cat((rot_mat1, rot_mat2, rot_mat3), 0)

    return rot_mat


def quaternion_matrix_numpy(q):
    """Return rotation matrix from quaternion.

    """
    # return torch.eye(3, device='cuda', dtype=torch.float, requires_grad=True)
    epsilon = 1e-5

    n = np.dot(q, q)
    if n < epsilon:
        return np.eye(3)
    q = np.sqrt(2.0 / n) * q
    q = np.outer(q, q)
    rot_mat1 = np.array([1.0-q[2, 2]-q[3, 3], q[1, 2]-q[3, 0], q[1, 3]+q[2, 0]])[None]
    rot_mat2 = np.array([q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3], q[2, 3]-q[1, 0]])[None]
    rot_mat3 = np.array([q[1, 3]-q[2, 0], q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2]])[None]
    rot_mat = np.concatenate((rot_mat1, rot_mat2, rot_mat3), 0)
    return rot_mat


def quaternion_from_matrix_numpy(matrix, isprecise=False):
    """Return quaternion from rotation matrix.

    If isprecise is True, the input matrix is assumed to be a precise rotation
    matrix and a faster algorithm is used.
    """
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    if isprecise:
        q = np.empty((4, ))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 1, 2, 3
            if M[1, 1] > M[0, 0]:
                i, j, k = 2, 3, 1
            if M[2, 2] > M[i, i]:
                i, j, k = 3, 1, 2
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = np.array([[m00-m11-m22, 0.0,         0.0,         0.0],
                      [m01+m10,     m11-m00-m22, 0.0,         0.0],
                      [m02+m20,     m12+m21,     m22-m00-m11, 0.0],
                      [m21-m12,     m02-m20,     m10-m01,     m00+m11+m22]])
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[[3, 0, 1, 2], np.argmax(w)]
    if q[0] < 0.0:
        np.negative(q, q)
    return q


# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :

    assert(isRotationMatrix(R))

    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])


# Calculates Rotation Matrix given euler angles.
def eulerAnglesToRotationMatrix(theta) :

    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])


    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])

    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])


    R = np.dot(R_z, np.dot( R_y, R_x ))

    return R