import cv2
import numpy as np
from compute_transoformations import compute_projection
from pnp_algorithms import pnp


def vis_corner_cuboids(img, corners, objpoints3D=[], corners_3d=[]):
    if corners.shape[1] > 9:
        K = np.reshape([572.4114, 0.0, 325.2611, 0.0, 573.57043, 242.04899, 0.0, 0.0, 1.0], (3, 3))
        corners2D_pr = np.array(corners, dtype=float).T

        ################ E-PnP based Pose Estimation
        R_pr, t_pr = pnp(objpoints3D, corners2D_pr, K)
        Rt_pr = np.concatenate((R_pr, t_pr), axis=1)
        corners_3d = np.concatenate((np.transpose(corners_3d), np.ones((1, 9))), axis=0)
        corners = compute_projection(corners_3d, Rt_pr, K)
        ################

        for i in xrange(0, 9):
            x1 = int(corners[0][i])
            y1 = int(corners[1][i])
            cv2.putText(img, str(i), (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1, 16)

        for i in xrange(1, 4):
            x1 = int(corners[0][i])
            y1 = int(corners[1][i])
            x2= int(corners[0][(i+1)])
            y2= int(corners[1][(i+1)])
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 1)

        cv2.line(img, (int(corners[0][1]),int(corners[1][1])),(int(corners[0][(4)]),int(corners[1][(4)])),(255,255,255),1)

        for i in xrange(5, 8):
            x1 = int(corners[0][i])
            y1 = int(corners[1][i])
            x2= int(corners[0][(i+1)])
            y2= int(corners[1][(i+1)])
            cv2.line(img, (x1, y1), (x2, y2),(255,255,255),1)

        cv2.line(img,(int(corners[0][8]),int(corners[1][8])),(int(corners[0][5]),int(corners[1][5])),(255,255,255),1)
        i=5
        cv2.line(img,(int(corners[0][i]),int(corners[1][i])),(int(corners[0][(i-4)]),int(corners[1][(i-4)])),(255,255,255),1)
        i=6
        cv2.line(img,(int(corners[0][i]),int(corners[1][i])),(int(corners[0][(i-4)]),int(corners[1][(i-4)])),(255,255,255),1)
        i=7
        cv2.line(img,(int(corners[0][i]),int(corners[1][i])),(int(corners[0][(i-4)]),int(corners[1][(i-4)])),(255,255,255),1)
        i=8
        cv2.line(img,(int(corners[0][i]),int(corners[1][i])),(int(corners[0][(i-4)]),int(corners[1][(i-4)])),(255,255,255),1)
        return img

    for i in xrange(0, 9):
        x1 = int(corners[0][i])
        y1 = int(corners[1][i])
        cv2.putText(img, str(i), (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1, 16)

    for i in xrange(1, 4):
        x1 = int(corners[0][i])
        y1 = int(corners[1][i])
        x2= int(corners[0][(i+1)])
        y2= int(corners[1][(i+1)])
        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 1)

    cv2.line(img, (int(corners[0][1]),int(corners[1][1])),(int(corners[0][(4)]),int(corners[1][(4)])),(255,255,255),1)

    for i in xrange(5, 8):
        x1 = int(corners[0][i])
        y1 = int(corners[1][i])
        x2= int(corners[0][(i+1)])
        y2= int(corners[1][(i+1)])
        cv2.line(img, (x1, y1), (x2, y2),(255,255,255),1)

    cv2.line(img,(int(corners[0][8]),int(corners[1][8])),(int(corners[0][5]),int(corners[1][5])),(255,255,255),1)
    i=5
    cv2.line(img,(int(corners[0][i]),int(corners[1][i])),(int(corners[0][(i-4)]),int(corners[1][(i-4)])),(255,255,255),1)
    i=6
    cv2.line(img,(int(corners[0][i]),int(corners[1][i])),(int(corners[0][(i-4)]),int(corners[1][(i-4)])),(255,255,255),1)
    i=7
    cv2.line(img,(int(corners[0][i]),int(corners[1][i])),(int(corners[0][(i-4)]),int(corners[1][(i-4)])),(255,255,255),1)
    i=8
    cv2.line(img,(int(corners[0][i]),int(corners[1][i])),(int(corners[0][(i-4)]),int(corners[1][(i-4)])),(255,255,255),1)
    return img



def vis_corner_points(img, corners, objpoints3D, vertices, color=(255,255,255), K=None):
    if K is None:
        K = np.reshape([572.4114, 0.0, 325.2611, 0.0, 573.57043, 242.04899, 0.0, 0.0, 1.0], (3, 3))

    corners2D_pr = np.array(corners, dtype=float).T

    R_pr, t_pr = pnp(objpoints3D, corners2D_pr, K)
    Rt_pr = np.concatenate((R_pr, t_pr), axis=1)
    vertices = np.concatenate((np.transpose(vertices), np.ones((1, vertices.shape[0]))), axis=0)
    vertices2d = compute_projection(vertices, Rt_pr, K)
    vertices2d = vertices2d.astype('int')

    mask_image = np.zeros(img.shape).astype('uint8')

    for i in xrange(0, vertices2d.shape[1]):
        x1 = int(vertices2d[0][i])
        y1 = int(vertices2d[1][i])
        if x1 < 640 and y1 <480 and x1 >= 0 and y1 >= 0:
            cv2.putText(mask_image, '.', (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 16)

    imgray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    cv2.drawContours(img, contours, -1, color, 1)

    return img
