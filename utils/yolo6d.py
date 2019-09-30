import cv2, os
import numpy as np
import trimesh, math, pickle
import cfgs.config_yolo6d as cfg
from random import randint
from pnp_algorithms import *
from load_meshinfo import *
from visualisation_functions import *
from preprocess_fns import *
from postprocess_fns import *
from proposals_cullnet import *


def refine_2dboxes(dets, corners):
    """refined_dets = refine_2dboxes(dets, corners)

    Refines the 2d projections by using the pose calculated by PnP algorithm with model mesh corners and
    detected 2d projections of the corner points.

    dets: num_detections * cfg.args.num_detection_points * 2
    corners: cfg.args.num_detection_points  * 3

    refined_dets: num_detections * cfg.args.num_detection_points * 2
    """
    cam_K = np.reshape([572.4114, 0.0, 325.2611, 0.0, 573.57043, 242.04899, 0.0, 0.0, 1.0], (3, 3))

    refined_dets = np.zeros(dets.shape)
    dets = np.reshape(dets, [dets.shape[0], cfg.args.num_detection_points, 2])
    dets = dets.astype(float)

    for det_ind in range(dets.shape[0]):
        ret, rvec, tvec = cv2.solvePnP(np.reshape(corners, (cfg.args.num_detection_points, 1, 3)), np.reshape(dets[det_ind],(cfg.args.num_detection_points,1,2)),
                                       cam_K, distCoeffs=None, flags=cv2.SOLVEPNP_EPNP)
        verts = cv2.projectPoints(corners, rvec, tvec, cam_K, distCoeffs=None)
        refined_dets[det_ind] = verts[0].reshape((1, cfg.args.num_detection_points * 2))
    return refined_dets


def confidence_function(cuboid_pred,cuboid_gt):
    alpha = 2
    d_thresh = 30
    distance_pred = np.subtract(cuboid_pred, cuboid_gt)
    distance_pred = np.absolute(distance_pred)
    conf = np.where(distance_pred<d_thresh, np.exp(alpha*(-(distance_pred/d_thresh))), 0)
    conf = np.average(conf)
    return conf


def corner_confidence9(pr_corners, gt_corners, th=15, sharpness=2):
    ''' gt_corners: Ground-truth 2D projections of the 3d point on surface, shape: (2* num_detection_points,) type: list
        pr_corners: Prediction for the 2D projections of the 3d point on surface, shape: (2* num_detection_points,), type: list
        visibility_indices: mask for visibiility of each 2d projection whether it is visible or not in the image, shape: (num_detection_points,), type: list
        th        : distance threshold, type: int
        sharpness : sharpness of the exponential that assigns a confidence value to the distance
        -----------
        return    : a confidence value for the prediction
    '''
    dist = gt_corners - pr_corners
    dist = np.reshape(dist, ((cfg.args.num_detection_points), 2))
    eps = 1e-5
    dist = np.sqrt(np.sum((dist)**2, axis=1))
    mask = (dist < th)
    conf = np.exp(sharpness * (1 - dist.astype(int)/th)) - 1
    conf0 = np.exp(sharpness) - 1 + eps
    conf = conf / conf0.repeat(cfg.args.num_detection_points)
    # conf = 1.0 - dist/th
    # visibility_indices = np.nonzero(visibility_indices)
    conf = mask * conf
    return np.mean(conf)


def nms_detections(dets, scores, thresh):
    order = scores.argsort()[::-1]
    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=np.int)

    keep = []
    for _i in range(ndets):
        i = order[_i]
        if suppressed[i] == 1:
            continue
        keep.append(i)
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            ovr = corner_confidence9(dets[i, :], dets[j, :])
            if ovr >= thresh:
                suppressed[j] = 1

    return keep


def threecrossthree_suppression(dets, scores, cell_inds, out_size, conf):
    max_ind = scores.argmax()
    W, H = out_size
    cell_inds_map = dict(zip(cell_inds, np.arange(len(cell_inds))))
    keep = []

    i = max_ind
    idx = cell_inds[i]
    keep.append(i)
    weighted_average_numer = scores[i] * dets[i, :]
    weighted_average_denom = scores[i]
    neighbourhood_global = [idx-W-1, idx-W, idx-W+1, idx-1, idx+1, idx+W-1, idx+W, idx+W+1]

    for neighbourhood_idx_global in neighbourhood_global:
        if cell_inds_map.has_key(neighbourhood_idx_global):
            if conf[cell_inds_map.get(neighbourhood_idx_global)] >= 0.5:
                weighted_average_numer += scores[cell_inds_map.get(neighbourhood_idx_global)] * \
                                          dets[cell_inds_map.get(neighbourhood_idx_global),:]
                weighted_average_denom += scores[cell_inds_map.get(neighbourhood_idx_global)]
    weighted_average = weighted_average_numer / weighted_average_denom

    pruned_dets = weighted_average
    pruned_dets = np.array(pruned_dets)

    return pruned_dets, keep


def residual_suppression(dets, scores, cell_inds, confs, objpoints3D, thresh):
    res_conf = []
    keep = np.where(scores >= thresh)
    dets = dets[keep]
    scores = scores[keep]
    confs = confs[keep]
    cell_inds = cell_inds[keep]

    for det in dets:
        det = np.reshape(det, ((cfg.args.num_detection_points), 2))
        Rt_pr = DLT(objpoints3D, det, cfg.cam_K)
        projected_correspondences = compute_projection(np.c_[np.array(objpoints3D), np.ones((len(objpoints3D), 1))].transpose(), Rt_pr, cfg.cam_K)
        res_conf.append(corner_confidence9(projected_correspondences.T, det))

    scores = confs * res_conf
    keep = keep[0][np.argmax(scores)]
    return keep


def residual_analysis(dets, gt_box, scores, cell_inds, confs, objpoints3D, vertices, thresh):
    res_conf = []
    residual_mean = []
    residual_max = []
    residual_median = []

    keep = np.where(scores >= thresh)
    dets = dets[keep]
    scores = scores[keep]
    confs = confs[keep]
    cell_inds = cell_inds[keep]
    correct = []

    vertices = np.c_[np.array(vertices), np.ones((len(vertices), 1))].transpose()
    corners2D_gt_corrected = np.reshape(gt_box, ((cfg.args.num_detection_points), 2))
    R_gt, t_gt = pnp(objpoints3D, corners2D_gt_corrected, cfg.cam_K)
    Rt_gt = np.concatenate((R_gt, t_gt), axis=1)
    proj_2d_gt = compute_projection(vertices, Rt_gt, cfg.cam_K)
    gt_cellind = math.floor((corners2D_gt_corrected[0][0]/640.0) * 20) + math.floor((corners2D_gt_corrected[0][1]/640.0) * 20) * 20
    gt_cell_residual = float((corners2D_gt_corrected[0][0]/640.0) * 20- math.floor((corners2D_gt_corrected[0][0]/640.0) * 20)), float(((corners2D_gt_corrected[0][1]/640.0) * 20)- math.floor((corners2D_gt_corrected[0][1]/640.0) * 20))
    keep = keep[0]
    for det in dets:
        det = np.reshape(det, ((cfg.args.num_detection_points), 2))
        Rt_pr = DLT(objpoints3D, det, cfg.cam_K)
        projected_correspondences = compute_projection(np.c_[np.array(objpoints3D), np.ones((len(objpoints3D), 1))].transpose(), Rt_pr, cfg.cam_K)
        res_conf.append(corner_confidence9(projected_correspondences.T, det))
        residual_mean.append(np.mean(np.sqrt(np.sum((projected_correspondences.T - det)**2, axis=1))))
        residual_max.append(np.max(np.sqrt(np.sum((projected_correspondences.T - det)**2, axis=1))))
        residual_median.append(np.median(np.sqrt(np.sum((projected_correspondences.T - det)**2, axis=1))))
        corners2D_pr = det
        Rt_pr = DLT(objpoints3D, corners2D_pr, cfg.cam_K)
        proj_2d_pred = compute_projection(vertices, Rt_pr, cfg.cam_K)
        norm = np.linalg.norm(proj_2d_gt - proj_2d_pred, axis=0)
        pixel_dist = np.mean(norm)
        if pixel_dist <= 5:
            correct.append(1)
        else:
            correct.append(0)
    res_conf = np.array(res_conf)
    correct = np.nonzero(correct)[0]
    if not keep[np.argmax(confs)] in keep[correct]:
        print 'correct:',  keep[correct], 'gt_cell:', gt_cellind, gt_cell_residual, 'confs:', keep[np.argmax(confs)], 'scores:', keep[np.argmax(scores)], 'res_conf:', keep[np.argmax(res_conf)], 'res_conf_emperics:', keep[np.nonzero(confs > 0.5)[0][np.argmax(res_conf[confs > 0.5])]], 'resconf *scores:', keep[np.argmax(res_conf * scores)], 'residual_mean:', keep[np.argmin(residual_mean)], 'residual_median:', keep[np.argmin(residual_median)], 'residual_max:', keep[np.argmin(residual_max)]
        print 'Note this'
    return keep


def draw_detection(im, bboxes, scores, cls_inds, cfg, labels, thr=0.5, objpoints3D=[], corners_3d=[], vertices=[], gt_RT=[]):
    # draw image
    colors = cfg.colors

    imgcv = np.copy(im)
    h, w, _ = imgcv.shape
    for i, box in enumerate(bboxes):
        if scores[i] < thr:
            continue
        cls_indx = cls_inds[i]

        thick = int((h + w) / 300)

        if cfg.args.num_detection_points == 9:
            vis_corner_points(imgcv, np.reshape(box, (cfg.args.num_detection_points, 2)).transpose(), objpoints3D, vertices)
            # vis_corner_cuboids(imgcv, np.reshape(box, (9, 2)).transpose())
        else:
            vis_corner_points(imgcv, np.reshape(box, (cfg.args.num_detection_points, 2)).transpose(), objpoints3D, vertices)
            # vis_corner_cuboids(imgcv, np.reshape(box, (cfg.args.num_detection_points, 2)).transpose(), objpoints3D, corners_3d)

        # cv2.rectangle(imgcv,
        #               (box[0], box[1]), (box[2], box[3]),
        #               colors[cls_indx], thick)
        mess = '%s: %.3f' % (labels[cls_indx], scores[i])
        cv2.putText(imgcv, mess, (int(box[0]), int(box[1]) - 12),
                    0, 1e-3 * h, colors[cls_indx], thick // 3)

    return imgcv


# def draw_detection_multiobj(im, bboxes, scores, cls_inds, cfg, labels, thr=0.5):
#     # draw image
#     colors = cfg.colors
#
#     imgcv = np.copy(im)
#     h, w, _ = imgcv.shape
#     for i, box in enumerate(bboxes):
#         if scores[i] < thr:
#             continue
#         cls_indx = cls_inds[i]
#
#         thick = int((h + w) / 300)
#
#         if cfg.args.num_detection_points == 9:
#             # vis_corner_points(imgcv, np.reshape(box, (cfg.args.num_detection_points, 2)).transpose(), objpoints3D, vertices)
#             vis_corner_cuboids(imgcv, np.reshape(box, (9, 2)).transpose())
#         else:
#             vis_corner_points(imgcv, np.reshape(box, (cfg.args.num_detection_points, 2)).transpose(), objpoints3D, vertices)
#             # vis_corner_cuboids(imgcv, np.reshape(box, (cfg.args.num_detection_points, 2)).transpose(), objpoints3D, corners_3d)
#
#         # cv2.rectangle(imgcv,
#         #               (box[0], box[1]), (box[2], box[3]),
#         #               colors[cls_indx], thick)
#         mess = '%s: %.3f' % (labels[cls_indx], scores[i])
#         cv2.putText(imgcv, mess, (int(box[0]), int(box[1]) - 12),
#                     0, 1e-3 * h, colors[cls_indx], thick // 3)
#
#     return imgcv


# def draw_detection_debug(im, bboxes, scores, cls_inds, cfg, labels, thr=0.5, objpoints3D=[], corners_3d=[], vertices=[], gt_RT=[]):
#     # draw image
#     colors = cfg.colors
#
#     imgcv = np.copy(im)
#     h, w, _ = imgcv.shape
#
#     thick = int((h + w) / 300)
#
#     if gt_RT.size != 0:
#         minimum_error = 10000
#         Rt_gt = np.reshape(gt_RT, (3, 4))
#         vertices_append = np.concatenate((np.transpose(vertices), np.ones((1, vertices.shape[0]))), axis=0)
#         for i, box in enumerate(bboxes):
#             corners2D_pr = np.reshape(box, ((cfg.args.num_detection_points), 2))
#             R_pr, t_pr = pnp(objpoints3D, corners2D_pr, cfg.cam_K)
#             Rt_pr = np.concatenate((R_pr, t_pr), axis=1)
#             proj_2d_gt = compute_projection(vertices_append, Rt_gt, cfg.cam_K)
#             proj_2d_pred = compute_projection(vertices_append, Rt_pr, cfg.cam_K)
#             norm = np.linalg.norm(proj_2d_gt - proj_2d_pred, axis=0)
#             pixel_dist = np.mean(norm)
#             if pixel_dist < minimum_error:
#                 correct_idx = i
#                 minimum_error = pixel_dist
#
#         box = bboxes[correct_idx]
#         cls_indx = cls_inds[correct_idx]
#
#         if cfg.args.num_detection_points == 9:
#             vis_corner_points(imgcv, np.reshape(box, (cfg.args.num_detection_points, 2)).transpose(), objpoints3D, vertices, color=(0,0,0))
#             # vis_corner_cuboids(imgcv, np.reshape(box, (9, 2)).transpose())
#         else:
#             vis_corner_points(imgcv, np.reshape(box, (cfg.args.num_detection_points, 2)).transpose(), objpoints3D, vertices, color=(0,0,0))
#             # vis_corner_cuboids(imgcv, np.reshape(box, (cfg.args.num_detection_points, 2)).transpose(), objpoints3D, corners_3d)
#
#         # cv2.rectangle(imgcv,
#         #               (box[0], box[1]), (box[2], box[3]),
#         #               colors[cls_indx], thick)
#         mess = '%s: %.3f' % (labels[cls_indx], scores[correct_idx])
#         cv2.putText(imgcv, mess, (int(box[0]), int(box[1]) - 12),
#                     0, 1e-3 * h, (0,0,0), thick // 3)
#
#     box = bboxes[np.argmax(scores)]
#     cls_indx = cls_inds[np.argmax(scores)]
#
#     if cfg.args.num_detection_points == 9:
#         vis_corner_points(imgcv, np.reshape(box, (cfg.args.num_detection_points, 2)).transpose(), objpoints3D, vertices)
#         # vis_corner_cuboids(imgcv, np.reshape(box, (9, 2)).transpose())
#     else:
#         vis_corner_points(imgcv, np.reshape(box, (cfg.args.num_detection_points, 2)).transpose(), objpoints3D, vertices)
#         # vis_corner_cuboids(imgcv, np.reshape(box, (cfg.args.num_detection_points, 2)).transpose(), objpoints3D, corners_3d)
#
#     # cv2.rectangle(imgcv,
#     #               (box[0], box[1]), (box[2], box[3]),
#     #               colors[cls_indx], thick)
#     mess = '%s: %.3f' % (labels[cls_indx], scores[np.argmax(scores)])
#     cv2.putText(imgcv, mess, (int(box[0]), int(box[1]) - 12),
#                 0, 1e-3 * h, colors[cls_indx], thick // 3)
#
#     return imgcv