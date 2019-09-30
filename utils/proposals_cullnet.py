import cv2, os
import numpy as np
import cfgs.config_yolo6d as cfg
from random import randint
from pnp_algorithms import pnp
from compute_transoformations import compute_projection, compute_transformation


def pose_proposals(objpoints3D, vertices, cam_K, obj_diameter, data):
    bboxes_batch, scores_batch, cls_inds_batch, gt_boxes_batch, gtimage_batch = data

    vertices = np.c_[np.array(vertices), np.ones((len(vertices), 1))].transpose()

    num_classes = cfg.num_classes

    d_th_2d = 10.0
    d_th_3d = 2 * 0.1 * obj_diameter

    bbox_pred = np.reshape(bboxes_batch, [-1, (2 * cfg.args.num_detection_points)])
    scores = np.reshape(scores_batch, [-1])
    cls_inds = np.reshape(cls_inds_batch, [-1])

    keep = np.zeros(len(bbox_pred), dtype=np.int)

    if scores.shape[0] < cfg.args.topk:
        scores = np.tile(scores, cfg.args.topk)
        cls_inds = np.tile(cls_inds, cfg.args.topk)
        bbox_pred = np.tile(bboxes_batch, [cfg.args.topk, 1])
        keep = np.zeros(len(bbox_pred), dtype=np.int)

    inds = np.where(cls_inds == 0)[0]
    c_scores = scores[inds]
    c_keep = np.argpartition(c_scores, -16)[-16:]
    keep[inds[c_keep]] = 1

    keep = np.where(keep > 0)
    bbox_pred = bbox_pred[keep]
    # scores = scores[keep]

    R_gt, t_gt = pnp(objpoints3D, np.reshape(gt_boxes_batch, ((cfg.args.num_detection_points), 2)), cam_K)
    Rt_gt = np.concatenate((R_gt, t_gt), axis=1)
    proj_2d_gt = compute_projection(vertices, Rt_gt, cam_K)

    transform_3d_gt = compute_transformation(vertices, Rt_gt)

    Rt_pr_patches = np.zeros([17, 3, 4]).astype(np.float32)
    gtconf2d_patches = np.zeros(17).astype(np.float32)
    gtconf3d_patches = np.zeros(17).astype(np.float32)
    corner_patches = np.zeros([17, 4]).astype(np.int)
    bboxes_patches = np.zeros([17, (2 * cfg.args.num_detection_points)]).astype(np.float32)

    for i in xrange(16):
        R_pr, t_pr = pnp(objpoints3D, np.reshape(bbox_pred[i], ((cfg.args.num_detection_points), 2)), cam_K)
        Rt_pr = np.concatenate((R_pr, t_pr), axis=1)
        proj_2d_pred = compute_projection(vertices, Rt_pr, cam_K)
        transform_3d_pred = compute_transformation(vertices, Rt_pr)

        min_x, min_y, max_x, max_y = int(np.min(proj_2d_pred[0])), int(np.min(proj_2d_pred[1])), \
                                     int(np.max(proj_2d_pred[0])), \
                                     int(np.max(proj_2d_pred[1]))

        corner_patches[i] = [min_x, min_y, max_x, max_y]
        Rt_pr_patches[i] = Rt_pr

        bboxes_patches[i] = bbox_pred[i]

        norm2d = np.linalg.norm(proj_2d_gt - proj_2d_pred, axis=0)
        norm3d = np.linalg.norm(transform_3d_gt - transform_3d_pred, axis=0)

        pixel_dist = np.mean(norm2d)
        vertex_dist = np.mean(norm3d)

        conf = max(np.exp(2.0 * (1.0 - (pixel_dist/(2*d_th_2d)))) - 1, 0)
        conf0 = np.exp(([2.0])) - 1
        gtconf2d_patches[i] = conf/conf0

        conf = max(np.exp(2.0 * (1.0 - (vertex_dist/(2*d_th_3d)))) - 1, 0)
        conf0 = np.exp(([2.0])) - 1
        gtconf3d_patches[i] = conf/conf0

    i = 16
    Rt_pr = Rt_gt
    proj_2d_pred = compute_projection(vertices, Rt_pr, cam_K)
    transform_3d_pred = compute_transformation(vertices, Rt_pr)

    min_x, min_y, max_x, max_y = int(np.min(proj_2d_pred[0])), int(np.min(proj_2d_pred[1])), \
                                 int(np.max(proj_2d_pred[0])), \
                                 int(np.max(proj_2d_pred[1]))

    corner_patches[i] = [min_x, min_y, max_x, max_y]
    Rt_pr_patches[i] = Rt_pr

    norm2d = np.linalg.norm(proj_2d_gt - proj_2d_pred, axis=0)
    norm3d = np.linalg.norm(transform_3d_gt - transform_3d_pred, axis=0)

    pixel_dist = np.mean(norm2d)
    vertex_dist = np.mean(norm3d)

    objkeypoints_projection = compute_projection(np.c_[np.array(objpoints3D), np.ones((len(objpoints3D), 1))].transpose(), Rt_pr, cam_K).T
    bboxes_patches[i] = np.reshape(objkeypoints_projection, ((cfg.args.num_detection_points) * 2))

    conf = max(np.exp(2.0 * (1.0 - (pixel_dist/(2*d_th_2d)))) - 1, 0)
    conf0 = np.exp(([2.0])) - 1
    gtconf2d_patches[i] = conf/conf0

    conf = max(np.exp(2.0 * (1.0 - (vertex_dist/(2*d_th_3d)))) - 1, 0)
    conf0 = np.exp(([2.0])) - 1
    gtconf3d_patches[i] = conf/conf0

    return Rt_pr_patches, corner_patches, gtconf2d_patches, gtconf3d_patches, bboxes_patches


def pose_proposals_nearby_test(objpoints3D, vertices, cam_K, obj_diameter, data):
    bboxes_batch, scores_batch, cls_inds_batch, gt_boxes_batch, gtimage_batch = data

    vertices = np.c_[np.array(vertices), np.ones((len(vertices), 1))].transpose()

    num_classes = cfg.num_classes

    d_th_2d = 10.0
    d_th_3d = 2 * 0.1 * obj_diameter

    bbox_pred = np.reshape(bboxes_batch, [-1, (2 * cfg.args.num_detection_points)])
    scores = np.reshape(scores_batch, [-1])
    cls_inds = np.reshape(cls_inds_batch, [-1])

    keep = np.zeros(len(bbox_pred), dtype=np.int)

    # print c_scores.shape
    if scores.shape[0] < cfg.args.topk:
        scores = np.tile(scores, cfg.args.topk)
        cls_inds = np.tile(cls_inds, cfg.args.topk)
        bbox_pred = np.tile(bbox_pred, [cfg.args.topk, 1])
        keep = np.zeros(len(bbox_pred), dtype=np.int)

    inds = np.where(cls_inds == 0)[0]
    c_scores = scores[inds]

    c_keep = np.argpartition(c_scores, -cfg.args.topk)[-cfg.args.topk:]
    keep[inds[c_keep]] = 1

    keep = np.where(keep > 0)
    bbox_pred = bbox_pred[keep]
    # scores = scores[keep]

    R_gt, t_gt = pnp(objpoints3D, np.reshape(gt_boxes_batch, ((cfg.args.num_detection_points), 2)), cam_K)
    Rt_gt = np.concatenate((R_gt, t_gt), axis=1)
    proj_2d_gt = compute_projection(vertices, Rt_gt, cam_K)

    transform_3d_gt = compute_transformation(vertices, Rt_gt)


    Rt_pr_patches = np.zeros([cfg.k_proposals_test, 3, 4]).astype(np.float32)
    gtconf2d_patches = np.zeros(cfg.k_proposals_test).astype(np.float32)
    gtconf3d_patches = np.zeros(cfg.k_proposals_test).astype(np.float32)
    corner_patches = np.zeros([cfg.k_proposals_test, 4]).astype(np.int)
    bboxes_patches = np.zeros([cfg.k_proposals_test, (2 * cfg.args.num_detection_points)]).astype(np.float32)

    n_nearby = cfg.args.nearby_test

    for i in xrange(cfg.args.topk):
        R_pr, t_pr = pnp(objpoints3D, np.reshape(bbox_pred[i], ((cfg.args.num_detection_points), 2)), cam_K)
        Rt_pr_ori = np.concatenate((R_pr, t_pr), axis=1)

        for j in xrange(n_nearby):
            Rt_pr = Rt_pr_ori.copy()
            Rt_pr[2, 3] = Rt_pr[2, 3] - (int(n_nearby/2) * 0.02 * obj_diameter) + j * (0.02 * obj_diameter)

            proj_2d_pred = compute_projection(vertices, Rt_pr, cam_K)
            transform_3d_pred = compute_transformation(vertices, Rt_pr)

            min_x, min_y, max_x, max_y = int(np.min(proj_2d_pred[0])), int(np.min(proj_2d_pred[1])), \
                                         int(np.max(proj_2d_pred[0])), \
                                         int(np.max(proj_2d_pred[1]))

            corner_patches[(n_nearby*i) + j] = [min_x, min_y, max_x, max_y]
            Rt_pr_patches[(n_nearby*i) + j] = Rt_pr

            objkeypoints_projection = compute_projection(np.c_[np.array(objpoints3D), np.ones((len(objpoints3D), 1))].transpose(), Rt_pr, cam_K).T
            bboxes_patches[(n_nearby*i) + j] = np.reshape(objkeypoints_projection, ((cfg.args.num_detection_points) * 2))

            norm2d = np.linalg.norm(proj_2d_gt - proj_2d_pred, axis=0)
            norm3d = np.linalg.norm(transform_3d_gt - transform_3d_pred, axis=0)

            pixel_dist = np.mean(norm2d)
            vertex_dist = np.mean(norm3d)

            conf = max(np.exp(2.0 * (1.0 - (pixel_dist/(2*d_th_2d)))) - 1, 0)
            conf0 = np.exp(([2.0])) - 1
            gtconf2d_patches[(n_nearby*i) + j] = conf/conf0

            conf = max(np.exp(2.0 * (1.0 - (vertex_dist/(2*d_th_3d)))) - 1, 0)
            conf0 = np.exp(([2.0])) - 1
            gtconf3d_patches[(n_nearby*i) + j] = conf/conf0

    return Rt_pr_patches, corner_patches, gtconf2d_patches, gtconf3d_patches, bboxes_patches