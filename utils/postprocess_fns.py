import numpy as np
import cfgs.config_yolo6d as cfg
import math
from utils.pnp_algorithms import pnp_ransac, pnp
from utils.compute_transoformations import compute_projection


def confidence_function_fast(bbox_np_b, gt_boxes_b, th=30, sharpness=2):
    gt_boxes_b = np.tile(gt_boxes_b, [bbox_np_b.shape[0], 1])
    bbox_np_b = bbox_np_b.reshape(bbox_np_b.shape[0], cfg.args.num_detection_points, 2)
    gt_boxes_b = gt_boxes_b.reshape(gt_boxes_b.shape[0], cfg.args.num_detection_points, 2)
    dist = np.linalg.norm(bbox_np_b - gt_boxes_b, axis=2)
    eps = 1e-5

    conf = np.where(dist < th, np.exp(sharpness * (1.0 - (dist/th))) - 1, 0)
    conf0 = np.exp(([sharpness])) - 1 + eps
    conf = conf/conf0
    conf = np.mean(conf, axis=1, keepdims=True)
    return conf


# not using anchors, simply transforming to prediction domain to the 0-1 scale coordinate system
def cell_to_imagescale(bbox_pred, anchors, H, W):
    bsize = bbox_pred.shape[0]
    num_anchors = anchors.shape[0]
    bbox_out = np.zeros((bsize, H*W, num_anchors, cfg.args.num_detection_points * 2), dtype=np.float)

    cols = np.repeat(np.tile(np.arange(W), H), cfg.args.num_detection_points).reshape(H*W, 1, cfg.args.num_detection_points)
    rows = np.repeat(np.repeat(np.arange(H), W), cfg.args.num_detection_points).reshape(H*W, 1, cfg.args.num_detection_points)
    bbox_out[:, :, :, 0::2] = (bbox_pred[:, :, :, 0::2] + cols) / W
    bbox_out[:, :, :, 1::2] = (bbox_pred[:, :, :, 1::2] + rows) / H

    return bbox_out


def postprocess(im_shape, thresh,
                size_index, non_nms, data):
    """
    bbox_pred: (bsize, HxW, num_anchors, 4)
               ndarray of float (sig(tx), sig(ty), exp(tw), exp(th))
    conf_pred: (bsize, HxW, num_anchors, 1)
    prob_pred: (bsize, HxW, num_anchors, num_classes)
    """

    # bbox_pred, conf_pred, prob_pred = data
    bbox_pred, conf_pred, prob_pred = data
    bsize, hw, num_anchors, _ = bbox_pred.shape

    num_classes = cfg.num_classes
    anchors = cfg.anchors
    W, H = int(math.sqrt(hw)), int(math.sqrt(hw))

    assert bbox_pred.shape[0] == 1, 'postprocess only support one image per batch'  # noqa

    bbox_pred = cell_to_imagescale(
        np.ascontiguousarray(bbox_pred, dtype=np.float),
        np.ascontiguousarray(anchors, dtype=np.float),
        H, W)
    bbox_pred = np.reshape(bbox_pred, [-1, (2 * cfg.args.num_detection_points)])
    bbox_pred[:, 0::2] *= float(im_shape[1])
    bbox_pred[:, 1::2] *= float(im_shape[0])

    conf_pred = np.reshape(conf_pred, [-1])
    prob_pred = np.reshape(prob_pred, [-1, num_classes])

    cell_inds = np.arange(H*W)
    cls_inds = np.argmax(prob_pred, axis=1)
    prob_pred = prob_pred[(np.arange(prob_pred.shape[0]), cls_inds)]
    scores = conf_pred * prob_pred

    assert len(scores) == len(bbox_pred), '{}, {}'.format(scores.shape, bbox_pred.shape)

    # 3*3 suppression
    keep = np.zeros(len(bbox_pred), dtype=np.int)

    for i in range(num_classes):
        inds = np.where(cls_inds == i)[0]
        if len(inds) == 0:
            continue
        c_conf = conf_pred[inds]
        c_bboxes = bbox_pred[inds]
        c_scores = scores[inds]
        c_cell_inds = cell_inds[inds]

        ##### keeping the box with maximum score only
        if non_nms:
            c_keep = np.where(c_scores >= thresh)
        else:
            c_keep = np.argmax(c_scores)

        keep[inds[c_keep]] = 1

    # keep = nms_detections(bbox_pred, scores, 0.3)

    if non_nms:
        keep = np.where(keep > 0)
        bbox_pred = bbox_pred[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]
    else:
        ##### keeping the box with maximum score only
        keep = np.argmax(scores)
        bbox_pred = bbox_pred[None, keep]
        scores = scores[None, keep, ]
        cls_inds = cls_inds[None, keep]

    return bbox_pred, scores, cls_inds


def final_postprocess(data):
    """
    data: as below-->
    bboxes_batch: (bsize, num_detections, cfg.args.num_detection_points)
    scores_batch: (bsize, num_detections, 1)
    cls_inds_batch: (bsize, num_detections, 1)
    """

    topk = cfg.args.topk

    bbox_pred, scores, cls_inds = data

    keep = np.zeros(len(bbox_pred), dtype=np.int)

    if scores.shape[0] < cfg.args.topk:
        scores = np.tile(scores, cfg.args.topk)
        cls_inds = np.tile(cls_inds, cfg.args.topk)
        bbox_pred = np.tile(bbox_pred, [cfg.args.topk, 1])
        keep = np.zeros(len(bbox_pred), dtype=np.int)

    ### cls_inds is zero because other classes are background in single object pose estimation setting
    inds = np.where(cls_inds == 0)[0]

    c_scores = scores[inds]
    c_keep = np.argpartition(c_scores, -topk)[-topk:]
    keep[inds[c_keep]] = 1

    keep = np.where(keep > 0)
    bbox_pred = bbox_pred[keep]
    scores = scores[keep]
    cls_inds = cls_inds[keep]

    return bbox_pred, scores, cls_inds


def final_postprocess_withplotlogs(data):
    """
    data: as below-->
    bboxes_batch: (bsize, num_detections, cfg.args.num_detection_points)
    scores_batch: (bsize, num_detections, 1)
    cls_inds_batch: (bsize, num_detections, 1)
    """

    topk = cfg.args.topk

    bbox_pred, scores, cls_inds, gt_boxes = data

    keep = np.zeros(len(bbox_pred), dtype=np.int)

    if scores.shape[0] < cfg.args.topk:
        scores = np.tile(scores, cfg.args.topk)
        cls_inds = np.tile(cls_inds, cfg.args.topk)
        bbox_pred = np.tile(bbox_pred, [cfg.args.topk, 1])
        keep = np.zeros(len(bbox_pred), dtype=np.int)

    ### cls_inds is zero because other classes are background in single object pose estimation setting
    inds = np.where(cls_inds == 0)[0]

    c_scores = scores[inds]
    c_keep = np.argpartition(c_scores, -topk)[-topk:]
    keep[inds[c_keep]] = 1

    keep = np.where(keep > 0)
    bbox_pred = bbox_pred[keep]
    scores = scores[keep]
    cls_inds = cls_inds[keep]

    gt_confs = confidence_function_fast(bbox_pred, gt_boxes)

    return bbox_pred, scores, gt_confs[:, 0], cls_inds


def pickone_postprocess(data):
    """
    data: as below-->
    bboxes_batch: (bsize, num_detections, cfg.args.num_detection_points)
    scores_batch: (bsize, num_detections, 1)
    cls_inds_batch: (bsize, num_detections, 1)
    """

    bbox_pred, scores, cls_inds = data

    keep = np.zeros(len(bbox_pred), dtype=np.int)

    ### cls_inds is zero because other classes are background in single object pose estimation setting
    inds = np.where(cls_inds == 0)[0]

    c_scores = scores[inds]
    c_keep = np.argmax(c_scores)
    keep[inds[c_keep]] = 1

    keep = np.where(keep > 0)
    bbox_pred = bbox_pred[keep]
    scores = scores[keep]
    cls_inds = cls_inds[keep]

    return bbox_pred, scores, cls_inds


def ransac_postprocess(cam_K, objpoints3D, data):
    """
    data: as below-->
    bboxes_batch: (bsize, num_detections, cfg.args.num_detection_points)
    scores_batch: (bsize, num_detections, 1)
    cls_inds_batch: (bsize, num_detections, 1)
    """

    bbox_pred, scores, cls_inds = data

    elongated_bbox = np.reshape(bbox_pred, (bbox_pred.shape[0], (cfg.args.num_detection_points), 2))
    elongated_bbox = np.reshape(elongated_bbox, [-1, 2])
    repeated_objpoints3D = np.tile(objpoints3D.T, bbox_pred.shape[0]).T

    R, t = pnp_ransac(repeated_objpoints3D, elongated_bbox, cam_K)
    Rt = np.concatenate((R, t), axis=1)

    corners_3d = np.concatenate((np.transpose(objpoints3D), np.ones((1, 9))), axis=0)

    bbox_pred = np.zeros((1, 2*cfg.args.num_detection_points))
    ### cls_inds is zero because other classes are background in single object pose estimation setting
    bbox_pred[0, :] = compute_projection(corners_3d, Rt, cam_K).T.reshape([-1])
    conf_pred = np.array([1.0])
    cls_inds = np.array([0], dtype=np.int)

    return bbox_pred, conf_pred, cls_inds


def seg_cullnet_postprocess(im_shape, size_index, data):
    keypoints, conf_new = data

    bbox_pred = np.zeros((1, 2*cfg.args.num_detection_points))

    keep = np.argmax(conf_new)

    bbox_pred[0, :] = keypoints[keep]
    conf_pred = np.array([1.0])
    cls_inds = np.array([0], dtype=np.int)
    return bbox_pred, conf_pred, cls_inds
