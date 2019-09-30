import numpy as np
import cfgs.config_yolo6d as cfg
import cv2
from utils.pnp_algorithms import pnp
from utils.load_meshinfo import threed_corners, threed_correspondences

if cfg.args.num_detection_points == 9:
    objpoints3D = threed_corners(cfg.args.class_name)
else:
    objpoints3D = threed_correspondences(cfg.args.class_name)


def my_collate(batch):
    # size_index = randint(0, len(cfg.multi_scale_inp_size) - 1)
    size_index = 3

    # multi-scale
    w, h = cfg.multi_scale_inp_size[size_index]

    imgs = []
    annots = []
    gt_classes = []
    gt_RT = []
    origin_gtboxes = []
    origin_im = []
    samples = {}

    for data in batch:
        origin_gtboxes_b = np.asarray(data['gt_boxes'], dtype=np.float).copy()
        gt_boxes = np.asarray(data['gt_boxes'], dtype=np.float)
        # if len(gt_boxes) > 0:
        #     gt_boxes[:, 0::2] *= float(w) / data['image'].shape[1]
        #     gt_boxes[:, 1::2] *= float(h) / data['image'].shape[0]
        # data['image'] = cv2.resize(data['image'], (w, h))

        if cfg.args.diff_aspectratio:
            ### diff aspect ratio 640 X 640
            if len(gt_boxes) > 0:
                gt_boxes[:, 0::2] *= float(w) / data['image'].shape[1]
                gt_boxes[:, 1::2] *= float(h) / data['image'].shape[0]
            data['image'] = cv2.resize(data['image'], (w, h))
        else:
            #### same aspect ratio 640 X 640
            old_size = data['image'].shape
            delta_w = w - old_size[1]
            delta_h = h - old_size[0]
            data['image'] = cv2.copyMakeBorder(data['image'], 0, delta_h, 0, delta_w, cv2.BORDER_CONSTANT,
                                               value=[0, 0, 0])

        if cfg.args.dataset_name=='LINEMOD':
            K = cfg.cam_K
        elif cfg.args.dataset_name=='YCB':
            K = cfg.cam_K1

        if cfg.args.gt_RT_available and not cfg.args.random_trans_scale:
            Rt = data['gt_RT']
        else:
            R, t = pnp(objpoints3D, origin_gtboxes_b.reshape(cfg.args.num_detection_points, 2), K)
            Rt = np.concatenate((R, t), axis=1)

        imgs.append(data['image'])
        annots.append(gt_boxes)
        gt_classes.append(data['gt_classes'])
        origin_im.append(data['origin_im'])
        gt_RT.append(Rt.reshape(12))
        origin_gtboxes.append(origin_gtboxes_b)

    samples['image'] = np.stack(imgs, axis=0)
    samples['gt_boxes'] = annots
    samples['origin_gtboxes'] = origin_gtboxes
    samples['gt_classes'] = gt_classes
    samples['origin_im'] = origin_im
    samples['dontcare'] = []
    samples['gt_RT'] = gt_RT

    return samples, size_index


def my_collate_cullnet(batch):
    # size_index = randint(0, len(cfg.multi_scale_inp_size) - 1)
    size_index = 3

    # multi-scale
    w, h = cfg.multi_scale_inp_size[size_index]

    imgs = []
    annots = []
    gt_classes = []
    gt_RT = []
    origin_gtboxes = []
    origin_im = []
    samples = {}
    rgb_patches = []
    gt_2dconfs = []
    gt_3dconfs = []
    bboxes = []

    for data in batch:
        origin_gtboxes_b = np.asarray(data['gt_boxes'], dtype=np.float).copy()
        gt_boxes = np.asarray(data['gt_boxes'], dtype=np.float)
        # if len(gt_boxes) > 0:
        #     gt_boxes[:, 0::2] *= float(w) / data['image'].shape[1]
        #     gt_boxes[:, 1::2] *= float(h) / data['image'].shape[0]
        # data['image'] = cv2.resize(data['image'], (w, h))

        if cfg.args.diff_aspectratio:
            ### diff aspect ratio 640 X 640
            if len(gt_boxes) > 0:
                gt_boxes[:, 0::2] *= float(w) / data['image'].shape[1]
                gt_boxes[:, 1::2] *= float(h) / data['image'].shape[0]
            data['image'] = cv2.resize(data['image'], (w, h))
        else:
            #### same aspect ratio 640 X 640
            old_size = data['image'].shape
            delta_w = w - old_size[1]
            delta_h = h - old_size[0]
            data['image'] = cv2.copyMakeBorder(data['image'], 0, delta_h, 0, delta_w, cv2.BORDER_CONSTANT,
                                               value=[0, 0, 0])

        if cfg.args.dataset_name=='LINEMOD':
            K = cfg.cam_K
        elif cfg.args.dataset_name=='YCB':
            K = cfg.cam_K1

        if cfg.args.gt_RT_available and not cfg.args.random_trans_scale:
            Rt = data['gt_RT']
        else:
            R, t = pnp(objpoints3D, origin_gtboxes_b.reshape(cfg.args.num_detection_points, 2), K)
            Rt = np.concatenate((R, t), axis=1)

        rgb_patches_b, gt_2dconfs_b, gt_3dconfs_b, bboxes_b = data['pose_proposals']

        imgs.append(data['image'])
        annots.append(gt_boxes)
        gt_classes.append(data['gt_classes'])
        origin_im.append(data['origin_im'])
        gt_RT.append(Rt.reshape(12))
        origin_gtboxes.append(origin_gtboxes_b)

        rgb_patches.append(rgb_patches_b)
        gt_2dconfs.append(gt_2dconfs_b)
        gt_3dconfs.append(gt_3dconfs_b)
        bboxes.append(bboxes_b)

    samples['image'] = np.stack(imgs, axis=0)
    samples['gt_boxes'] = annots
    samples['origin_gtboxes'] = origin_gtboxes
    samples['gt_classes'] = gt_classes
    samples['origin_im'] = origin_im
    samples['dontcare'] = []
    samples['gt_RT'] = gt_RT
    samples['rgb_patches'] = rgb_patches
    samples['gt_2dconfs'] = gt_2dconfs
    samples['gt_3dconfs'] = gt_3dconfs
    samples['bboxes'] = bboxes

    return samples, size_index
