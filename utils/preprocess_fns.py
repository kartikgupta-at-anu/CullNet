import cv2, os
import numpy as np
import cfgs.config_yolo6d as cfg
from random import randint
from .im_transform_yolo6d import imcv2_affine_trans, random_distort_image
from compute_transoformations import compute_projection, compute_transformation
from rotation_transformations import *
import pickle, random


def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """
    if boxes.shape[0] == 0:
        return boxes

    for i in range(0, cfg.args.num_detection_points):
        # x1 >= 0 and x1 < im_shape[1]
        boxes[:, 2*i::(2 * cfg.args.num_detection_points)] = np.maximum(
            np.minimum(boxes[:, 2*i::(2 * cfg.args.num_detection_points)], im_shape[1] - 1), 0)
        # y1 >= 0 and y1 < im_shape[0]
        boxes[:, (2*i)+1::(2 * cfg.args.num_detection_points)] = np.maximum(
            np.minimum(boxes[:, (2*i)+1::(2 * cfg.args.num_detection_points)], im_shape[0] - 1), 0)
    return boxes


def _offset_boxes(boxes, im_shape, scale, offs, flip):
    if len(boxes) == 0:
        return boxes
    boxes = np.asarray(boxes, dtype=np.float)
    boxes *= scale

    boxes[:, 0::2] -= offs[0]
    boxes[:, 1::2] -= offs[1]

    return boxes


def augment_Rt(Rt, obj_diameter):
    R = Rt[0:3, 0:3]
    t = Rt[0:3, 3]
    t_aug = t + [round(random.uniform(-obj_diameter/10.0, obj_diameter/10.0), 5),
                 round(random.uniform(-obj_diameter/10.0, obj_diameter/10.0), 5),
                 round(random.uniform(-obj_diameter/5.0, obj_diameter/5.0), 5)]

    theta_euler_rad = rotationMatrixToEulerAngles(R)
    theta_euler_deg = [math.degrees(theta_euler_rad[0]), math.degrees(theta_euler_rad[1]), math.degrees(theta_euler_rad[2])]
    theta_euler_deg_aug = np.array(theta_euler_deg, dtype=np.float32) + [random.randint(-12.0, 12.0),
                                                                         random.randint(-12.0, 12.0),
                                                                         random.randint(-12.0, 12.0)]
    theta_euler_rad_aug = np.array([math.radians(theta_euler_deg_aug[0]), math.radians(theta_euler_deg_aug[1]), math.radians(theta_euler_deg_aug[2])], dtype=np.float)
    R_aug = eulerAnglesToRotationMatrix(theta_euler_rad_aug)
    Rt_aug = np.concatenate((R_aug, np.expand_dims(t_aug, 1)), axis=1)
    return Rt_aug


def preprocess_train(data):
    im_path, blob = data

    boxes, gt_classes, gt_RT = blob['boxes'], blob['gt_classes'], blob['gt_RT']

    im = cv2.imread(im_path)

    if cfg.args.random_erasing:
        bbox = [int(np.min(boxes[:, 0::2])), int(np.min(boxes[:, 1::2])), int(np.max(boxes[:, 0::2])), int(np.max(boxes[:, 1::2]))]
        occ_augmentor = RandomErasing()
        im[bbox[1]:bbox[3], bbox[0]:bbox[2], :] = occ_augmentor(im[bbox[1]:bbox[3], bbox[0]:bbox[2], :])

    if cfg.args.random_trans_scale:
        im, trans_param = imcv2_affine_trans(im)
        scale, offs, flip = trans_param
        boxes = _offset_boxes(boxes, im.shape, scale, offs, flip)

    ori_im = np.copy(im)

    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = random_distort_image(im)
    boxes = np.asarray(boxes, dtype=np.float)
    return im, boxes, gt_classes, [], ori_im, gt_RT


def preprocess_train_cullnet(data):
    im_path, index, blob, vertices, objpoints3D, obj_diameter = data

    vertices = np.c_[np.array(vertices), np.ones((len(vertices), 1))].transpose()

    backbone_model_dir = os.path.join(cfg.TRAIN_DIR, cfg.args.exp_name)
    net1_cachedir_train = backbone_model_dir + '/net1_output/train'

    cache_file = net1_cachedir_train + '/' + index + '.pkl'
    with open(cache_file, 'rb') as fid:
        net1_out = pickle.load(fid)
        Rt_pr_patches, corner_patches, gt_2dconfs, gt_3dconfs, bboxes = net1_out

    boxes, gt_classes, gt_RT = blob['boxes'], blob['gt_classes'], blob['gt_RT']

    im = cv2.imread(im_path)

    if cfg.args.random_erasing:
        bbox = [int(np.min(boxes[:, 0::2])), int(np.min(boxes[:, 1::2])), int(np.max(boxes[:, 0::2])), int(np.max(boxes[:, 1::2]))]
        occ_augmentor = RandomErasing()
        im[bbox[1]:bbox[3], bbox[0]:bbox[2], :] = occ_augmentor(im[bbox[1]:bbox[3], bbox[0]:bbox[2], :])

    if cfg.args.random_trans_scale:
        im, trans_param = imcv2_affine_trans(im)
        scale, offs, flip = trans_param
        boxes = _offset_boxes(boxes, im.shape, scale, offs, flip)

    ori_im = np.copy(im)

    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = random_distort_image(im)
    boxes = np.asarray(boxes, dtype=np.float)

    ################## for cullnet purpose
    if cfg.args.cullnet_inconf=='concat':
        rgb_patches = np.zeros([cfg.args.k_proposals, cfg.args.cullnet_input, cfg.args.cullnet_input, 4]).astype(np.float32)
        rgb_patches_ori = np.zeros([cfg.args.k_proposals, cfg.args.cullnet_input, cfg.args.cullnet_input, 3]).astype('uint8')
        mask_patches = np.zeros([cfg.args.k_proposals, cfg.args.cullnet_input, cfg.args.cullnet_input, 1]).astype(np.int8)
    else:
        rgb_patches = np.zeros([cfg.args.k_proposals, cfg.args.cullnet_input, cfg.args.cullnet_input, 3]).astype(np.float32)
        rgb_patches_ori = np.zeros([cfg.args.k_proposals, cfg.args.cullnet_input, cfg.args.cullnet_input, 3]).astype('uint8')
        mask_patches = np.zeros([cfg.args.k_proposals, cfg.args.cullnet_input, cfg.args.cullnet_input, 1]).astype(np.int8)

    gt_2dconfs_data = np.zeros([cfg.args.k_proposals, 1]).astype(np.float32)
    gt_3dconfs_data = np.zeros([cfg.args.k_proposals, 1]).astype(np.float32)
    bboxes_data = np.zeros([cfg.args.k_proposals, (2 * cfg.args.num_detection_points)]).astype(np.float32)

    if cfg.args.dataset_name=='LINEMOD':
        K = cfg.cam_K
    elif cfg.args.dataset_name=='YCB':
        K = cfg.cam_K1

    ori_image_appended = np.zeros([1480, 1640, 3], dtype='uint8')
    ori_image_appended[500:500+480, 500:500+640, :] = ori_im

    for i in xrange(17):
        object_out_bool = False
        proj_2d_pred = compute_projection(vertices, Rt_pr_patches[i], K)

        min_x, min_y, max_x, max_y = corner_patches[i]
        mask_image_appended = np.zeros([1480, 1640, 3])
        mask_image_appended[proj_2d_pred[1].astype(np.int) + 500, proj_2d_pred[0].astype(np.int) + 500] = 1.0
        mask_image = mask_image_appended[500:500+480, 500:500+640]
        mask_image_appended = np.zeros([1480, 1640, 3])
        mask_image_appended[500:500+480, 500:500+640, :] = mask_image

        try:
            rgb_patches_ori[i] = cv2.resize(ori_image_appended[500+min_y:500+max_y, 500+min_x:500+max_x], (cfg.args.cullnet_input, cfg.args.cullnet_input))
            rgb_patch = cv2.cvtColor(rgb_patches_ori[i], cv2.COLOR_BGR2RGB)

            mask_patch = cv2.resize(mask_image_appended[500+min_y:500+max_y, 500+min_x:500+max_x], (cfg.args.cullnet_input, cfg.args.cullnet_input))
        except:
            min_y, min_x = randint(0, 400), randint(0, 560)
            max_y, max_x = min_y+40, min_x+40
            rgb_patches_ori[i] = cv2.resize(ori_im[min_y:max_y, min_x:max_x], (cfg.args.cullnet_input, cfg.args.cullnet_input))
            rgb_patch = cv2.cvtColor(rgb_patches_ori[i], cv2.COLOR_BGR2RGB)
            mask_patch = cv2.resize(mask_image[min_y:max_y, min_x:max_x], (cfg.args.cullnet_input, cfg.args.cullnet_input))
            print "Problem in the proposal output so random patch taken of size 40 X 40"
            object_out_bool = True

        if (min_x<=0 and max_x<=0) or (min_y<=0 and max_y<=0) or (min_x>=640 and max_x>=640) or (min_y>=480 and max_y>=480):
            object_out_bool = True

        mask_patch = np.where(mask_patch[:, :, 0:1] > 0.5, 1.0, 0.0)
        mask_patches[i] = mask_patch

        if cfg.args.cullnet_inconf=='concat':
            rgb_patch = rgb_patch.astype(np.float) / 255.0
            rgb_patch[:, :, 0] -= 0.485
            rgb_patch[:, :, 1] -= 0.456
            rgb_patch[:, :, 2] -= 0.406
            rgb_patch[:, :, 0] /= 0.229
            rgb_patch[:, :, 1] /= 0.224
            rgb_patch[:, :, 2] /= 0.225
            rgb_patches[i] = np.concatenate((rgb_patch, mask_patch[:, :, 0:1]), 2)
        else:
            rgb_patch = rgb_patch.astype(np.float) / 255.0
            rgb_patch[:, :, 0] -= 0.485
            rgb_patch[:, :, 1] -= 0.456
            rgb_patch[:, :, 2] -= 0.406
            rgb_patch[:, :, 0] /= 0.229
            rgb_patch[:, :, 1] /= 0.224
            rgb_patch[:, :, 2] /= 0.225

            rgb_patches[i] = rgb_patch * mask_patch

        bboxes_data[i] = bboxes[i]

        if not object_out_bool:
            gt_2dconfs_data[i] = gt_2dconfs[i]
            gt_3dconfs_data[i] = gt_3dconfs[i]
        else:
            gt_2dconfs_data[i] = 0.0
            gt_3dconfs_data[i] = 0.0

    d_th_2d = 10.0
    d_th_3d = 2 * 0.1 * obj_diameter

    Rt_gt = Rt_pr_patches[16]
    proj_2d_gt = compute_projection(vertices, Rt_gt, K)
    transform_3d_gt = compute_transformation(vertices, Rt_gt)

    for i in xrange(17, cfg.args.k_proposals):
        object_out_bool = False

        Rt_pr_aug = augment_Rt(Rt_gt, obj_diameter)
        proj_2d_pred = compute_projection(vertices, Rt_pr_aug, K)

        min_x, min_y, max_x, max_y = int(np.min(proj_2d_pred[0])), int(np.min(proj_2d_pred[1])), \
                                     int(np.max(proj_2d_pred[0])), \
                                     int(np.max(proj_2d_pred[1]))

        mask_image_appended = np.zeros([1480, 1640, 3])
        mask_image_appended[proj_2d_pred[1].astype(np.int) + 500, proj_2d_pred[0].astype(np.int) + 500] = 1.0
        mask_image = mask_image_appended[500:500+480, 500:500+640]
        mask_image_appended = np.zeros([1480, 1640, 3])
        mask_image_appended[500:500+480, 500:500+640, :] = mask_image
        try:
            rgb_patches_ori[i] = cv2.resize(ori_image_appended[500+min_y:500+max_y, 500+min_x:500+max_x], (cfg.args.cullnet_input, cfg.args.cullnet_input))
            rgb_patch = cv2.cvtColor(rgb_patches_ori[i], cv2.COLOR_BGR2RGB)
            mask_patch = cv2.resize(mask_image_appended[500+min_y:500+max_y, 500+min_x:500+max_x], (cfg.args.cullnet_input, cfg.args.cullnet_input))
        except:
            print min_x, max_x, min_y, max_y
            min_y, min_x = randint(0, 400), randint(0, 560)
            max_y, max_x = min_y+40, min_x+40
            rgb_patches_ori[i] = cv2.resize(ori_im[min_y:max_y, min_x:max_x], (cfg.args.cullnet_input, cfg.args.cullnet_input))
            rgb_patch = cv2.cvtColor(rgb_patches_ori[i], cv2.COLOR_BGR2RGB)
            mask_patch = cv2.resize(mask_image[min_y:max_y, min_x:max_x], (cfg.args.cullnet_input, cfg.args.cullnet_input))
            print "Problem in the proposal output so random patch taken of size 40 X 40"
            object_out_bool = True

        if (min_x<=0 and max_x<=0) or (min_y<=0 and max_y<=0) or (min_x>=640 and max_x>=640) or (min_y>=480 and max_y>=480):
            object_out_bool = True

        mask_patch = np.where(mask_patch[:, :, 0:1] > 0.5, 1.0, 0.0)
        mask_patches[i] = mask_patch

        if cfg.args.cullnet_inconf=='concat':
            rgb_patch = rgb_patch.astype(np.float) / 255.0
            rgb_patch[:, :, 0] -= 0.485
            rgb_patch[:, :, 1] -= 0.456
            rgb_patch[:, :, 2] -= 0.406
            rgb_patch[:, :, 0] /= 0.229
            rgb_patch[:, :, 1] /= 0.224
            rgb_patch[:, :, 2] /= 0.225
            rgb_patches[i] = np.concatenate((rgb_patch, mask_patch[:, :, 0:1]), 2)
        else:
            rgb_patch = rgb_patch.astype(np.float) / 255.0
            rgb_patch[:, :, 0] -= 0.485
            rgb_patch[:, :, 1] -= 0.456
            rgb_patch[:, :, 2] -= 0.406
            rgb_patch[:, :, 0] /= 0.229
            rgb_patch[:, :, 1] /= 0.224
            rgb_patch[:, :, 2] /= 0.225

            rgb_patches[i] = rgb_patch * mask_patch

        objkeypoints_projection = compute_projection(np.c_[np.array(objpoints3D), np.ones((len(objpoints3D), 1))].transpose(), Rt_pr_aug, K).T
        transform_3d_pred = compute_transformation(vertices, Rt_pr_aug)
        bboxes_data[i] = np.reshape(objkeypoints_projection, ((cfg.args.num_detection_points) * 2))

        if not object_out_bool:
            norm2d = np.linalg.norm(proj_2d_gt - proj_2d_pred, axis=0)
            norm3d = np.linalg.norm(transform_3d_gt - transform_3d_pred, axis=0)

            pixel_dist = np.mean(norm2d)
            vertex_dist = np.mean(norm3d)

            conf = max(np.exp(2.0 * (1.0 - (pixel_dist/(2*d_th_2d)))) - 1, 0)
            conf0 = np.exp(([2.0])) - 1
            gt_2dconfs_data[i] = conf/conf0

            conf = max(np.exp(2.0 * (1.0 - (vertex_dist/(2*d_th_3d)))) - 1, 0)
            conf0 = np.exp(([2.0])) - 1
            gt_3dconfs_data[i] = conf/conf0
        else:
            gt_2dconfs_data[i] = 0.0
            gt_3dconfs_data[i] = 0.0

    poseproposals_data = rgb_patches, gt_2dconfs_data, gt_3dconfs_data, bboxes_data

    return im, boxes, gt_classes, [], ori_im, gt_RT, poseproposals_data


def preprocess_test_cullnet(data):
    im_path, index, blob, vertices = data

    vertices = np.c_[np.array(vertices), np.ones((len(vertices), 1))].transpose()

    backbone_model_dir = os.path.join(cfg.TRAIN_DIR, cfg.args.exp_name)

    net1_cachedir_test = backbone_model_dir + '/net1_output_topk' + str(cfg.args.topk) + '_nearby' + str(cfg.args.nearby_test) + '/test'

    if cfg.args.datadirname=='singleobject_occtest':
        net1_cachedir_test = backbone_model_dir + '/net1_output_topk' + str(cfg.args.topk) + '_nearby' + str(cfg.args.nearby_test) + '_occlusion/test'
    if cfg.imdb_test=='linemod_testerror':
        net1_cachedir_test = backbone_model_dir + '/net1_output_topk' + str(cfg.args.topk) + '_nearby' + str(cfg.args.nearby_test) + '_linemodtrain_ori/test'


    cache_file = net1_cachedir_test + '/' + index + '.pkl'
    with open(cache_file, 'rb') as fid:
        net1_out = pickle.load(fid)
        Rt_pr_patches, corner_patches, gt_2dconfs, gt_3dconfs, bboxes = net1_out

    boxes, gt_classes, gt_RT = blob['boxes'], blob['gt_classes'], blob['gt_RT']

    im = cv2.imread(im_path)

    ori_im = np.copy(im)

    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = random_distort_image(im)
    boxes = np.asarray(boxes, dtype=np.float)

    ################## for cullnet purpose
    if cfg.args.cullnet_inconf=='concat':
        rgb_patches = np.zeros([cfg.k_proposals_test, cfg.args.cullnet_input, cfg.args.cullnet_input, 4]).astype(np.float32)
        rgb_patches_ori = np.zeros([cfg.k_proposals_test, cfg.args.cullnet_input, cfg.args.cullnet_input, 3]).astype('uint8')
        mask_patches = np.zeros([cfg.k_proposals_test, cfg.args.cullnet_input, cfg.args.cullnet_input, 1]).astype(np.int8)
    else:
        rgb_patches = np.zeros([cfg.k_proposals_test, cfg.args.cullnet_input, cfg.args.cullnet_input, 3]).astype(np.float32)
        rgb_patches_ori = np.zeros([cfg.k_proposals_test, cfg.args.cullnet_input, cfg.args.cullnet_input, 3]).astype('uint8')
        mask_patches = np.zeros([cfg.k_proposals_test, cfg.args.cullnet_input, cfg.args.cullnet_input, 1]).astype(np.int8)

    if cfg.args.dataset_name=='LINEMOD':
        K = cfg.cam_K
    elif cfg.args.dataset_name=='YCB':
        K = cfg.cam_K1

    ori_image_appended = np.zeros([1480, 1640, 3], dtype='uint8')
    ori_image_appended[500:500+480, 500:500+640, :] = ori_im

    for i in xrange(cfg.k_proposals_test):
        proj_2d_pred = compute_projection(vertices, Rt_pr_patches[i], K)

        min_x, min_y, max_x, max_y = corner_patches[i]
        mask_image_appended = np.zeros([1480, 1640, 3])
        mask_image_appended[proj_2d_pred[1].astype(np.int) + 500, proj_2d_pred[0].astype(np.int) + 500] = 1.0
        mask_image = mask_image_appended[500:500+480, 500:500+640]
        mask_image_appended = np.zeros([1480, 1640, 3])
        mask_image_appended[500:500+480, 500:500+640, :] = mask_image
        try:
            rgb_patches_ori[i] = cv2.resize(ori_image_appended[500+min_y:500+max_y, 500+min_x:500+max_x], (cfg.args.cullnet_input, cfg.args.cullnet_input))
            rgb_patch = cv2.cvtColor(rgb_patches_ori[i], cv2.COLOR_BGR2RGB)
            mask_patch = cv2.resize(mask_image_appended[500+min_y:500+max_y, 500+min_x:500+max_x], (cfg.args.cullnet_input, cfg.args.cullnet_input))
        except:
            min_y, min_x = randint(0, 400), randint(0, 560)
            max_y, max_x = min_y+40, min_x+40
            rgb_patches_ori[i] = cv2.resize(ori_im[min_y:max_y, min_x:max_x], (cfg.args.cullnet_input, cfg.args.cullnet_input))
            rgb_patch = cv2.cvtColor(rgb_patches_ori[i], cv2.COLOR_BGR2RGB)
            mask_patch = cv2.resize(mask_image[min_y:max_y, min_x:max_x], (cfg.args.cullnet_input, cfg.args.cullnet_input))
            print "Problem in the proposal output so random patch taken of size 40 X 40"

        mask_patch = np.where(mask_patch[:, :, 0:1] > 0.5, 1.0, 0.0)
        mask_patches[i] = mask_patch

        if cfg.args.cullnet_inconf=='concat':
            rgb_patch = rgb_patch.astype(np.float) / 255.0
            rgb_patch[:, :, 0] -= 0.485
            rgb_patch[:, :, 1] -= 0.456
            rgb_patch[:, :, 2] -= 0.406
            rgb_patch[:, :, 0] /= 0.229
            rgb_patch[:, :, 1] /= 0.224
            rgb_patch[:, :, 2] /= 0.225
            rgb_patches[i] = np.concatenate((rgb_patch, mask_patch[:, :, 0:1]), 2)
        else:
            rgb_patch = rgb_patch.astype(np.float) / 255.0
            rgb_patch[:, :, 0] -= 0.485
            rgb_patch[:, :, 1] -= 0.456
            rgb_patch[:, :, 2] -= 0.406
            rgb_patch[:, :, 0] /= 0.229
            rgb_patch[:, :, 1] /= 0.224
            rgb_patch[:, :, 2] /= 0.225

            rgb_patches[i] = rgb_patch * mask_patch

    poseproposals_data = rgb_patches, gt_2dconfs, gt_3dconfs, bboxes

    return im, boxes, gt_classes, [], ori_im, gt_RT, poseproposals_data


def preprocess_test(data, size):

    im, blob = data
    boxes, gt_classes, gt_RT = blob['boxes'], blob['gt_classes'], blob['gt_RT']

    if isinstance(im, str):
        im = cv2.imread(im)

    ori_im = np.copy(im)
    ori_boxes = np.copy(boxes)

    if size is not None:
        w, h = size

        if cfg.args.diff_aspectratio:
            ## diff aspect ratio 640 X 640
            im = cv2.resize(im, (w, h))
        else:
            #### same aspect ratio 640 X 640
            old_size = im.shape
            delta_w = w - old_size[1]
            delta_h = h - old_size[0]
            im = cv2.copyMakeBorder(im, 0, delta_h, 0, delta_w, cv2.BORDER_CONSTANT,
                                    value=[0, 0, 0])

    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    gt_boxes = np.asarray(boxes, dtype=np.float)

    if cfg.args.diff_aspectratio:
        # diff aspect ratio 640 X 640
        if len(gt_boxes) > 0:
            gt_boxes[:, 0::2] *= float(w) / ori_im.shape[1]
            gt_boxes[:, 1::2] *= float(h) / ori_im.shape[0]

    # im = im / 255.

    return im, ori_im, gt_boxes, ori_boxes, gt_classes, gt_RT