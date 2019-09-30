import os
import cv2
import numpy as np
import pickle
import torch
import torch.nn.functional as F
import fcntl
import yaml, time, errno

from random import randint
from functools import partial
from multiprocessing import Pool
from torch.utils.data import DataLoader
from yolo6d_darknet_variedpoints_v3 import Darknet
import utils.yolo6d as yolo_utils
import utils.network as net_utils
from utils.timer import Timer
from datasets.dataset import Dataset_class
import cfgs.config_yolo6d as cfg


def mkdir(path, max_depth=3):
    parent, child = os.path.split(path)
    if not os.path.exists(parent) and max_depth > 1:
        mkdir(parent, max_depth - 1)

    if not os.path.exists(path):
        os.mkdir(path)


def save_results(datadir_name, class_name, accuracy_twod, accuracy_threed, cfg, epoch='minimum', occlusion=False, non_nms=False, topk=16, topk_ransac=False):

    result_filename = cfg.TEST_DIR + '/' + datadir_name + '.yml'

    if cfg.args.seg_cullnet:
        result_filename = cfg.TEST_DIR + '/' + datadir_name + '_' + 'cullnet.yml'

    if occlusion:
        result_filename = cfg.TEST_DIR + '/' + datadir_name + '_' + 'occlusion.yml'

    if non_nms:
        result_filename = cfg.TEST_DIR + '/' + datadir_name + '_' + 'non_nms.yml'

    if topk_ransac:
        result_filename = cfg.TEST_DIR + '/' + datadir_name + '_' + 'top' + str(topk) + '_ransac.yml'

    if os.path.exists(result_filename):
        outfile = open(result_filename, 'r+')
        data = yaml.load(outfile)
    else:
        data = {}

    if epoch in data:
        data[epoch]['Reprojection_metric'][class_name] = round(float(accuracy_twod) * 100, 2)
        data[epoch]['ADD'][class_name] = round(float(accuracy_threed) * 100, 2)
    else:
        data[epoch] = {}
        data[epoch]['Reprojection_metric'] = {}
        data[epoch]['ADD'] = {}
        data[epoch]['Reprojection_metric'][class_name] = round(float(accuracy_twod) * 100, 2)
        data[epoch]['ADD'][class_name] = round(float(accuracy_threed) * 100, 2)

    with open(result_filename, "w") as outfile_write:
        yaml.dump(data, outfile_write, default_flow_style=False)

    print('saved results in the results file')


def save_resultsdist(dists2d, dists3d, dir, cls):
    filename_2d = os.path.join(dir, cls+'_2d.txt')
    filename_3d = os.path.join(dir, cls+'_3d.txt')

    with open(filename_2d, 'w') as f:
        for item in dists2d:
            f.write("%s\n" % item)

    with open(filename_3d, 'w') as f:
        for item in dists3d:
            f.write("%s\n" % item)


def test_net(net, imdb, dataloader, args, output_dir, size_index, batch_size, iter_count=0, thresh=0.5, vis=True, verbose=True, summary=None):
    net.eval()

    test_loss = 0

    if cfg.args.num_detection_points == 9:
        objpoints3D = yolo_utils.threed_corners(args.class_name)
    else:
        objpoints3D = yolo_utils.threed_correspondences(args.class_name)

    corners_3d = yolo_utils.threed_corners(args.class_name)
    vertices = yolo_utils.threed_vertices(args.class_name)

    num_images = imdb.__len__()

    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(imdb._num_classes)]

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    pool = Pool(processes=4)
    batch_num_summary = randint(0, num_images/batch_size - 1)

    if args.confidence_plotlogs:
        confidence_plotlogs = {}
        confidence_plotlogs['gt_conf'] = []
        confidence_plotlogs['backbone_conf'] = []

    for i_batch, sample_batched in enumerate(dataloader):
        ori_im = np.array(sample_batched['origin_im'])
        im_data = net_utils.tensor_to_variable(sample_batched['image'], is_cuda=True,
                                               volatile=True).permute(0, 3, 1, 2)

        _t['im_detect'].tic()

        with torch.no_grad():
            bbox_pred_all, conf_pred_all, score_pred_all = net(im_data)
            if args.mGPUs:
                coord_loss_var, conf_objloss_var, conf_noobjloss_var, cls_loss_var = net.module.loss(bbox_pred_all,
                                                                                                     conf_pred_all,
                                                                                                     score_pred_all,
                                                                                                     sample_batched[
                                                                                                         'gt_boxes'],
                                                                                                     sample_batched[
                                                                                                         'gt_classes'],
                                                                                                     sample_batched[
                                                                                                         'gt_RT'],
                                                                                                     [],
                                                                                                     size_index)
            else:
                coord_loss_var, conf_objloss_var, conf_noobjloss_var, cls_loss_var = net.loss(bbox_pred_all, conf_pred_all,
                                                                                              score_pred_all,
                                                                                              sample_batched[
                                                                                                  'gt_boxes'],
                                                                                              sample_batched[
                                                                                                  'gt_classes'],
                                                                                              sample_batched[
                                                                                                  'gt_RT'],
                                                                                              [],
                                                                                              size_index)

        prob_pred_all = []

        for i in xrange(3):
            prob_pred_all.append(F.softmax(score_pred_all[i].view(-1, score_pred_all[i].size()[-1]), dim=1).view_as(score_pred_all[i]))

        coord_loss = coord_loss_var.data.cpu().numpy()
        conf_objloss = conf_objloss_var.data.cpu().numpy()
        conf_noobjloss = conf_noobjloss_var.data.cpu().numpy()
        cls_loss = cls_loss_var.data.cpu().numpy()
        test_loss = test_loss + coord_loss + conf_objloss + conf_noobjloss + cls_loss

        detect_time = _t['im_detect'].toc()

        _t['misc'].tic()

        ##### concatenating outputs at multiple scale feature maps after postprocessing operation
        bbox_pred = bbox_pred_all[0].data.cpu().numpy()
        conf_pred = conf_pred_all[0].data.cpu().numpy()
        prob_pred = prob_pred_all[0].data.cpu().numpy()
        targets1 = pool.map(partial(yolo_utils.postprocess, sample_batched['origin_im'][0].shape, thresh, size_index, cfg.args.non_nms),
                            ((bbox_pred[[b]], conf_pred[[b]], prob_pred[[b]])
                             for b in range(im_data.shape[0])))
        bbox_pred = bbox_pred_all[1].data.cpu().numpy()
        conf_pred = conf_pred_all[1].data.cpu().numpy()
        prob_pred = prob_pred_all[1].data.cpu().numpy()
        targets2 = pool.map(partial(yolo_utils.postprocess, sample_batched['origin_im'][0].shape, thresh, size_index, cfg.args.non_nms),
                            ((bbox_pred[[b]], conf_pred[[b]], prob_pred[[b]])
                             for b in range(im_data.shape[0])))
        bbox_pred = bbox_pred_all[2].data.cpu().numpy()
        conf_pred = conf_pred_all[2].data.cpu().numpy()
        prob_pred = prob_pred_all[2].data.cpu().numpy()
        targets3 = pool.map(partial(yolo_utils.postprocess, sample_batched['origin_im'][0].shape, thresh, size_index, cfg.args.non_nms),
                            ((bbox_pred[[b]], conf_pred[[b]], prob_pred[[b]])
                             for b in range(im_data.shape[0])))

        bboxes_batch = [np.concatenate((row1[0], row2[0], row3[0])) for row1, row2, row3 in zip(targets1, targets2, targets3)]
        scores_batch = [np.concatenate((row1[1], row2[1], row3[1])) for row1, row2, row3 in zip(targets1, targets2, targets3)]
        cls_inds_batch = [np.concatenate((row1[2], row2[2], row3[2])) for row1, row2, row3 in zip(targets1, targets2, targets3)]
        #########

        if cfg.args.non_nms:
            if args.confidence_plotlogs:
                targets = pool.map(yolo_utils.final_postprocess_withplotlogs,
                                   ((bboxes_batch[b], scores_batch[b], cls_inds_batch[b], sample_batched['origin_gt_boxes'][b])
                                    for b in range(im_data.shape[0])))
                bboxes_batch = [row[0] for row in targets]
                scores_batch = [row[1] for row in targets]
                scores_gt_batch = [row[2] for row in targets]
                cls_inds_batch = [row[3] for row in targets]

                for row in targets:
                    confidence_plotlogs['gt_conf'] += row[2].tolist()
                    confidence_plotlogs['backbone_conf'] += row[1].tolist()
            else:
                targets = pool.map(yolo_utils.final_postprocess,
                                   ((bboxes_batch[b], scores_batch[b], cls_inds_batch[b])
                                    for b in range(im_data.shape[0])))
                bboxes_batch = [row[0] for row in targets]
                scores_batch = [row[1] for row in targets]
                cls_inds_batch = [row[2] for row in targets]

        if cfg.args.pick_one:
            targets = pool.map(yolo_utils.pickone_postprocess,
                               ((bboxes_batch[b], scores_batch[b], cls_inds_batch[b])
                                for b in range(im_data.shape[0])))
            bboxes_batch = [row[0] for row in targets]
            scores_batch = [row[1] for row in targets]
            cls_inds_batch = [row[2] for row in targets]

        if cfg.args.topk_ransac:
            targets = pool.map(partial(yolo_utils.ransac_postprocess, cfg.cam_K, objpoints3D),
                               ((bboxes_batch[b], scores_batch[b], cls_inds_batch[b])
                                for b in range(im_data.shape[0])))
            bboxes_batch = [row[0] for row in targets]
            scores_batch = [row[1] for row in targets]
            cls_inds_batch = [row[2] for row in targets]

        if summary and i_batch == batch_num_summary:
            imnum_summary = randint(0, sample_batched['image'].shape[0] - 1)
            image = sample_batched['origin_im'][imnum_summary]
            vis_scaleid = randint(0, 2)
            bboxes_sum, scores_sum, cls_inds_sum = yolo_utils.postprocess(sample_batched['origin_im'][imnum_summary].shape, 0.5, size_index, False, (bbox_pred_all[vis_scaleid][imnum_summary:imnum_summary+1].data.cpu().numpy(),
                                                                                                                                                    conf_pred_all[vis_scaleid][imnum_summary:imnum_summary+1].data.cpu().numpy(),
                                                                                                                                                    prob_pred_all[vis_scaleid][imnum_summary:imnum_summary+1].data.cpu().numpy()))
            im2show = yolo_utils.draw_detection(image, bboxes_sum, scores_sum, cls_inds_sum, cfg, imdb._classes, 0.5, objpoints3D, corners_3d, vertices)
            summary.add_image('predict_' + imdb._image_set, cv2.cvtColor(im2show, cv2.COLOR_BGR2RGB), iter_count)

        for batch_id in range(im_data.shape[0]):
            if vis:
                det_im = yolo_utils.draw_detection(ori_im[batch_id].copy(),
                                                   bboxes_batch[batch_id],
                                                   scores_batch[batch_id],
                                                   cls_inds_batch[batch_id],
                                                   cfg, imdb._classes,
                                                   thresh, objpoints3D, corners_3d, vertices)

            bboxes = bboxes_batch[batch_id]
            scores = scores_batch[batch_id]
            cls_inds = cls_inds_batch[batch_id]

            for j in range(imdb._num_classes):
                inds = np.where(cls_inds == j)[0]
                if len(inds) == 0:
                    all_boxes[j][i_batch * batch_size + batch_id] = np.empty([0, 2 * args.num_detection_points + 1],
                                                                             dtype=np.float32)
                    continue
                ##
                bboxes_batch[batch_id][inds] = bboxes[inds]
                ##
                c_bboxes = bboxes[inds]
                c_scores = scores[inds]
                c_dets = np.hstack((c_bboxes,
                                    c_scores[:, np.newaxis])).astype(np.float32,
                                                                     copy=False)
                all_boxes[j][i_batch * batch_size + batch_id] = c_dets

            if vis:
                if args.num_detection_points > 9:
                    gt_image = yolo_utils.vis_corner_points(ori_im[batch_id].copy(),
                                                            np.reshape(sample_batched['origin_gt_boxes'][batch_id], (2, args.num_detection_points),
                                                                       order='F'), objpoints3D, vertices)
                    cuboid_gtimage = yolo_utils.vis_corner_cuboids(gt_image,
                                                                   np.reshape(sample_batched['origin_gt_boxes'][batch_id], (2, args.num_detection_points),
                                                                              order='F'), objpoints3D, corners_3d)
                else:
                    gt_image = yolo_utils.vis_corner_points(ori_im[batch_id].copy(),
                                                            np.reshape(sample_batched['origin_gt_boxes'][batch_id], (2, args.num_detection_points),
                                                                       order='F'), objpoints3D, vertices)
                    cuboid_gtimage = yolo_utils.vis_corner_cuboids(gt_image,
                                                                   np.reshape(sample_batched['origin_gt_boxes'][batch_id], (2, args.num_detection_points),
                                                                              order='F'))
                im2show = np.hstack((det_im, cuboid_gtimage))

                cv2.imwrite(test_output_dir + '/' + imdb._image_indexes[i_batch*batch_size + batch_id] + '.jpg', im2show)

        nms_time = _t['misc'].toc()

        if i_batch % 4 == 0:
            print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s'.format(i_batch, num_images/batch_size, detect_time, nms_time))  # noqa
            _t['im_detect'].clear()
            _t['misc'].clear()

    pool.close()
    pool.join()

    det_file = os.path.join(output_dir, 'detections.pkl')
    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    if args.confidence_plotlogs:
        with open('confidence_plotlogs/' + args.class_name + '_backbone_logs.yml', 'w') as outfile:
            yaml.dump(confidence_plotlogs, outfile, default_flow_style=False)

    print('Evaluating detections')
    accuracy_epoch, twod_dists, threed_dists = imdb.evaluate_detections(all_boxes, output_dir, verbose)
    return accuracy_epoch, twod_dists, threed_dists, test_loss/len(dataloader)


if __name__ == '__main__':
    args = cfg.args
    cv2.setNumThreads(1)

    print('Called with args:')
    print(args)
    # hyperparameters
    test_output_dir = os.path.join(cfg.test_output_dir)

    mkdir(test_output_dir, max_depth=3)

    train_output_dir = os.path.join(cfg.TRAIN_DIR, args.exp_name)

    test_output_dir_correct = test_output_dir + '/correct'

    if not os.path.exists(test_output_dir_correct):
        os.makedirs(test_output_dir_correct)

    if args.torch_cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    if args.min_train_loss:
        restore_file = os.path.join(train_output_dir, 'min_train_loss', 'model_second.pth.tar')
    if args.epoch_save:
        restore_file = os.path.join(train_output_dir, 'epoch_save', 'model_' + str(args.start_epoch) + '.pth.tar')

    checkpoint = torch.load(restore_file)
    print("=> loading checkpoint '{}' from '{}'".format(restore_file, checkpoint['epoch']))

    # Specify the number of workers
    kwargs = {'num_workers': 2, 'pin_memory': True}

    # data loader
    imdb = Dataset_class(cfg.imdb_test, args.class_name, cfg.DATA_DIR, yolo_utils.preprocess_test,
                         cfg.multi_scale_inp_size[args.image_size_index])
    dataloader = DataLoader(imdb, batch_size = args.batch_size, shuffle=False, **kwargs)

    net = Darknet()
    net.load_state_dict(checkpoint['state_dict'])
    net.cuda()
    print('load net succ...')

    accuracy_epoch, twod_dists, threed_dists, test_loss_epoch = test_net(net, imdb, dataloader, args, test_output_dir, args.image_size_index, args.batch_size, thresh=args.thresh, vis=args.vis)
    if args.save_results_bool:
        resultsdist_dir = cfg.TEST_DIR + '/' + cfg.args.datadirname + '_yolov3_topk' + str(args.topk)
        if not os.path.exists(resultsdist_dir):
            os.makedirs(resultsdist_dir)
        save_resultsdist(twod_dists, threed_dists, resultsdist_dir, args.class_name)
        save_results(cfg.args.datadirname, args.class_name, accuracy_epoch[0], accuracy_epoch[1], cfg, str(args.start_epoch), non_nms=args.non_nms, topk=args.topk, topk_ransac=args.topk_ransac)