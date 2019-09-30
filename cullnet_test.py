import os
import cv2
import numpy as np
import pickle, math, yaml
import torch
import torch.nn.functional as F

from random import randint
from functools import partial
from multiprocessing import Pool
from torch.utils.data import DataLoader
from yolo6d_darknet_variedpoints_v3 import Darknet
import utils.yolo6d as yolo_utils
import utils.network as net_utils
from utils.timer import Timer
from datasets.dataset import Dataset_class
from datasets.dataset_cullnet import CullNetDataset_class
import cfgs.config_yolo6d as cfg


if cfg.args.seg_cullnet:
    from yolo6d_cullnet import Cullnet


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
        if cfg.args.bias_correction == 'mean':
            result_filename = cfg.TEST_DIR + '/' + datadir_name + '_' + 'cull_meanbias.yml'
        if cfg.args.bias_correction == 'median':
            result_filename = cfg.TEST_DIR + '/' + datadir_name + '_' + 'cull_medianbias.yml'
        if cfg.args.bias_correction == 'mode':
            result_filename = cfg.TEST_DIR + '/' + datadir_name + '_' + 'cull_modebias.yml'

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


def produce_bbnet_patches_test(net1, sample_batched, objpoints3D, vertices, obj_diameter):
    net1.eval()
    batch = sample_batched
    non_nms = cfg.args.non_nms
    thresh = cfg.args.thresh
    size_index = 3

    pool = Pool(processes=1)

    if cfg.args.dataset_name=='LINEMOD':
        K = cfg.cam_K

    elif cfg.args.dataset_name=='YCB':
        K = cfg.cam_K1

    if cfg.args.seg_cullnet:

        with torch.no_grad():
            im_data = net_utils.tensor_to_variable(sample_batched['image'], is_cuda=True,
                                                   volatile=True).permute(0, 3, 1, 2)

            bbox_pred_all, conf_pred_all, score_pred_all = net1(im_data)
            prob_pred_all = []

            for i in xrange(3):
                prob_pred_all.append(F.softmax(score_pred_all[i].view(-1, score_pred_all[i].size()[-1]), dim=1).view_as(score_pred_all[i]))

            ##### concatenating outputs at multiple scale feature maps after postprocessing operation
            bbox_pred = bbox_pred_all[0].data.cpu().numpy()
            conf_pred = conf_pred_all[0].data.cpu().numpy()
            prob_pred = prob_pred_all[0].data.cpu().numpy()
            targets1 = pool.map(partial(yolo_utils.postprocess, batch['origin_im'][0].shape, thresh, size_index, non_nms),
                                ((bbox_pred[[b]], conf_pred[[b]], prob_pred[[b]])
                                 for b in range(im_data.shape[0])))
            bbox_pred = bbox_pred_all[1].data.cpu().numpy()
            conf_pred = conf_pred_all[1].data.cpu().numpy()
            prob_pred = prob_pred_all[1].data.cpu().numpy()
            targets2 = pool.map(partial(yolo_utils.postprocess, batch['origin_im'][0].shape, thresh, size_index, non_nms),
                                ((bbox_pred[[b]], conf_pred[[b]], prob_pred[[b]])
                                 for b in range(im_data.shape[0])))
            bbox_pred = bbox_pred_all[2].data.cpu().numpy()
            conf_pred = conf_pred_all[2].data.cpu().numpy()
            prob_pred = prob_pred_all[2].data.cpu().numpy()
            targets3 = pool.map(partial(yolo_utils.postprocess, batch['origin_im'][0].shape, thresh, size_index, non_nms),
                                ((bbox_pred[[b]], conf_pred[[b]], prob_pred[[b]])
                                 for b in range(im_data.shape[0])))
            bboxes_batch = [np.concatenate((row1[0], row2[0], row3[0])) for row1, row2, row3 in zip(targets1, targets2, targets3)]
            scores_batch = [np.concatenate((row1[1], row2[1], row3[1])) for row1, row2, row3 in zip(targets1, targets2, targets3)]
            cls_inds_batch = [np.concatenate((row1[2], row2[2], row3[2])) for row1, row2, row3 in zip(targets1, targets2, targets3)]
            ##########

            targets = pool.map(partial(yolo_utils.pose_proposals_nearby_test, objpoints3D, vertices, K, obj_diameter),
                               ((bboxes_batch[b], scores_batch[b], cls_inds_batch[b], batch['origin_gt_boxes'][b],
                                 batch['origin_im'][b]) for b in range(im_data.shape[0])))
            Rt_pr_patch = [row[0] for row in targets]
            corner_patch = [row[1] for row in targets]
            gtconf2d_patch = [row[2] for row in targets]
            gtconf3d_patch = [row[3] for row in targets]
            bboxes_batch = [row[4] for row in targets]

        pool.close()
        pool.join()

        return Rt_pr_patch, corner_patch, gtconf2d_patch, gtconf3d_patch, bboxes_batch


def test_net(net2, imdb, dataloader, args, output_dir, size_index, batch_size, objpoints3D, corners_3d, vertices,
             iter_count=0, thresh=0.5, vis=True, verbose=True, summary=None):

    net2.eval()
    cv2.setNumThreads(1)
    test_loss = 0

    pool = Pool(processes=2)

    num_images = imdb.__len__()

    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(imdb._num_classes)]

    # timers
    _t = {'network': Timer(), 'postpro': Timer()}
    # corners3d = yolo_utils.threed_corners(dataset='linemod')
    batch_num_summary = randint(0, num_images/batch_size - 1)

    if args.confidence_plotlogs:
        confidence_plotlogs = {}
        confidence_plotlogs['gt_conf'] = []
        confidence_plotlogs['cullnet_conf'] = []

    network_time = 0
    postpro_time = 0

    for i_batch, sample_batched in enumerate(dataloader):

        _t['network'].tic()
        rgb_patches, gt_2dconfs, gt_3dconfs, bboxes = sample_batched['pose_proposals']

        ori_im = np.array(sample_batched['origin_im'])

        if cfg.args.cullnet_confidence == 'conf2d':
            gt_confs = gt_2dconfs
        elif cfg.args.cullnet_confidence == 'conf3d':
            gt_confs = gt_3dconfs

        with torch.no_grad():
            if cfg.args.seg_cullnet:

                confidence_new_batch = []

                rgb_patch_np = np.array(rgb_patches)
                gtconf_patch_np = np.array(gt_confs)
                bboxes_batch = bboxes

                current_batch_size = rgb_patch_np.shape[0]

                subnetwork_batchsize = 128
                if cfg.args.cullnet_type == 'vgg19_bn':
                    subnetwork_batchsize = 128
                if cfg.args.cullnet_type == 'allconvnet':
                    subnetwork_batchsize = 64
                if cfg.args.cullnet_type == 'allconvnet_small':
                    subnetwork_batchsize = 256
                if cfg.args.cullnet_type == 'resnet18':
                    subnetwork_batchsize = 512
                if cfg.args.cullnet_type == 'resnet18_gn' or cfg.args.cullnet_type == 'resnet18concat_gn':
                    subnetwork_batchsize = 320
                if cfg.args.cullnet_type == 'resnet50' or cfg.args.cullnet_type == 'resnet50_gn' or cfg.args.cullnet_type == 'resnet50concat_gn':
                    subnetwork_batchsize = 160

                if cfg.args.sub_bs_test is not None:
                    subnetwork_batchsize = int(cfg.args.sub_bs_test)

                partition_size = (current_batch_size*cfg.k_proposals_test) / subnetwork_batchsize
                subnetwork_numimages = int(math.ceil(subnetwork_batchsize/cfg.k_proposals_test))

                for i in range(partition_size):
                    rgb_patches_var = net_utils.np_to_variable(rgb_patch_np[i*subnetwork_numimages:(i+1)*subnetwork_numimages],
                                                               is_cuda=True,
                                                               volatile=True).permute(0, 1, 4, 2, 3)

                    if cfg.args.cullnet_inconf=='concat':
                        confidence_new = net2(rgb_patches_var.view(-1, 4, cfg.args.cullnet_input, cfg.args.cullnet_input))
                    else:
                        confidence_new = net2(rgb_patches_var.view(-1, 3, cfg.args.cullnet_input, cfg.args.cullnet_input))

                    gtconf_patch = gtconf_patch_np[i*subnetwork_numimages:(i+1)*subnetwork_numimages].reshape(-1, 1)

                    if args.confidence_plotlogs:
                        confidence_plotlogs['gt_conf'] += gtconf_patch[:, 0].tolist()
                        confidence_plotlogs['cullnet_conf'] += confidence_new[:, 0].tolist()

                    if args.mGPUs:
                        conf_loss_var = net2.module.loss(confidence_new, gtconf_patch)
                    else:
                        conf_loss_var = net2.loss(confidence_new, gtconf_patch)

                    confidence_new_np = confidence_new.data.cpu().numpy()
                    ### debugging purpose
                    # confidence_new_np = gtconf_patch

                    confidence_new_batch.append(confidence_new_np)
                    # bbox_pred.register_hook(extract)

                    conf_loss_np = conf_loss_var.data.cpu().numpy()
                    test_loss += conf_loss_np

                confidence_new_batch = np.array(confidence_new_batch)

                left_overpatches = (current_batch_size*cfg.k_proposals_test) % subnetwork_batchsize

                confidence_new_batch = confidence_new_batch.reshape(partition_size * subnetwork_numimages, cfg.k_proposals_test)

                if i_batch == len(dataloader) - 1 and left_overpatches > 0:
                    rgb_patches_var = net_utils.np_to_variable(rgb_patch_np[partition_size*subnetwork_numimages:],
                                                               is_cuda=True,
                                                               volatile=True).permute(0, 1, 4, 2, 3)

                    if cfg.args.cullnet_inconf=='concat':
                        confidence_new = net2(rgb_patches_var.view(-1, 4, cfg.args.cullnet_input, cfg.args.cullnet_input))
                    else:
                        confidence_new = net2(rgb_patches_var.view(-1, 3, cfg.args.cullnet_input, cfg.args.cullnet_input))

                    gtconf_patch = gtconf_patch_np[partition_size*subnetwork_numimages:].reshape(-1, 1)

                    if args.confidence_plotlogs:
                        confidence_plotlogs['gt_conf'] += gtconf_patch[:, 0].tolist()
                        confidence_plotlogs['cullnet_conf'] += confidence_new[:, 0].tolist()

                    if args.mGPUs:
                        conf_loss_var = net2.module.loss(confidence_new, gtconf_patch)
                    else:
                        conf_loss_var = net2.loss(confidence_new, gtconf_patch)

                    confidence_new_np = confidence_new.data.cpu().numpy()

                    ### debugging purpose
                    # confidence_new_np = gtconf_patch
                    confidence_new_np = confidence_new_np.reshape(-1, cfg.k_proposals_test)
                    confidence_new_batch = np.concatenate((confidence_new_batch, confidence_new_np), 0)
                    # bbox_pred.register_hook(extract)
                    conf_loss_np = conf_loss_var.data.cpu().numpy()
                    test_loss += conf_loss_np

        network_time += _t['network'].toc()

        _t['postpro'].tic()

        if cfg.args.seg_cullnet:
            targets = pool.map(partial(yolo_utils.seg_cullnet_postprocess, sample_batched['origin_im'][0].shape, size_index),
                               ((bboxes_batch[b], confidence_new_batch[b])
                                for b in range(rgb_patch_np.shape[0])))

            bboxes_batch = [row[0] for row in targets]
            scores_batch = [row[1] for row in targets]
            cls_inds_batch = [row[2] for row in targets]

        ###########
        # targets = pool.map(yolo_utils.final_postprocess,
        #                     ((bboxes_batch[b], scores_batch[b], cls_inds_batch[b])
        #                      for b in range(im_data.shape[0])))
        # bboxes_batch = [row[0] for row in targets]
        # scores_batch = [row[1] for row in targets]
        # cls_inds_batch = [row[2] for row in targets]

        if summary and i_batch == batch_num_summary:
            imnum_summary = randint(0, sample_batched['image'].shape[0] - 1)
            image = sample_batched['origin_im'][imnum_summary]
            # bboxes_sum, scores_sum, cls_inds_sum = yolo_utils.seg_cullnet_postprocess(sample_batched['origin_im'][imnum_summary].shape, size_index, (bboxes_batch[imnum_summary],
            #                                                                   confidence_new_batch[imnum_summary]))
            im2show = yolo_utils.draw_detection(image, bboxes_batch[imnum_summary], scores_batch[imnum_summary],
                                                cls_inds_batch[imnum_summary], cfg, imdb._classes, 0.5, objpoints3D,
                                                corners_3d, vertices)
            summary.add_image('predict_' + imdb._image_set, cv2.cvtColor(im2show, cv2.COLOR_BGR2RGB), iter_count)

        for batch_id in range(rgb_patch_np.shape[0]):
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
                # bboxes[inds] = yolo_utils.refine_2dboxes(bboxes[inds], corners3d[j])
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
                    cuboid_gtimage = yolo_utils.vis_corner_cuboids(ori_im[batch_id].copy(),
                                                                   np.reshape(sample_batched['origin_gt_boxes'][batch_id], (2, args.num_detection_points),
                                                                              order='F'))
                im2show = np.hstack((det_im, cuboid_gtimage, ori_im[batch_id].copy()))

                cv2.imwrite(test_output_dir + '/' + imdb._image_indexes[i_batch*batch_size + batch_id] + '.jpg', im2show)
                # cv2.imshow('test', im2show)
                # cv2.waitKey(0)

        postpro_time += _t['postpro'].toc()

    # print('Culling network time: {:.3f}s ,, Postprocessing time: {:.3f}s'.format(network_time, postpro_time))
    # print('Total Images: {:d}'.format(imdb.__len__()))

    pool.close()
    pool.join()

    det_file = os.path.join(output_dir, 'detections.pkl')
    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    if args.confidence_plotlogs:
        with open('confidence_plotlogs/' + args.class_name + '_cullnet_logs.yml', 'w') as outfile:
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

    backbone_model_dir = os.path.join(cfg.TRAIN_DIR, args.exp_name)

    net1_cachedir_test = backbone_model_dir + '/net1_output_topk' + str(args.topk) + '_nearby' + str(args.nearby_test) + '/test'

    if args.datadirname=='singleobject_occtest':
        net1_cachedir_test = backbone_model_dir + '/net1_output_topk' + str(args.topk) + '_nearby' + str(args.nearby_test) + '_occlusion/test'

    if cfg.imdb_test=='linemod_testerror':
        net1_cachedir_test = backbone_model_dir + '/net1_output_topk' + str(args.topk) + '_nearby' + str(args.nearby_test) + '_linemodtrain_ori/test'

    test_output_dir_correct = test_output_dir + '/correct'

    if not os.path.exists(test_output_dir_correct):
        os.makedirs(test_output_dir_correct)
    if not os.path.exists(net1_cachedir_test):
        os.makedirs(net1_cachedir_test)

    if args.torch_cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    if cfg.args.cull_net:
        model_cullnet_filename = 'model_cullnet.pth.tar'
    if cfg.args.seg_cullnet:
        model_cullnet_filename = 'model_seg_cullnet.pth.tar'
        if args.epoch_save:
            model_cullnet_filename = 'model_seg_cullnet' + str(args.start_epoch) + '.pth.tar'

    if cfg.args.cullnet_confidence == 'conf2d':
        expname_cullnet = args.exp_name_cullnet + '_' + cfg.args.cullnet_confidence
    if cfg.args.cullnet_confidence == 'conf3d':
        expname_cullnet = args.exp_name_cullnet + '_' + cfg.args.cullnet_confidence

    if args.min_train_loss:
        restore_file1 = os.path.join(backbone_model_dir, 'epoch_save', 'model_99.pth.tar')
        restore_file2 = os.path.join(backbone_model_dir, 'cullnet', expname_cullnet, 'min_train_loss', model_cullnet_filename)

    if args.epoch_save:
        restore_file1 = os.path.join(backbone_model_dir, 'epoch_save', 'model_99.pth.tar')
        restore_file2 = os.path.join(backbone_model_dir, 'cullnet', expname_cullnet, 'epoch_save', model_cullnet_filename)

    checkpoint1 = torch.load(restore_file1)
    checkpoint2 = torch.load(restore_file2)

    print("=> loading checkpoint '{}' from '{}' and '{}'".format(restore_file1, checkpoint1['epoch'], checkpoint2['epoch']))

    # Specify the number of workers
    kwargs = {'num_workers': 4, 'pin_memory': True}

    # data loader
    imdb = Dataset_class(cfg.imdb_test, args.class_name, cfg.DATA_DIR, yolo_utils.preprocess_test,
                         cfg.multi_scale_inp_size[args.image_size_index])
    dataloader = DataLoader(imdb, batch_size = args.batch_size, shuffle=False, **kwargs)

    net1 = Darknet()
    net2 = Cullnet()
    net1.load_state_dict(checkpoint1['state_dict'])
    net1.cuda()
    net2.load_state_dict(checkpoint2['state_dict'])
    net2.cuda()

    print('load net succ...')

    if cfg.args.num_detection_points == 9:
        objpoints3D = yolo_utils.threed_corners(args.class_name)
    else:
        objpoints3D = yolo_utils.threed_correspondences(args.class_name)

    corners_3d = yolo_utils.threed_corners(args.class_name)
    vertices = yolo_utils.threed_vertices(args.class_name)

    if cfg.args.dataset_name == 'LINEMOD':
        with open("cfgs/diameter_linemod.yml", 'r') as stream:
            diam_data = yaml.load(stream)

    obj_diameter = diam_data[args.class_name]

    imdb_cullnettest = CullNetDataset_class(cfg.imdb_test, args.class_name, cfg.DATA_DIR, yolo_utils.preprocess_test_cullnet, None)
    dataloader_cullnettest = DataLoader(imdb_cullnettest, batch_size=args.batch_size, shuffle=False, **kwargs)

    backbone_net_time = 0
    _t = Timer()

    if not(len(os.listdir(net1_cachedir_test)) == len(imdb)):
        for i_batch, sample_batched in enumerate(dataloader):
            im_batch = sample_batched['origin_im']
            _t.tic()
            # batch
            Rt_pr_patches, corner_patches, gt_2dconfs, gt_3dconfs, bboxes = produce_bbnet_patches_test(net1, sample_batched,
                                                                                                       objpoints3D, vertices, obj_diameter)

            backbone_net_time += _t.toc()

            for b_id in xrange(len(im_batch)):
                net1_out = (Rt_pr_patches[b_id], corner_patches[b_id], gt_2dconfs[b_id], gt_3dconfs[b_id], bboxes[b_id])
                filename = imdb._image_indexes[i_batch*args.batch_size + b_id]
                cache_file = net1_cachedir_test + '/' + filename + '.pkl'
                with open(cache_file, 'wb') as fid:
                    pickle.dump(net1_out, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt net1_output to {}'.format(net1_cachedir_test))

    accuracy_epoch, twod_dists, threed_dists, test_loss_epoch = test_net(net2, imdb_cullnettest, dataloader_cullnettest, args, test_output_dir, args.image_size_index, args.batch_size, objpoints3D, corners_3d, vertices, thresh=args.thresh, vis=args.vis)

    _t.clear()

    if args.save_results_bool:
        resultsdist_dir = cfg.TEST_DIR + '/' + cfg.args.datadirname + '_topk' + str(args.topk) + '_nearby' + str(args.nearby_test) + '_cullnetyolov3'

        if not os.path.exists(resultsdist_dir):
            os.makedirs(resultsdist_dir)

        save_resultsdist(twod_dists, threed_dists, resultsdist_dir, args.class_name)
        save_results(cfg.args.datadirname + '_topk' + str(args.topk) + '_nearby' + str(args.nearby_test) , args.class_name, accuracy_epoch[0], accuracy_epoch[1], cfg, str(args.start_epoch), non_nms=args.non_nms, topk=args.topk, topk_ransac=args.topk_ransac)
