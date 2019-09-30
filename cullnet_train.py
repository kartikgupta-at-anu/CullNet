import os
import torch
import datetime
import torch.nn.functional as F
import numpy as np
import cv2, yaml, time, errno, math
from torch.utils.data import DataLoader
from yolo6d_darknet_variedpoints_v3 import Darknet
from cullnet_test import produce_bbnet_patches_test
from datasets.dataset import Dataset_class
from datasets.dataset_cullnet import CullNetDataset_class
from tensorboardX import SummaryWriter
from functools import partial
from multiprocessing import Pool
import utils.yolo6d as yolo_utils
import utils.network as net_utils
from utils.timer import Timer
from random import randint
from utils.collate_fns_self import *
import pickle

if cfg.args.seg_cullnet:
    from yolo6d_cullnet import Cullnet


def produce_bbnet_patches(net1, sample_batched, objpoints3D, vertices, obj_diameter):
    net1.eval()
    batch = sample_batched[0]
    size_index = sample_batched[1]
    im = batch['image']
    gt_boxes = batch['gt_boxes']
    gt_classes = batch['gt_classes']
    gt_RT = batch['gt_RT']
    non_nms = cfg.args.non_nms
    thresh = cfg.args.thresh

    if cfg.args.dataset_name=='LINEMOD':
        K = cfg.cam_K

    elif cfg.args.dataset_name=='YCB':
        K = cfg.cam_K1

    pool = Pool(processes=4)

    if cfg.args.seg_cullnet:
        with torch.no_grad():
            im_data = net_utils.np_to_variable(im,
                                               is_cuda=True,
                                               volatile=False).permute(0, 3, 1, 2)

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

            targets = pool.map(partial(yolo_utils.pose_proposals, objpoints3D, vertices, K, obj_diameter),
                               ((bboxes_batch[b], scores_batch[b], cls_inds_batch[b], batch['origin_gtboxes'][b],
                                 batch['origin_im'][b]) for b in range(im_data.shape[0])))
            Rt_pr_patch = [row[0] for row in targets]
            corner_patch = [row[1] for row in targets]
            gtconf2d_patch = [row[2] for row in targets]
            gtconf3d_patch = [row[3] for row in targets]
            bboxes_batch = [row[4] for row in targets]

        pool.close()
        pool.join()

        return Rt_pr_patch, corner_patch, gtconf2d_patch, gtconf3d_patch, bboxes_batch


def train_batch(net2, rgb_patch, gtconf_patch, train_loss_epoch, conf_loss_epoch):

    net2.train()

    if cfg.args.seg_cullnet:
        confidence_new_batch = []

        rgb_patch_np = np.array(rgb_patch)
        gtconf_patch_np = np.array(gtconf_patch)

        current_batch_size = rgb_patch_np.shape[0]

        subnetwork_batchsize = 80
        if cfg.args.cullnet_type == 'vgg19_bn':
            subnetwork_batchsize = 32
        if cfg.args.cullnet_type == 'allconvnet':
            subnetwork_batchsize = 32
        if cfg.args.cullnet_type == 'allconvnet_small':
            subnetwork_batchsize = 128
        if cfg.args.cullnet_type == 'resnet18':
            subnetwork_batchsize = 512
        if cfg.args.cullnet_type == 'resnet18_gn' or cfg.args.cullnet_type == 'resnet18concat_gn':
            subnetwork_batchsize = 80
        if cfg.args.cullnet_type == 'resnet50concat_gn' or cfg.args.cullnet_type == 'resnet50_gn':
            subnetwork_batchsize = 80

        if not cfg.args.sub_bs == 80:
            subnetwork_batchsize = cfg.args.sub_bs

        ### partition_size is number of batches for the network2 using a batched output of network1
        partition_size = (current_batch_size*cfg.args.k_proposals) / subnetwork_batchsize
        subnetwork_numimages = int(math.ceil(subnetwork_batchsize/cfg.args.k_proposals))

        conf_loss_np_subnetwrk = 0

        for i in range(partition_size):
            if subnetwork_batchsize < cfg.args.k_proposals:
                b_id = (i * subnetwork_batchsize)/cfg.args.k_proposals
                p_id = (i * subnetwork_batchsize)%cfg.args.k_proposals
                rgb_patches_var = net_utils.np_to_variable(rgb_patch_np[b_id:b_id+1, p_id:
                                                                                     p_id + subnetwork_batchsize],
                                                           is_cuda=True,
                                                           volatile=True).permute(0, 1, 4, 2, 3)
                gtconf_patch = gtconf_patch_np[b_id:b_id+1, p_id:
                                                            p_id + subnetwork_batchsize].reshape(-1, 1)
            else:
                rgb_patches_var = net_utils.np_to_variable(rgb_patch_np[i*subnetwork_numimages:(i+1)*subnetwork_numimages],
                                                       is_cuda=True,
                                                       volatile=True).permute(0, 1, 4, 2, 3)
                gtconf_patch = gtconf_patch_np[i*subnetwork_numimages:(i+1)*subnetwork_numimages].reshape(-1, 1)

            if cfg.args.cullnet_inconf=='concat':
                confidence_new = net2(rgb_patches_var.view(-1, 4, cfg.args.cullnet_input, cfg.args.cullnet_input))
            else:
                confidence_new = net2(rgb_patches_var.view(-1, 3, cfg.args.cullnet_input, cfg.args.cullnet_input))

            if args.mGPUs:
                conf_loss_var = net2.module.loss(confidence_new, gtconf_patch)
            else:
                conf_loss_var = net2.loss(confidence_new, gtconf_patch)

            confidence_new_np = confidence_new.data.cpu().numpy()

            loss_var = conf_loss_var
            optimizer.zero_grad()
            loss_var.backward()

            optimizer.step()

            confidence_new_batch.append(confidence_new_np)

            conf_loss_np = conf_loss_var.data.cpu().numpy()
            conf_loss_np_subnetwrk += conf_loss_np

        confidence_new_batch = np.array(confidence_new_batch)

        left_overpatches = (current_batch_size*cfg.args.k_proposals) % subnetwork_batchsize

        if subnetwork_batchsize < cfg.args.k_proposals:
            confidence_new_batch = confidence_new_batch.reshape(int(partition_size * (float(subnetwork_batchsize)/
                                                                    cfg.args.k_proposals)), cfg.args.k_proposals)
        else:
            confidence_new_batch = confidence_new_batch.reshape(partition_size * subnetwork_numimages, cfg.args.k_proposals)

        if left_overpatches > 0:
            rgb_patches_var = net_utils.np_to_variable(rgb_patch_np[partition_size*subnetwork_numimages:],
                                                       is_cuda=True,
                                                       volatile=True).permute(0, 1, 4, 2, 3)
            if cfg.args.cullnet_inconf=='concat':
                confidence_new = net2(rgb_patches_var.view(-1, 4, cfg.args.cullnet_input, cfg.args.cullnet_input))
            else:
                confidence_new = net2(rgb_patches_var.view(-1, 3, cfg.args.cullnet_input, cfg.args.cullnet_input))

            gtconf_patch = gtconf_patch_np[partition_size*subnetwork_numimages:].reshape(-1, 1)

            if args.mGPUs:
                conf_loss_var = net2.module.loss(confidence_new, gtconf_patch)
            else:
                conf_loss_var = net2.loss(confidence_new, gtconf_patch)

            loss_var = conf_loss_var
            optimizer.zero_grad()
            loss_var.backward()
            optimizer.step()

            confidence_new_np = confidence_new.data.cpu().numpy()
            confidence_new_np = confidence_new_np.reshape(-1, cfg.args.k_proposals)
            confidence_new_batch = np.concatenate((confidence_new_batch, confidence_new_np), 0)
            conf_loss_np = conf_loss_var.data.cpu().numpy()
            conf_loss_np_subnetwrk += conf_loss_np

        train_loss_np = conf_loss_np_subnetwrk

        train_loss_epoch += train_loss_np
        conf_loss_epoch += conf_loss_np_subnetwrk

        return confidence_new_batch, conf_loss_np_subnetwrk, train_loss_np, train_loss_epoch, conf_loss_epoch


def save_checkpoint(state, second_training, save_dir, epoch_save=False):
    if not epoch_save:
        if second_training:
            model_file = os.path.join(save_dir, 'model_seg_cullnet_second.pth.tar')
        else:
            model_file = os.path.join(save_dir, 'model_seg_cullnet.pth.tar')
    else:
        model_file = os.path.join(save_dir, 'model_seg_cullnet' + str(state['epoch']) + '.pth.tar')

    torch.save(state, model_file)


if __name__=='__main__':
    args = cfg.args
    cv2.setNumThreads(1)

    print('Called with args:')
    print(args)

    backbone_model_dir = os.path.join(cfg.TRAIN_DIR, args.exp_name)
    train_output_dir = os.path.join(cfg.TRAIN_DIR, args.exp_name, 'cullnet', args.exp_name_cullnet + '_' + args.cullnet_confidence)
    mintrainloss_dir = os.path.join(cfg.TRAIN_DIR, args.exp_name, 'cullnet', args.exp_name_cullnet + '_' + args.cullnet_confidence, 'min_train_loss')
    epochsave_dir = os.path.join(cfg.TRAIN_DIR, args.exp_name, 'cullnet', args.exp_name_cullnet + '_' + args.cullnet_confidence, 'epoch_save')

    net1_cachedir_train = backbone_model_dir + '/net1_output/train'

    net1_cachedir_test = backbone_model_dir + '/net1_output_topk' + str(args.topk) + '_nearby' + str(args.nearby_test) + '/test'

    if not os.path.exists(net1_cachedir_train):
        os.makedirs(net1_cachedir_train)
    if not os.path.exists(net1_cachedir_test):
        os.makedirs(net1_cachedir_test)

    if not os.path.exists(cfg.val_output_dir):
        os.makedirs(cfg.val_output_dir)
    if not os.path.exists(epochsave_dir):
        os.makedirs(epochsave_dir)
    if not os.path.exists(mintrainloss_dir):
        os.makedirs(mintrainloss_dir)

    if cfg.args.num_detection_points == 9:
        objpoints3D = yolo_utils.threed_corners(args.class_name)
    else:
        objpoints3D = yolo_utils.threed_correspondences(args.class_name)

    if cfg.args.dataset_name == 'LINEMOD':
        with open("cfgs/diameter_linemod.yml", 'r') as stream:
            diam_data = yaml.load(stream)

    obj_diameter = diam_data[args.class_name]

    corners_3d = yolo_utils.threed_corners(args.class_name)
    vertices = yolo_utils.threed_vertices(args.class_name)

    if args.torch_cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    # Specify the number of workers
    kwargs = {'num_workers': 2, 'pin_memory': True}

    # Dataloader
    imdb_train = Dataset_class(cfg.imdb_traincullnet, args.class_name, cfg.DATA_DIR, yolo_utils.preprocess_train, None)
    dataloader_train = DataLoader(imdb_train, batch_size=args.batch_size, shuffle=False, collate_fn= my_collate, **kwargs)

    imdb_test = Dataset_class(cfg.imdb_test, args.class_name, cfg.DATA_DIR, yolo_utils.preprocess_test,
                              cfg.multi_scale_inp_size[3])
    dataloader_test = DataLoader(imdb_test, batch_size=args.batch_size, shuffle=False, **kwargs)

    batch_per_epoch = len(dataloader_train)
    print('load data sucessful...')

    train_loss_epoch = 0
    coord_loss_epoch = 0
    conf_loss_epoch = 0
    min_train_loss = float("inf")

    # Model load
    net1 = Darknet()
    net2 = Cullnet()

    lr = args.lr

    print "Loading the pretrained model"
    if args.min_train_loss:
        restore_file = os.path.join(backbone_model_dir, 'min_train_loss', 'model_second.pth.tar')
    if args.epoch_save:
        restore_file = os.path.join(backbone_model_dir, 'epoch_save', 'model_' + str(args.start_epoch) + '.pth.tar')

    checkpoint = torch.load(restore_file)
    print("=> loading checkpoint '{}' from '{}'".format(restore_file, checkpoint['epoch']))
    net1.load_state_dict(checkpoint['state_dict'])
    net1.cuda()
    net2.cuda()
    print('load net succ...')
    if not cfg.args.ADAM_optim:
        optimizer = torch.optim.SGD(net2.parameters(), lr=lr, momentum=cfg.momentum,
                                    weight_decay=cfg.weight_decay)
    else:
        optimizer = torch.optim.Adam(net2.parameters(), lr=args.lr, weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    start_epoch = checkpoint['epoch']
    max_epoch = args.max_epochs
    min_train_loss = float("inf")
    min_error2d = float("inf")
    min_error3d = float("inf")

    if args.mGPUs:
        net2 = torch.nn.DataParallel(net2)

    # Tensorboard
    use_tensorboard = args.use_tensorboard and SummaryWriter is not None
    if use_tensorboard:
        summary_writer = SummaryWriter(os.path.join(cfg.TRAIN_DIR, 'runs', args.exp_name, 'cullnet', args.exp_name_cullnet +
                                                    '_' + args.cullnet_confidence,
                                                    datetime.datetime.now().strftime('%b%d_%H-%M-%S')))
    else:
        summary_writer = None

    t = Timer()

    ### Getting pose proposals and saving it to disk
    if not(len(os.listdir(net1_cachedir_train)) == len(imdb_train)):
        for i_batch, sample_batched in enumerate(dataloader_train):
            im_batch = sample_batched[0]['origin_im']
            # batch
            Rt_pr_patches, corner_patches, gt_2dconfs, gt_3dconfs, bboxes = produce_bbnet_patches(net1, sample_batched,
                                                                                                  objpoints3D, vertices, obj_diameter)

            for b_id in xrange(len(im_batch)):
                net1_out = (Rt_pr_patches[b_id], corner_patches[b_id], gt_2dconfs[b_id], gt_3dconfs[b_id], bboxes[b_id])
                filename = imdb_train._image_indexes[i_batch*args.batch_size + b_id]
                cache_file = net1_cachedir_train + '/' + filename + '.pkl'
                with open(cache_file, 'wb') as fid:
                    pickle.dump(net1_out, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt net1_output to {}'.format(net1_cachedir_train))

    if not(len(os.listdir(net1_cachedir_test)) == len(imdb_test)):
        for i_batch, sample_batched in enumerate(dataloader_test):
            im_batch = sample_batched['origin_im']
            # batch
            Rt_pr_patches, corner_patches, gt_2dconfs, gt_3dconfs, bboxes = produce_bbnet_patches_test(net1, sample_batched,
                                                                                                  objpoints3D, vertices, obj_diameter)

            for b_id in xrange(len(im_batch)):
                net1_out = (Rt_pr_patches[b_id], corner_patches[b_id], gt_2dconfs[b_id], gt_3dconfs[b_id], bboxes[b_id])
                filename = imdb_test._image_indexes[i_batch*args.batch_size + b_id]
                cache_file = net1_cachedir_test + '/' + filename + '.pkl'
                with open(cache_file, 'wb') as fid:
                    pickle.dump(net1_out, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt net1_output to {}'.format(net1_cachedir_test))

    # Specify the number of workers
    kwargs = {'num_workers': 4, 'pin_memory': True}

    # data loader
    imdb_cullnettrain = CullNetDataset_class(cfg.imdb_traincullnet, args.class_name, cfg.DATA_DIR, yolo_utils.preprocess_train_cullnet, None)
    dataloader_cullnettrain = DataLoader(imdb_cullnettrain, batch_size=args.batch_size, shuffle=False, collate_fn= my_collate_cullnet, **kwargs)

    # Training
    for epoch in range(start_epoch, max_epoch):

        batch_num_summary = randint(0, len(dataloader_cullnettrain) - 1)

        for i_batch, sample_batched in enumerate(dataloader_cullnettrain):
            t.tic()

            rgb_patches = sample_batched[0]['rgb_patches']
            gt_2dconfs = sample_batched[0]['gt_2dconfs']
            gt_3dconfs = sample_batched[0]['gt_3dconfs']
            bboxes = sample_batched[0]['bboxes']

            size_index = sample_batched[1]
            im_batch = sample_batched[0]['origin_im']
            gt_boxes = sample_batched[0]['gt_boxes']
            gt_classes = sample_batched[0]['gt_classes']

            if cfg.args.cullnet_confidence == 'conf2d':
                gt_confs = gt_2dconfs
            elif cfg.args.cullnet_confidence == 'conf3d':
                gt_confs = gt_3dconfs

            confidence_new_batch, conf_loss, \
            train_loss, train_loss_epoch, conf_loss_epoch = train_batch(net2, rgb_patches,
                                                                        gt_confs, train_loss_epoch, conf_loss_epoch)

            duration = t.toc()

            if summary_writer and i_batch==batch_num_summary:
                # plot results
                imnum_summary = randint(0, len(im_batch) - 1)
                image = im_batch[imnum_summary]
                keypoints = bboxes[imnum_summary]
                grid_index = confidence_new_batch[imnum_summary]
                if cfg.args.seg_cullnet:
                    bboxes, scores, cls_inds = yolo_utils.seg_cullnet_postprocess(image.shape, size_index,
                                                                                  (keypoints, grid_index))

                im2show = yolo_utils.draw_detection(image, bboxes, scores, cls_inds, cfg, imdb_train._classes, 0.5, objpoints3D, corners_3d, vertices)
                summary_writer.add_image('predict_train', cv2.cvtColor(im2show, cv2.COLOR_BGR2RGB), epoch)

            iter_count = (epoch + float(i_batch)/batch_per_epoch) * 1000

            if i_batch % cfg.log_interval == 0:

                print(('epoch %d[%d/%d], loss: %.3f, conf_loss: %.3f'
                       '(%.2f s/batch, rest:%s)' %
                       (epoch, i_batch, batch_per_epoch, train_loss, conf_loss, duration,
                        str(datetime.timedelta(seconds=int((batch_per_epoch - i_batch) * duration))))))  # noqa

                if summary_writer:
                    summary_writer.add_scalar('loss_train', train_loss, iter_count)
                    summary_writer.add_scalar('loss_conf', conf_loss, iter_count)
                    summary_writer.add_scalar('learning_rate', lr, iter_count)

                t.clear()
                print("image_size {}".format(cfg.multi_scale_inp_size[size_index]))

        #### Saving checkpoints and results
        if epoch > 0:
            if summary_writer:
                summary_writer.add_scalar('train_loss_epoch', train_loss_epoch / batch_per_epoch, epoch)
                summary_writer.add_scalar('conf_loss_epoch', conf_loss_epoch / batch_per_epoch, epoch)

            if epoch in cfg.save_epochs:
                save_dir = os.path.join(train_output_dir, 'epoch_save')
                if args.mGPUs:
                    save_checkpoint({
                        'epoch': epoch,
                        'min_train_loss': train_loss_epoch,
                        'state_dict': net2.module.state_dict(),
                        'optimizer': optimizer.state_dict()}, args.second_training, save_dir=save_dir, epoch_save=True)
                else:
                    save_checkpoint({
                        'epoch': epoch,
                        'min_train_loss': train_loss_epoch,
                        'state_dict': net2.state_dict(),
                        'optimizer': optimizer.state_dict()}, args.second_training, save_dir=save_dir, epoch_save=True)
                print('saved checkpoint for epoch {:d}'.format(epoch))

            if train_loss_epoch < min_train_loss:
                min_train_loss = train_loss_epoch
                save_dir = os.path.join(train_output_dir, 'min_train_loss')
                if args.mGPUs:
                    save_checkpoint({
                        'epoch': epoch,
                        'min_train_loss': min_train_loss,
                        'state_dict': net2.module.state_dict(),
                        'optimizer': optimizer.state_dict()}, args.second_training, save_dir=save_dir)
                else:
                    save_checkpoint({
                        'epoch': epoch,
                        'min_train_loss': min_train_loss,
                        'state_dict': net2.state_dict(),
                        'optimizer': optimizer.state_dict()}, args.second_training, save_dir=save_dir)
                print('saved checkpoint for minimum train loss')

            train_loss_epoch = 0
            conf_loss_epoch = 0

        if not cfg.args.ADAM_optim:
            if epoch in cfg.lr_decay_epochs:
                lr *= cfg.lr_decay
                optimizer = torch.optim.SGD(net2.parameters(), lr=lr,
                                            momentum=cfg.momentum,
                                            weight_decay=cfg.weight_decay)
        else:
            scheduler.step()


