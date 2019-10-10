import os
import torch
import datetime
import torch.nn.functional as F
import numpy as np
import cv2, yaml, time, errno
from torch.utils.data import DataLoader
from yolo6d_v3_darknet_def import Darknet
from datasets.dataset import Dataset_class
from tensorboardX import SummaryWriter
import utils.yolo6d as yolo_utils
import utils.network as net_utils
from utils.timer import Timer
import cfgs.config_yolo6d as cfg
from random import randint


def train_batch(net, sample_batched, train_loss_epoch, coord_loss_epoch, conf_loss_epoch):

    net.train()
    batch = sample_batched[0]
    size_index = sample_batched[1]
    im = batch['image']
    gt_boxes = batch['gt_boxes']
    gt_classes = batch['gt_classes']
    gt_RT = batch['gt_RT']
    dontcare = batch['dontcare']

    # forward
    im_data = net_utils.np_to_variable(im,
                                       is_cuda=True,
                                       volatile=False).permute(0, 3, 1, 2)

    bbox_pred, conf_pred, score_pred = net(im_data)

    if args.mGPUs:
        coord_loss_var, conf_objloss_var, conf_noobjloss_var, cls_loss_var = net.module.loss(bbox_pred,
                                                                                             conf_pred, score_pred,
                                                                                             gt_boxes, gt_classes, gt_RT,
                                                                                             dontcare,
                                                                                             size_index)
    else:
        coord_loss_var, conf_objloss_var, conf_noobjloss_var, cls_loss_var = net.loss(bbox_pred, conf_pred,
                                                                                      score_pred,
                                                                                      gt_boxes, gt_classes, gt_RT, dontcare,
                                                                                      size_index)

    loss_var = cfg.lambda_coord * coord_loss_var + cfg.lambda_objconf * conf_objloss_var + \
               cfg.lambda_noobjconf * conf_noobjloss_var + cfg.lambda_class * cls_loss_var

    coord_loss_np = coord_loss_var.data.cpu().numpy()
    conf_objloss_np = conf_objloss_var.data.cpu().numpy()
    conf_noobjloss_np = conf_noobjloss_var.data.cpu().numpy()
    cls_loss_np = cls_loss_var.data.cpu().numpy()
    train_loss_np = loss_var.data.cpu().numpy()
    train_loss_epoch += train_loss_np
    coord_loss_epoch += coord_loss_np
    conf_loss_epoch += conf_objloss_np

    prob_pred = []

    for i in xrange(3):
        prob_pred.append(F.softmax(score_pred[i].view(-1, score_pred[i].size()[-1]), dim=1).view_as(score_pred[i]))

    vis_scaleid = randint(0, 2)
    ### for visualisation of predictions in tensorboard
    bbox_pred_np = bbox_pred[vis_scaleid].data[0:1].cpu().numpy()
    conf_pred_np = conf_pred[vis_scaleid].data[0:1].cpu().numpy()
    prob_pred_np = prob_pred[vis_scaleid].data[0:1].cpu().numpy()

    optimizer.zero_grad()
    loss_var.backward()

    torch.nn.utils.clip_grad_norm(net.parameters(), 5)
    optimizer.step()

    return bbox_pred_np, conf_pred_np, prob_pred_np, coord_loss_np, conf_objloss_np, conf_noobjloss_np, cls_loss_np, train_loss_np, train_loss_epoch, coord_loss_epoch, conf_loss_epoch


def my_collate(batch):
    size_index = 3

    w, h = cfg.multi_scale_inp_size[size_index]

    imgs = []
    annots = []
    gt_classes = []
    gt_RT = []
    origin_im = []
    samples = {}

    for data in batch:
        gt_boxes = np.asarray(data['gt_boxes'], dtype=np.float)

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
        K = cfg.cam_K
        K = cfg.cam_K
        R, t = yolo_utils.pnp(objpoints3D, gt_boxes.reshape(cfg.args.num_detection_points, 2), K)
        Rt = np.concatenate((R, t), axis=1)

        imgs.append(data['image'])
        annots.append(gt_boxes)
        gt_classes.append(data['gt_classes'])
        origin_im.append(data['origin_im'])
        gt_RT.append(Rt.reshape(12))

    samples['image'] = np.stack(imgs, axis=0)
    samples['gt_boxes'] = annots
    samples['gt_classes'] = gt_classes
    samples['origin_im'] = origin_im
    samples['dontcare'] = []
    samples['gt_RT'] = gt_RT

    return samples, size_index


def save_checkpoint(state, second_training, save_dir, epoch_save=False):
    if not epoch_save:
        if second_training:
            model_file = os.path.join(save_dir, 'model_second.pth.tar')
        else:
            model_file = os.path.join(save_dir, 'model.pth.tar')
    else:
        model_file = os.path.join(save_dir, 'model_' + str(state['epoch']) + '.pth.tar')

    torch.save(state, model_file)


if __name__=='__main__':
    args = cfg.args
    cv2.setNumThreads(1)

    print('Called with args:')
    print(args)

    train_output_dir = os.path.join(cfg.TRAIN_DIR, args.exp_name)
    epochsave_dir = os.path.join(cfg.TRAIN_DIR, args.exp_name, 'epoch_save')
    mintrainloss_dir = os.path.join(cfg.TRAIN_DIR, args.exp_name, 'min_train_loss')

    if not os.path.exists(train_output_dir):
        os.makedirs(train_output_dir)
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

    corners_3d = yolo_utils.threed_corners(args.class_name)
    vertices = yolo_utils.threed_vertices(args.class_name)

    if args.torch_cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    # Specify the number of workers
    kwargs = {'num_workers': 2, 'pin_memory': True}

    # data loader
    imdb_train = Dataset_class(cfg.imdb_train, args.class_name, cfg.DATA_DIR, yolo_utils.preprocess_train, None)
    dataloader_train = DataLoader(imdb_train, batch_size=args.batch_size, shuffle=True, collate_fn= my_collate, **kwargs)

    batch_per_epoch = len(dataloader_train)
    print('load data sucessful...')

    train_loss_epoch = 0
    coord_loss_epoch = 0
    conf_loss_epoch = 0
    min_train_loss = float("inf")
    min_error2d = float("inf")
    min_error3d = float("inf")

    # model load
    net = Darknet()

    lr = args.lr

    if args.second_training:
        print "Loading the pretrained model"
        if args.min_train_loss:
            restore_file = os.path.join(train_output_dir, 'min_train_loss', 'model.pth.tar')
        checkpoint = torch.load(restore_file)
        print("=> loading checkpoint '{}' from '{}'".format(restore_file, checkpoint['epoch']))
        net.load_state_dict(checkpoint['state_dict'])
        net.cuda()
        print('load net succ...')
        if not cfg.args.ADAM_optim:
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=cfg.momentum,
                                        weight_decay=cfg.weight_decay)
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=cfg.weight_decay)

        start_epoch = checkpoint['epoch']
        max_epoch = args.max_epochs
        min_train_loss = checkpoint['min_train_loss']
        cfg.lambda_objconf = 1
        cfg.lambda_noobjconf = 1
    else:
        # net.load_from_npz(cfg.pretrained_model, num_conv=18)
        net.load_weights(cfg.pretrained_model_yolov3)
        net.cuda()
        print('load net succ...')
        # optimizer
        start_epoch = 0
        max_epoch = args.max_epochs
        if not cfg.args.ADAM_optim:
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=cfg.momentum,
                                        weight_decay=cfg.weight_decay)
        else:
            optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=cfg.weight_decay)

    if args.mGPUs:
        net = torch.nn.DataParallel(net)

    # tensorboard
    use_tensorboard = args.use_tensorboard and SummaryWriter is not None
    if use_tensorboard:
        summary_writer = SummaryWriter(os.path.join(cfg.TRAIN_DIR, 'runs', args.exp_name,
                                                    datetime.datetime.now().strftime('%b%d_%H-%M-%S')))
    else:
        summary_writer = None

    t = Timer()

    for epoch in range(start_epoch, max_epoch):
        for i_batch, sample_batched in enumerate(dataloader_train):

            t.tic()
            # batch
            size_index = sample_batched[1]
            im = sample_batched[0]['origin_im']
            gt_boxes = sample_batched[0]['gt_boxes']
            gt_classes = sample_batched[0]['gt_classes']

            bbox_pred, conf_pred, prob_pred, coord_loss, conf_objloss, \
            conf_noobjloss, cls_loss, train_loss, train_loss_epoch, coord_loss_epoch, conf_loss_epoch = train_batch(net,
                                                                                                                    sample_batched,
                                                                                                                    train_loss_epoch,
                                                                                                                    coord_loss_epoch,
                                                                                                                    conf_loss_epoch)

            duration = t.toc()

            iter_count = (epoch + float(i_batch)/batch_per_epoch) * 1000

            if i_batch % cfg.log_interval == 0:

                print(('epoch %d[%d/%d], loss: %.3f, coord_loss: %.3f, conf_objloss: %.3f, conf_noobjloss: %.3f '
                       'cls_loss: %.3f (%.2f s/batch, rest:%s)' %
                       (epoch, i_batch, batch_per_epoch, train_loss, coord_loss,
                        conf_objloss, conf_noobjloss, cls_loss, duration,
                        str(datetime.timedelta(seconds=int((batch_per_epoch - i_batch) * duration))))))  # noqa

                if summary_writer:
                    summary_writer.add_scalar('loss_train', train_loss, iter_count)
                    summary_writer.add_scalar('loss_coord', coord_loss, iter_count)
                    summary_writer.add_scalar('loss_objconf', conf_objloss, iter_count)
                    summary_writer.add_scalar('loss_noobjconf', conf_noobjloss, iter_count)
                    summary_writer.add_scalar('loss_cls', cls_loss, iter_count)
                    summary_writer.add_scalar('learning_rate', lr, iter_count)

                t.clear()
                print("image_size {}".format(cfg.multi_scale_inp_size[size_index]))

            if (epoch % 1 == 0 and (i_batch + 1) % batch_per_epoch == 0) and epoch > 0:

                if summary_writer:
                    summary_writer.add_scalar('train_loss_epoch', train_loss_epoch / batch_per_epoch, epoch)
                    summary_writer.add_scalar('coord_loss_epoch', coord_loss_epoch / batch_per_epoch, epoch)
                    summary_writer.add_scalar('conf_loss_epoch', conf_loss_epoch / batch_per_epoch, epoch)

                    # plot results
                    image = im[0]
                    bboxes, scores, cls_inds = yolo_utils.postprocess(image.shape, cfg.args.thresh, size_index, False,
                                                                      (bbox_pred, conf_pred, prob_pred))
                    im2show = yolo_utils.draw_detection(image, bboxes, scores, cls_inds, cfg, imdb_train._classes, 0.5, objpoints3D, corners_3d, vertices)
                    summary_writer.add_image('predict_train', cv2.cvtColor(im2show, cv2.COLOR_BGR2RGB), epoch)

                if args.second_training:
                    if epoch in cfg.save_epochs:
                        save_dir = os.path.join(train_output_dir, 'epoch_save')
                        if args.mGPUs:
                            save_checkpoint({
                                'epoch': epoch,
                                'min_train_loss': train_loss_epoch,
                                'state_dict': net.module.state_dict(),
                                'optimizer': optimizer.state_dict()}, args.second_training, save_dir=save_dir, epoch_save=True)
                        else:
                            save_checkpoint({
                                'epoch': epoch,
                                'min_train_loss': train_loss_epoch,
                                'state_dict': net.state_dict(),
                                'optimizer': optimizer.state_dict()}, args.second_training, save_dir=save_dir, epoch_save=True)
                        print('saved checkpoint for epoch {:d}'.format(epoch))

                if train_loss_epoch < min_train_loss:
                    min_train_loss = train_loss_epoch
                    save_dir = os.path.join(train_output_dir, 'min_train_loss')
                    if args.mGPUs:
                        save_checkpoint({
                            'epoch': epoch,
                            'min_train_loss': min_train_loss,
                            'state_dict': net.module.state_dict(),
                            'optimizer': optimizer.state_dict()}, args.second_training, save_dir=save_dir)
                    else:
                        save_checkpoint({
                            'epoch': epoch,
                            'min_train_loss': min_train_loss,
                            'state_dict': net.state_dict(),
                            'optimizer': optimizer.state_dict()}, args.second_training, save_dir=save_dir)
                    print('saved checkpoint for minimum train loss')

                train_loss_epoch = 0
                coord_loss_epoch = 0
                conf_loss_epoch = 0

                if not cfg.args.ADAM_optim:
                    if epoch in cfg.lr_decay_epochs:
                        lr *= cfg.lr_decay
                        optimizer = torch.optim.SGD(net.parameters(), lr=lr,
                                                    momentum=cfg.momentum,
                                                    weight_decay=cfg.weight_decay)
