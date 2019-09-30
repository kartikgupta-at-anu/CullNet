import torch
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
import cfgs.config_yolo6d as cfg
from utils.yolo6d import threed_vertices, threed_correspondences, threed_corners
import utils.network as net_utils
from functools import partial
from multiprocessing import Pool
from utils.parse_config import *

vertices_old = threed_vertices(cfg.label_names[0])
corners3d = threed_corners(cfg.label_names[0])
vertices = np.c_[np.array(vertices_old), np.ones((len(vertices_old), 1))].transpose()
if cfg.args.num_detection_points == 9:
    objpoints3D = threed_corners(cfg.args.class_name)
else:
    objpoints3D = threed_correspondences(cfg.args.class_name)


#not using anchors currently, simply transforming to prediction domain to the 0-1 scale coordinate system
def yolo_to_cuboid(bbox_pred, anchors, H, W):
    bsize = bbox_pred.shape[0]
    num_anchors = anchors.shape[0]
    bbox_out = np.zeros((bsize, H*W, num_anchors, (2 * cfg.args.num_detection_points)), dtype=np.float)
    cols = np.repeat(np.tile(np.arange(W), H), cfg.args.num_detection_points).reshape(H*W, 1, cfg.args.num_detection_points)
    rows = np.repeat(np.repeat(np.arange(H), W), cfg.args.num_detection_points).reshape(H*W, 1, cfg.args.num_detection_points)
    bbox_out[:, :, :, 0::2] = (bbox_pred[:, :, :, 0::2] + cols) / W
    bbox_out[:, :, :, 1::2] = (bbox_pred[:, :, :, 1::2] + rows) / H
    return bbox_out


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


def confidence_cuboid(bbox_np_b, gt_boxes_b):
    confs = confidence_function_fast(bbox_np_b, gt_boxes_b)
    return confs


def _process_batch(data, size_index):
    # bbox_pred_np, gt_boxes, gt_classes, iou_pred_np, img = data
    bbox_pred_np, gt_boxes, gt_classes, gt_RT, iou_pred_np = data

    # net output
    hw, num_anchors, _ = bbox_pred_np.shape

    W, H = int(math.sqrt(hw)), int(math.sqrt(hw))
    inp_size = cfg.multi_scale_inp_size[size_index]
    out_size = (W, H)

    # gt
    _classes = np.zeros([hw, num_anchors, 1], dtype=np.float)
    _class_mask = np.zeros([hw, num_anchors, 1], dtype=np.float)
    _confs = np.zeros([hw, num_anchors, 1], dtype=np.float)
    _conf_objmask = np.zeros([hw, num_anchors, 1], dtype=np.float)
    _box_mask = np.zeros([hw, num_anchors, 1], dtype=np.float)
    _boxes = np.zeros([hw, num_anchors, (2 * cfg.args.num_detection_points)], dtype=np.float)

    # scale pred_bbox so that we can calculate gt confidences for the predicted boxes
    anchors = np.ascontiguousarray(cfg.anchors, dtype=np.float)
    bbox_pred_np = np.expand_dims(bbox_pred_np, 0)
    bbox_np = yolo_to_cuboid(
        np.ascontiguousarray(bbox_pred_np, dtype=np.float),
        anchors,
        H, W)
    # bbox_np = (hw, num_anchors, (x1, y1, x2, y2 ........,x18,y18))   range: 0 ~ 1
    bbox_np = bbox_np[0]
    bbox_np[:, :, 0::2] *= float(inp_size[0])  # rescale x
    bbox_np[:, :, 1::2] *= float(inp_size[1])  # rescale y

    # gt_boxes_b = np.asarray(gt_boxes[b], dtype=np.float)
    gt_boxes_b = np.asarray(gt_boxes, dtype=np.float)
    gt_RT_b = np.asarray(gt_RT, dtype=np.float)

    # for each cell, compare predicted_bbox and gt_bbox
    bbox_np_b = np.reshape(bbox_np, [-1, (2 * cfg.args.num_detection_points)])

    confs = confidence_cuboid(
        np.ascontiguousarray(bbox_np_b, dtype=np.float),
        np.ascontiguousarray(gt_boxes_b, dtype=np.float)
    )

    _confs = np.max(confs, axis=1).reshape(_confs.shape)

    # locate the cell of each gt_boxes
    # cell width and height
    cell_w = float(inp_size[0]) / W
    cell_h = float(inp_size[1]) / H

    # cell index of the gt box
    cx = gt_boxes_b[:, 0] / cell_w
    cy = gt_boxes_b[:, 1] / cell_h

    # cell index in terms of w*h indexed array ie. 1 d array rather than 2d array of cells
    cell_inds = np.floor(cy) * W + np.floor(cx)
    cell_inds = cell_inds.astype(np.int)

    target_boxes = np.empty(gt_boxes_b.shape, dtype=np.float)
    target_boxes[:, 0] = cx - np.floor(cx)  # cx
    target_boxes[:, 1] = cy - np.floor(cy)  # cy

    for corner_id in xrange(1, cfg.args.num_detection_points):
        target_boxes[:, 2*corner_id] = \
            ((gt_boxes_b[:, 2*corner_id] / inp_size[0]) * out_size[0]) - np.floor(cx)  # tw
        target_boxes[:, 2*corner_id+1] = \
            ((gt_boxes_b[:, 2*corner_id+1] / inp_size[1]) * out_size[1]) - np.floor(cy) # th

    for i, cell_ind in enumerate(cell_inds):
        if cell_ind >= hw or cell_ind < 0:
            print('cell inds size {}'.format(len(cell_inds)))
            print('cell over {} hw {}'.format(cell_ind, hw))
            continue
        a = 0

        _conf_objmask[cell_ind, a, :] = 1

        _box_mask[cell_ind, a, :] = 1
        _boxes[cell_ind, a, :] = target_boxes[i]

        _class_mask[cell_ind, a, :] = 1
        _classes[cell_ind, a, 0] = gt_classes[i]

    return _boxes, _confs, _classes, _box_mask, _conf_objmask, _class_mask


def create_modules(module_defs):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams['channels'])]
    module_list = nn.ModuleList()
    for i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        if module_def['type'] == 'convolutional':
            bn = int(module_def['batch_normalize'])
            filters = int(module_def['filters'])
            kernel_size = int(module_def['size'])
            pad = (kernel_size - 1) // 2 if int(module_def['pad']) else 0
            modules.add_module('conv_%d' % i, nn.Conv2d(in_channels=output_filters[-1],
                                                        out_channels=filters,
                                                        kernel_size=kernel_size,
                                                        stride=int(module_def['stride']),
                                                        padding=pad,
                                                        bias=not bn))
            if bn:
                modules.add_module('batch_norm_%d' % i, nn.BatchNorm2d(filters))
            if module_def['activation'] == 'leaky':
                modules.add_module('leaky_%d' % i, nn.LeakyReLU(0.1))

        elif module_def['type'] == 'maxpool':
            kernel_size = int(module_def['size'])
            stride = int(module_def['stride'])
            if kernel_size == 2 and stride == 1:
                modules.add_module('_debug_padding_%d' % i, nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
            modules.add_module('maxpool_%d' % i, maxpool)

        elif module_def['type'] == 'upsample':
            # upsample = nn.Upsample(scale_factor=int(module_def['stride']), mode='nearest')  # WARNING: deprecated
            upsample = Upsample(scale_factor=int(module_def['stride']), mode='nearest')
            modules.add_module('upsample_%d' % i, upsample)

        elif module_def['type'] == 'route':
            layers = [int(x) for x in module_def['layers'].split(',')]
            filters = sum([output_filters[i + 1 if i > 0 else i] for i in layers])
            modules.add_module('route_%d' % i, EmptyLayer())

        elif module_def['type'] == 'shortcut':
            filters = output_filters[int(module_def['from'])]
            modules.add_module('shortcut_%d' % i, EmptyLayer())

        elif module_def['type'] == 'yolo':
            anchor_idxs = [int(x) for x in module_def['mask'].split(',')]
            # Extract anchors
            yolo_layer = YOLOLayer()
            modules.add_module('yolo_%d' % i, yolo_layer)

        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list


class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()


class Upsample(nn.Module):
    # Custom Upsample layer (nn.Upsample gives deprecated warning message)

    def __init__(self, scale_factor=1, mode='nearest'):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.upsample(x, scale_factor=self.scale_factor, mode=self.mode)


class YOLOLayer(nn.Module):

    def __init__(self):
        super(YOLOLayer, self).__init__()
        self.nA = 1  # number of anchors
        self.nC = 2  # number of classes
        self.bbox_attrs = 2 * cfg.args.num_detection_points + cfg.num_classes + 1

    def forward(self, p):
        bs = p.shape[0]  # batch size
        nG = p.shape[2]  # number of grid points

        p = p.view(bs, self.nA, self.bbox_attrs, nG * nG).permute(0, 3, 1, 2).contiguous()  # prediction

        xy = p[..., 0:2]  # x, y
        width_height = p[..., 2:(2 * cfg.args.num_detection_points)] # width, height
        p_conf = torch.sigmoid(p[..., (2 * cfg.args.num_detection_points): (2 * cfg.args.num_detection_points) + 1])  # Conf
        p_cls = p[..., (2 * cfg.args.num_detection_points) + 1:]

        return torch.cat((xy, width_height, p_conf, p_cls), 3)


class Darknet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, img_size=416):
        super(Darknet, self).__init__()
        if cfg.args.num_detection_points == 9:
            cfg_path = 'cfgs/yolov3.cfg'
        if cfg.args.num_detection_points == 8:
            cfg_path = 'cfgs/yolov3_8surf.cfg'
        if cfg.args.num_detection_points == 12:
            cfg_path = 'cfgs/yolov3_12surf.cfg'

        self.module_defs = parse_model_config(cfg_path)
        self.module_defs[0]['cfg'] = cfg_path
        self.module_defs[0]['height'] = img_size
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        self.img_size = img_size
        if not cfg.args.cull_net and not cfg.args.seg_cullnet:
            self.pool = Pool(processes=4)

    def forward(self, x, targets=None, batch_report=False, var=0):
        layer_outputs = []
        output = []
        bbox_pred = []
        conf_pred = []
        score_pred = []

        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def['type'] in ['convolutional', 'upsample', 'maxpool']:
                x = module(x)
            elif module_def['type'] == 'route':
                layer_i = [int(x) for x in module_def['layers'].split(',')]
                x = torch.cat([layer_outputs[i] for i in layer_i], 1)
            elif module_def['type'] == 'shortcut':
                layer_i = int(module_def['from'])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def['type'] == 'yolo':
                x = module(x)
                # break;

                bbox_pred.append(x[:, :, :, 0: (2 * cfg.args.num_detection_points)])
                conf_pred.append(x[:, :, :, (2 * cfg.args.num_detection_points): (2 * cfg.args.num_detection_points) + 1])
                score_pred.append(x[:, :, :, (2 * cfg.args.num_detection_points) + 1:].contiguous())
                output.append(x)

            layer_outputs.append(x)

        return bbox_pred, conf_pred, score_pred

    def loss(self, bbox_pred, conf_pred, score_pred, gt_boxes, gt_classes, gt_RT, dontcare, size_index):
        coord_losses = []
        conf_objlosses = []
        conf_noobjlosses = []
        cls_losses = []

        ## Loss for output at different scales
        for i in xrange(3):
            bbox_pred_np = bbox_pred[i].data.cpu().numpy()
            conf_pred_np = conf_pred[i].data.cpu().numpy()

            _boxes, _confs, _classes, _box_mask, _conf_objmask, _class_mask = \
                self._build_target(bbox_pred_np,
                                   gt_boxes,
                                   gt_classes,
                                   gt_RT,
                                   dontcare,
                                   conf_pred_np,
                                   size_index)

            _boxes = net_utils.np_to_variable(_boxes)
            _confs = net_utils.np_to_variable(_confs)
            _classes = net_utils.np_to_variable(_classes, dtype=torch.LongTensor)

            cls_mask_idxs = np.nonzero(_class_mask)
            box_mask_idxs = np.nonzero(_box_mask)
            conf_objmask_idxs = np.nonzero(_conf_objmask)
            conf_noobjmask_idxs = np.nonzero(1. - _conf_objmask)

            #### for len(box_mask_idxs[0]) = 0
            if len(box_mask_idxs[0]) == 0:
                coord_losses.append(torch.tensor(0, device='cuda', dtype=torch.float, requires_grad=True))
                conf_objlosses.append(torch.tensor(0, device='cuda', dtype=torch.float, requires_grad=True))
                conf_noobjlosses.append(torch.tensor(0, device='cuda', dtype=torch.float, requires_grad=True))
                cls_losses.append(torch.tensor(0, device='cuda', dtype=torch.float, requires_grad=True))

                continue;

            coord_losses.append(nn.MSELoss(size_average=False)(
                bbox_pred[i][box_mask_idxs[0], box_mask_idxs[1], box_mask_idxs[2],
                :], _boxes[box_mask_idxs[0], box_mask_idxs[1], box_mask_idxs[2],
                    :]) / len(box_mask_idxs[0]))

            conf_objlosses.append(nn.MSELoss(size_average=False)(conf_pred[i][conf_objmask_idxs[0], conf_objmask_idxs[1],
                                                                 conf_objmask_idxs[2], :], _confs[conf_objmask_idxs[0], conf_objmask_idxs[1],
                                                                                           conf_objmask_idxs[2],:]) / len(conf_objmask_idxs[0]))

            conf_noobjlosses.append(nn.MSELoss(size_average=False)(conf_pred[i][conf_noobjmask_idxs[0], conf_noobjmask_idxs[1],
                                                                   conf_noobjmask_idxs[2], :], _confs[conf_noobjmask_idxs[0], conf_noobjmask_idxs[1],
                                                                                               conf_noobjmask_idxs[2], :]) / len(conf_noobjmask_idxs[0]))

            cls_losses.append(nn.CrossEntropyLoss(size_average=False)(score_pred[i][cls_mask_idxs[0], cls_mask_idxs[1], cls_mask_idxs[2],:],
                                                                      _classes[cls_mask_idxs[0], cls_mask_idxs[1], cls_mask_idxs[2], :].view(-1)) / len(cls_mask_idxs[0]))

        coord_loss = coord_losses[0] + coord_losses[1] + coord_losses[2]
        conf_objloss = conf_objlosses[0] + conf_objlosses[1] + conf_objlosses[2]
        conf_noobjloss = conf_noobjlosses[0] + conf_noobjlosses[1] + conf_noobjlosses[2]
        cls_loss = cls_losses[0] + cls_losses[1] + cls_losses[2]

        # print t2.toc()

        return coord_loss, conf_objloss, conf_noobjloss, cls_loss

    def _build_target(self, bbox_pred_np, gt_boxes, gt_classes, gt_RT, dontcare,
                      conf_pred_np, size_index):
        """
        :param bbox_pred: shape: (bsize, h x w, num_anchors, 4) :
                          (sig(tx), sig(ty), exp(tw), exp(th))
        """

        bsize = bbox_pred_np.shape[0]

        targets = self.pool.map(partial(_process_batch, size_index=size_index),
                                ((bbox_pred_np[b], gt_boxes[b],
                                  gt_classes[b], gt_RT[b], conf_pred_np[b])
                                 for b in range(bsize)))

        _boxes = np.stack(tuple((row[0] for row in targets)))
        _confs = np.stack(tuple((row[1] for row in targets)))
        _classes = np.stack(tuple((row[2] for row in targets)))
        _box_mask = np.stack(tuple((row[3] for row in targets)))
        _conf_mask = np.stack(tuple((row[4] for row in targets)))
        _class_mask = np.stack(tuple((row[5] for row in targets)))

        return _boxes, _confs, _classes, _box_mask, _conf_mask, _class_mask

    def load_weights(self, weights_path, cutoff=-1):
        if weights_path.endswith('darknet53.conv.74'):
            cutoff = 75
        elif weights_path.endswith('yolov3-tiny.conv.15'):
            cutoff = 16

        # Open the weights file
        fp = open(weights_path, 'rb')
        header = np.fromfile(fp, dtype=np.int32, count=5)  # First five are header values

        # Needed to write header when saving weights
        self.header_info = header

        self.seen = header[3]
        weights = np.fromfile(fp, dtype=np.float32)  # The rest are weights
        fp.close()

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if module_def['type'] == 'convolutional':
                conv_layer = module[0]
                if module_def['batch_normalize']:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w


if __name__ == '__main__':
    net = Darknet()
    net.load_weights('models/darknet19.weights.npz')