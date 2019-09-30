import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import resnet_gn, resnet_concat_gn
import cfgs.config_yolo6d as cfg
from utils.yolo6d import threed_vertices, threed_correspondences, threed_corners
import utils.network as net_utils

vertices_old = threed_vertices(cfg.label_names[0])
corners3d = threed_corners(cfg.label_names[0])
vertices = np.c_[np.array(vertices_old), np.ones((len(vertices_old), 1))].transpose()
if cfg.args.num_detection_points == 9:
    objpoints3D = threed_corners(cfg.args.class_name)
else:
    objpoints3D = threed_correspondences(cfg.args.class_name)


class Cullnet(nn.Module):
    def __init__(self):
        super(Cullnet, self).__init__()
        if cfg.args.cullnet_type == 'resnet50_gn':
            model = resnet_gn.resnet50(pretrained=False)
            self.vismodel = nn.Sequential(*list(model.children())[:-1])
            num_final_in = model.fc.in_features
            # linear
            out_channels = 1
            self.regression = net_utils.FC(num_final_in, out_channels, relu=False)

        elif cfg.args.cullnet_type == 'resnet50concat_gn':
            model = resnet_concat_gn.resnet50(pretrained=False)
            self.vismodel = nn.Sequential(*list(model.children())[:-1])
            num_final_in = model.fc.in_features
            # linear
            out_channels = 1
            self.regression = net_utils.FC(num_final_in, out_channels, relu=False)

    def forward(self, x):
        x = self.vismodel(x)
        x = x.view(x.size(0), -1)
        x = self.regression(x)

        out = F.sigmoid(x)

        return out

    def loss(self, conf_pred, gt_conf):

        _confs = net_utils.np_to_variable(gt_conf, volatile=True)

        conf_loss = nn.MSELoss(size_average=False)(conf_pred, _confs)/ len(gt_conf)

        return conf_loss

    def initialize_weights(self):
        """Initialize model weights.
        """

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


if __name__ == '__main__':
    net = Cullnet()
