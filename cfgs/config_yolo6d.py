import os
import numpy as np
import argparse


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a YOLOv3-6D and CullNet network')
    parser.add_argument('--start_epoch', dest='start_epoch',
                        help='starting epoch', default=1, type=int)
    parser.add_argument('--epochs', dest='max_epochs', help='number of epochs to train', default=20, type=int)
    parser.add_argument('--data_location', dest='data_location',
                        help='data location where JPEGImages and annotations can be found', default=None)
    parser.add_argument('--class_name', dest='class_name', help='classname', default=None)
    parser.add_argument('--exp_name', dest='exp_name', help='expname')
    parser.add_argument('--exp_name_cullnet', dest='exp_name_cullnet', help='expname of cullnet')
    parser.add_argument('--second_training_net_load', dest='second_training_net',
                        help='file location of net to be used in the second or final training')
    parser.add_argument('--mGPUs', dest='mGPUs', help='whether use multiple GPUs', action='store_true')
    parser.add_argument('--second_training', dest='second_training', help='second training for confidence values',
                        action='store_true')
    parser.add_argument('--bs', dest='batch_size', help='batch_size', default=16, type=int)
    parser.add_argument('--sub_bs', dest='sub_bs', help='subnetwork batch_size', default=80, type=int)
    parser.add_argument('--sub_bs_test', dest='sub_bs_test', help='subnetwork batch_size', default=None)

    parser.add_argument('--vis_debug', dest='vis_debug', help='visualise input data to network', action='store_true')
    parser.add_argument('--model', dest='model_location',
                        help='model location in models/training/')
    parser.add_argument('--image_size_index', type=int, default=3,
                        metavar='image_size_index',
                        help='setting images size index 0:320, 1:352, 2:384, 3:416, 4:448, 5:480, 6:512, 7:544, 8:576')
    parser.add_argument('--vis', dest='vis',
                        help='whether to visualise the detections or not',
                        action='store_true')
    parser.add_argument('--thresh', dest='thresh',
                        help='box pruning threshold',
                        default=0.1, type=float)
    parser.add_argument('--lambda_coord', dest='lambda_coord',
                        help='weight for projections loss',
                        default=1, type=float)
    parser.add_argument('--lambda_objconf', dest='lambda_objconf',
                        help='weight for cells that have object',
                        default=0, type=float)
    parser.add_argument('--lambda_noobjconf', dest='lambda_noobjconf',
                        help='weight for cells that dont have object',
                        default=0, type=float)
    parser.add_argument('--num_detection_points', dest='num_detection_points',
                        help='num_detection_points',
                        default=9, type=int)
    parser.add_argument('--datadirectory_name', dest='datadirname',
                        help='name of the directory where the dataset is')
    parser.add_argument('--dataset_name', dest='dataset_name',
                        help='name of the dataset', default='LINEMOD')
    parser.add_argument('--min_error2d', dest='min_error2d',
                        help='whether to take minimum error2d model for second training',
                        action='store_true')
    parser.add_argument('--epoch_save', dest='epoch_save',
                        help='whether to take epoch save model for second training',
                        action='store_true')
    parser.add_argument('--min_train_loss', dest='min_train_loss',
                        help='whether to take mintraining loss model for second training',
                        action='store_true')
    parser.add_argument('--torch_cudnn_benchmark', dest='torch_cudnn_benchmark', help='cudnn benchmark',
                        action='store_true')
    parser.add_argument('--diff_aspectratio', dest='diff_aspectratio', help='different aspect ratio by resizing the image',
                        action='store_true')
    parser.add_argument('--random_erasing', dest='random_erasing', help='whether or not to use random erasing as augmentations',
                        action='store_true')
    parser.add_argument('--random_trans_scale', dest='random_trans_scale', help='whether or not to use random translation and scaling'
                                                                                ' as augmentations',
                        action='store_true')
    parser.add_argument('--non_nms', dest='non_nms', help='when results for gt oracle for choosing best pose hypothesis',
                        action='store_true')
    parser.add_argument('--topk_ransac', dest='topk_ransac', help='use ransac on the topk pose proposals',
                        action='store_true')
    parser.add_argument('--pick_one', dest='pick_one', help='when results for gt oracle for not nonnms',
                        action='store_true')
    parser.add_argument('--cull_net', dest='cull_net', help='train cull net only',
                        action='store_true')
    parser.add_argument('--seg_cullnet', dest='seg_cullnet', help='train seg cull net only',
                        action='store_true')
    parser.add_argument('--normalized_cullnet', dest='normalized_cullnet', help='train normalized cull net',
                        action='store_true')
    parser.add_argument('--gt_RT_available', dest='gt_RT_available', help='ground truth pose anotation in RT available',
                        action='store_true')
    parser.add_argument('--ADAM_optim', dest='ADAM_optim', help='ADAM optimizer',
                        action='store_true')
    parser.add_argument('--ADI_metric', dest='ADI', help='ADI metric for symmetric objects',
                        action='store_true')
    parser.add_argument('--lr', dest='lr', help='starting learning rate', default=0.001, type=float)
    parser.add_argument('--cullnet_type', dest='cullnet_type', help='train seg cull net with specified network')
    parser.add_argument('--cullnet_input', dest='cullnet_input',
                        help='cullnet input resolution',
                        default=224, type=int)
    parser.add_argument('--gn_channels', dest='gn_channels',
                        help='gn_channels',
                        default=32, type=int)
    parser.add_argument('--topk', dest='topk',
                        help='top k proposals',
                        default=16, type=int)
    parser.add_argument('--k_proposals', dest='k_proposals',
                        help='k proposals',
                        default=32, type=int)
    parser.add_argument('--nearby_test', dest='nearby_test',
                        help='nearby for test',
                        default=1, type=int)
    parser.add_argument('--bias_correction', dest='bias_correction',
                        help='bias_correction',
                        default=None)
    parser.add_argument('--cullnet_inconf', dest='cullnet_inconf', help='train seg cull net with this input config',
                        default='mask_out')
    parser.add_argument('--cullnet_confidence', dest='cullnet_confidence', help='train seg cull net with 2d or 3d confidence',
                        default='conf2d')
    parser.add_argument('--cullnet_trainloader', dest='cullnet_trainloader', help='train seg cull net with these type of train images only',
                        default='realonly')
    parser.add_argument('--save_results', dest='save_results_bool', help='save the results to a yaml file (only for evaluation code)',
                        action='store_true')
    # # log and diaplay
    parser.add_argument('--log_3derror', dest='log_3derror', help='log errors in 3d',
                        action='store_true')
    parser.add_argument('--confidence_plotlogs', dest='confidence_plotlogs', help='log confidences',
                        action='store_true')
    parser.add_argument('--use_tensorboard', dest='use_tensorboard', help='whether use tensorflow tensorboard',
                        action='store_true')
    args = parser.parse_args()
    return args


args = parse_args()


# dataset specific
############################
# trained model

if args.dataset_name == 'LINEMOD':
    # LINEMOD
    cam_K = np.reshape([572.4114, 0.0, 325.2611, 0.0, 573.57043, 242.04899, 0.0, 0.0, 1.0], (3, 3))
    imdb_train = 'linemod_trainval'
    if args.cullnet_trainloader == "realonly":
        imdb_traincullnet = 'linemod_traincullnet'
    if args.cullnet_trainloader == "all":
        imdb_traincullnet = 'linemod_trainval'

    imdb_test = 'linemod_test'
    if args.log_3derror:
        imdb_test = 'linemod_trainerror'

    imdb_val = 'linemod_val'
    imdbocc_test = 'occlusion_test'

    label_names = (args.class_name, '__background__')
    num_classes = 2

    occlusion_classes = ('ape', 'can', 'cat', 'driller', 'duck', 'eggbox', 'glue',
                         'holepuncher')

anchors = np.asarray([(1, 1)],
                     dtype=np.float)
num_anchors = len(anchors)

##experimental settings
###########################

pretrained_fname = 'trained_models/darknet19.weights.npz'
pretrained_fname_yolov3 = 'trained_models/darknet53.conv.74'


lr_decay_epochs = {50, 75, 110}
save_epochs = {49, 99, 115}
saveresults_epochs = {99, 115}

lr_decay = 1./10

weight_decay = 0.0005
momentum = 0.9

k_proposals_test = int(args.topk * args.nearby_test)

############ for training yolo6d
####for first training
lambda_coord = 1
lambda_weights = 0
lambda_geometry = 0
lambda_objconf = args.lambda_objconf
lambda_noobjconf = args.lambda_noobjconf
lambda_class = 1

# input and output size
############################
multi_scale_inp_size = [np.array([320, 320], dtype=np.int),
                        np.array([352, 352], dtype=np.int),
                        np.array([384, 384], dtype=np.int),
                        np.array([416, 416], dtype=np.int),
                        np.array([448, 448], dtype=np.int),
                        np.array([480, 480], dtype=np.int),
                        np.array([512, 512], dtype=np.int),
                        np.array([544, 544], dtype=np.int),
                        np.array([576, 576], dtype=np.int),
                        np.array([608, 608], dtype=np.int),
                        np.array([640, 640], dtype=np.int)
                        ]   # w, h
multi_scale_out_size = [multi_scale_inp_size[0] / 32,
                        multi_scale_inp_size[1] / 32,
                        multi_scale_inp_size[2] / 32,
                        multi_scale_inp_size[3] / 32,
                        multi_scale_inp_size[4] / 32,
                        multi_scale_inp_size[5] / 32,
                        multi_scale_inp_size[6] / 32,
                        multi_scale_inp_size[7] / 32,
                        multi_scale_inp_size[8] / 32,
                        multi_scale_inp_size[9] / 32,
                        multi_scale_inp_size[10] / 32
                        ] # w, h



# for display
############################
def _to_color(indx, base):
    """ return (b, r, g) tuple"""
    base2 = base * base
    b = 2 - indx / base2
    r = 2 - (indx % base2) / base
    g = 2 - (indx % base2) % base
    return b * 127, r * 127, g * 127


base = int(np.ceil(pow(num_classes, 1. / 3)))
colors = [_to_color(x, base) for x in range(num_classes)]


# dir config
############################
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
MODEL_DIR = os.path.join(ROOT_DIR, 'models')
TRAIN_DIR = os.path.join(MODEL_DIR, 'training')
TEST_DIR = os.path.join(MODEL_DIR, 'testing')

pretrained_model = os.path.join(MODEL_DIR, pretrained_fname)
pretrained_model_yolov3 = os.path.join(MODEL_DIR, pretrained_fname_yolov3)
test_output_dir = os.path.join(TEST_DIR, args.exp_name, imdb_test)
val_output_dir = os.path.join(TEST_DIR, args.exp_name, imdb_val)

if args.dataset_name == 'LINEMOD':
    occtest_output_dir = os.path.join(TEST_DIR, args.exp_name, imdbocc_test)

rand_seed = 1024
use_tensorboard = True

log_interval = 50
