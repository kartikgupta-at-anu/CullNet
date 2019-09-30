import os
import pickle
import numpy as np
import cfgs.config_yolo6d as cfg
from utils.yolo6d import pnp, compute_projection, threed_corners, threed_vertices, compute_transformation, \
    threed_correspondences, DLT, calcampose, normalization, pnp_iterative, pnp_ransac
import cv2
import yaml
from scipy import spatial


_class_to_ind = dict(list(zip(cfg.label_names, list(range(len(cfg.label_names))))))

px_threshold = [5, 7, 10, 12, 15, 20, 25, 30]

if cfg.args.log_3derror:
    log_3derror = {}


def parse_rec(filename, pose_filename):
    objects = []
    with open(filename) as f:
        data = f.readlines()

    if cfg.args.gt_RT_available:
        # Reading pose annotation file
        with open(pose_filename) as f:
            pose_data = f.readlines()

    # Load object bounding boxes into a data frame.
    for ix, aline in enumerate(data):
        obj_struct = {}
        tokens = aline.strip().split()
        if len(tokens) != (2 * cfg.args.num_detection_points) + 1:
            continue

        cls = _class_to_ind[tokens[(2 * cfg.args.num_detection_points)]]
        obj_struct['name'] = cls
        obj_struct['bbox'] = map(float, tokens[0:(2 * cfg.args.num_detection_points)])
        obj_struct['rt'] = map(float, np.zeros(12))
        objects.append(obj_struct)

    if cfg.args.gt_RT_available:
        # Reading pose RT
        for ix, aline in enumerate(pose_data):
            tokens = aline.strip().split()
            if len(tokens) < 2:
                continue
            objects[ix]['rt'] = map(float, tokens[0:12])

    return objects


def reprojection_errors(bb_gt, gt_RT, bb_det, image_id, objpoints3D, vertices, class_name, obj_diameter, obj_bias=None):
    vertices = np.c_[np.array(vertices), np.ones((len(vertices), 1))].transpose()
    corners2D_gt_corrected = np.reshape(bb_gt, ((cfg.args.num_detection_points), 2))
    corners2D_pr = np.reshape(bb_det, ((cfg.args.num_detection_points), 2))

    if cfg.args.dataset_name == 'LINEMOD':
        K = cfg.cam_K

    if cfg.args.gt_RT_available:
        Rt_gt = np.reshape(gt_RT, (3, 4))
    else:
        R_gt, t_gt = pnp(objpoints3D, corners2D_gt_corrected, K)
        Rt_gt = np.concatenate((R_gt, t_gt), axis=1)

    R_pr, t_pr = pnp(objpoints3D, corners2D_pr, K)
    Rt_pr = np.concatenate((R_pr, t_pr), axis=1)

    if cfg.args.bias_correction is not None:
        Rt_pr[2, 3] = Rt_pr[2, 3] + obj_bias['z']

    # Compute pixel error
    proj_2d_gt = compute_projection(vertices, Rt_gt, K)
    proj_2d_pred = compute_projection(vertices, Rt_pr, K)
    norm = np.linalg.norm(proj_2d_gt - proj_2d_pred, axis=0)
    pixel_dist = np.mean(norm)

    # Compute 3D error
    transform_3d_gt = compute_transformation(vertices, Rt_gt)
    transform_3d_pred = compute_transformation(vertices, Rt_pr)

    if cfg.args.ADI:
        if class_name == 'glue' or class_name == 'eggbox':
            # Calculate distances to the nearest neighbors from pts_gt to pts_est
            nn_index = spatial.cKDTree(transform_3d_pred.T)
            nn_dists, _ = nn_index.query(transform_3d_gt.T, k=1)
            vertex_dist = nn_dists.mean()
        else:
            norm3d = np.linalg.norm(transform_3d_gt - transform_3d_pred, axis=0)
            vertex_dist = np.mean(norm3d)
    else:
        norm3d = np.linalg.norm(transform_3d_gt - transform_3d_pred, axis=0)
        vertex_dist = np.mean(norm3d)

    if cfg.args.log_3derror:
        log_3derror[image_id] = {}
        log_3derror[image_id]['x'] = round(float(np.mean(transform_3d_gt - transform_3d_pred, axis=1)[0]), 4)
        log_3derror[image_id]['y'] = round(float(np.mean(transform_3d_gt - transform_3d_pred, axis=1)[1]), 4)
        log_3derror[image_id]['z'] = round(float(np.mean(transform_3d_gt - transform_3d_pred, axis=1)[2]), 4)

    if vertex_dist < 0.1 * obj_diameter:
        vertex_error = 0
    else:
        vertex_error = 1

    return pixel_dist, vertex_error, vertex_dist


def dataset_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir):

    # first load gt
    if not os.path.isdir(cachedir):
        os.makedirs(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')


    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath.format(imagename), os.path.join(cfg.DATA_DIR, cfg.args.datadirname, classname, 'PoseAnnotations', '{:s}.txt').format(imagename))
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames)))
        # save
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'rb') as f:
            recs = pickle.load(f)

    if cfg.args.num_detection_points == 9:
        objpoints3D = threed_corners(classname)
    else:
        objpoints3D = threed_correspondences(classname)
    vertices = threed_vertices(classname, eval=True)

    if cfg.args.dataset_name == 'LINEMOD':
        with open("cfgs/diameter_linemod.yml", 'r') as stream:
            diam_data = yaml.load(stream)
        if cfg.args.bias_correction == 'mean':
            with open("cfgs/bias_linemod_mean.yml", 'r') as stream:
                bias_data = yaml.load(stream)
        if cfg.args.bias_correction == 'median':
            with open("cfgs/bias_linemod_median.yml", 'r') as stream:
                bias_data = yaml.load(stream)
        if cfg.args.bias_correction == 'mode':
            with open("cfgs/bias_linemod_mode.yml", 'r') as stream:
                bias_data = yaml.load(stream)

    obj_diameter = diam_data[classname]

    if cfg.args.bias_correction is not None:
        obj_bias = bias_data[classname]
    else:
        obj_bias = None

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == _class_to_ind[classname]]
        bbox = np.array([x['bbox'] for x in R])
        rt = np.array([x['rt'] for x in R])
        # det = [False] * len(R)
        npos = npos + len(R)
        class_recs[imagename] = {'bbox': bbox, 'rt': rt}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()
    if any(lines) == 1:

        splitlines = [x.strip().split(' ') for x in lines]
        image_ids = [x[0] for x in splitlines]
        # confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp_2d = {}

        for thresh in px_threshold:
            tp_2d[thresh] = {}
        rec2d = {}

        tp_3d = {}

        twod_dists = []
        threed_dists = []

        for d in xrange(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            BBGT = R['bbox'].astype(float)
            RTGT = R['rt'].astype(float)

            if BBGT.size > 0:
                pixel_dist, vertex_error, vertex_dist = reprojection_errors(BBGT, RTGT, bb, image_ids[d], objpoints3D,
                                                                vertices, classname, obj_diameter, obj_bias)

                twod_dists.append(pixel_dist)
                threed_dists.append(vertex_dist)

                if not vertex_error:
                    tp_3d[image_ids[d]] = 1

                # Estimating 2d projection error based on different thresholds
                for thresh in px_threshold:
                    if pixel_dist < thresh:
                        tp_2d[thresh][image_ids[d]] = 1

        # Moving detection images in correct and incorrect directories
        if cfg.args.vis:
            results_cache = cfg.test_output_dir + "/results2d_cache.txt"
            fo = open(results_cache, "w")

            for k, v in tp_2d[5].items():
                fo.write(str(k) + ' '+ str(v) + '\n')
                os.rename(cfg.test_output_dir + '/' + str(k) + '.jpg', cfg.test_output_dir + '/correct/' + str(k) + '.jpg')

            results_cache = cfg.test_output_dir + "/results3d_cache.txt"
            fo = open(results_cache, "w")

            for k, v in tp_3d.items():
                fo.write(str(k) + ' '+ str(v) + '\n')

        # Estimating 2d projection error based on different thresholds
        for thresh in px_threshold:
            tp_2d[thresh] = sum(tp_2d[thresh].values())

        tp_3d = sum(tp_3d.values())

        for thresh in px_threshold:
            rec2d[thresh] = tp_2d[thresh] / float(npos)
        rec3d = tp_3d / float(npos)
    else:
        rec2d = -1
        rec3d = -1

    if cfg.args.log_3derror:
        with open('3derror_logs/' + classname + '_3derror_logs.yml', 'w') as outfile:
            yaml.dump(log_3derror, outfile, default_flow_style=False)

    return rec2d, rec3d, twod_dists, threed_dists
