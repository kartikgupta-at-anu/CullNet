import pickle
import uuid
from torch.utils.data import Dataset
import numpy as np
import os, yaml
import cfgs.config_yolo6d as cfg
import utils.yolo6d as yolo_utils
from .dataset_eval import dataset_eval


class CullNetDataset_class(Dataset):
    def __init__(self, imdb_name, class_name, datadir, transform, dst_size):
        meta = imdb_name.split('_')
        self._image_set = meta[1]
        self._name = imdb_name

        self._data_path = os.path.join(datadir, cfg.args.datadirname, class_name)
        self._data_dir = self._data_path
        self._classes = cfg.label_names

        self._num_classes = len(self._classes)

        self._class_to_ind = dict(list(zip(self._classes, list(range(self._num_classes)))))
        self._image_ext = '.jpg'

        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        self.config = {'cleanup': True, 'use_salt': True}

        self.transform = transform
        self._dst_size = dst_size

        self._image_indexes = []
        self._image_names = []
        self._annotations = []

        self._vertices = yolo_utils.threed_vertices(class_name)
        self._objpoints3d = yolo_utils.threed_corners(class_name)

        if cfg.args.dataset_name == 'LINEMOD':
            with open("cfgs/diameter_linemod.yml", 'r') as stream:
                diam_data = yaml.load(stream)

        self._objdiam = diam_data[cfg.args.class_name]

        self.load_dataset()

    def __len__(self):
        return len(self._image_indexes)

    def __getitem__(self, index):
        if self._name == 'linemod_trainerror' or self._name == 'linemod_test' or self._name == 'linemod_val' or \
                self._name == 'occlusion_test':
            image, origin_gt_boxes, gt_classes, dontcare, origin_im, gt_RT, poseproposal_data = self.transform(
                [self._image_names[index], self._image_indexes[index], self._annotations[index], self._vertices])
            sample = {'image': image, 'origin_im': origin_im,
                      'origin_gt_boxes': origin_gt_boxes,
                      'gt_classes': gt_classes,  'gt_RT': gt_RT, 'pose_proposals': poseproposal_data}
        else:
            image, gt_boxes, gt_classes, dontcare, origin_im, gt_RT, poseproposal_data = self.transform(
                [self._image_names[index], self._image_indexes[index], self._annotations[index], self._vertices,
                 self._objpoints3d, self._objdiam])
            sample = {'image': image, 'gt_boxes': gt_boxes, 'gt_classes': gt_classes, 'dontcare': dontcare,
                      'origin_im': origin_im, 'gt_RT': gt_RT, 'pose_proposals': poseproposal_data}
        return sample

    def load_dataset(self):
        self._image_indexes = self._load_image_set_index()
        self._image_names = [self.image_path_from_index(index)
                             for index in self._image_indexes]
        self._annotations = self._load_dataset_annotations()

    def evaluate_detections(self, all_boxes, output_dir=None, verbose=True):
        self._write_dataset_results_file(all_boxes)
        accuracy2d, accuracy3d, twod_dists, threed_dists = self._do_python_eval(verbose, output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_dataset_results_file_template().format(cls)
                os.remove(filename)
        return [accuracy2d, accuracy3d], twod_dists, threed_dists
    # -------------------------------------------------------------
    def image_path_from_index(self, index):
        image_path = os.path.join(self._data_path, 'JPEGImages',
                                  index + self._image_ext)
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        image_set_file = os.path.join(self._data_path, 'ImageSets',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def _load_dataset_annotations(self):
        cache_file = os.path.join(self.cache_path, self._name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self._name, cache_file))
            return roidb

        gt_roidb = [self._annotation_from_index(index)
                    for index in self._image_indexes]
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

    def _annotation_from_index(self, index):
        filename = os.path.join(self._data_path, 'Annotations', index + '.txt')
        pose_filename = os.path.join(self._data_path, 'PoseAnnotations', index + '.txt')

        with open(filename) as f:
            data = f.readlines()

        if cfg.args.gt_RT_available:
            with open(pose_filename) as f:
                pose_data = f.readlines()

        num_objs = len(data)
        boxes = np.zeros((num_objs, 2 * cfg.args.num_detection_points), dtype=np.float)
        gt_RT = np.zeros((num_objs, 12), dtype=np.float)

        gt_classes = np.zeros((num_objs), dtype=np.int)
        # Load object bounding boxes into a data frame.
        for ix, aline in enumerate(data):
            tokens = aline.strip().split()
            if len(tokens) != (2 * cfg.args.num_detection_points) + 1:
                continue

            cls = self._class_to_ind[tokens[(2 * cfg.args.num_detection_points)]]
            gt_classes[ix] = cls
            boxes[ix, :] = map(float, tokens[0:(2 * cfg.args.num_detection_points)])

        if cfg.args.gt_RT_available:
            for ix, aline in enumerate(pose_data):
                tokens = aline.strip().split()
                if len(tokens) < 2:
                    continue
                gt_RT[ix, :] = map(float, tokens[0:12])

        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_RT': gt_RT,
                'flipped': False}

    def _get_dataset_results_file_template(self):
        filename = self._get_comp_id() + '_det_' + self._image_set + \
                   '_{:s}.txt'
        filedir = os.path.join(self._data_path,
                               'results')
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path

    def _write_dataset_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            print('Writing {} Dataset results file'.format(cls))
            filename = self._get_dataset_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self._image_indexes):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f}'.
                                format(index, dets[k, -1]))
                        for l in range(cfg.args.num_detection_points * 2):
                            f.write(' {:.1f}'.format(dets[k, l]))
                        f.write('\n')

    def _do_python_eval(self, verbose, output_dir='output'):
        annopath = os.path.join(
            self._data_path,
            'Annotations',
            '{:s}.txt')
        imagesetfile = os.path.join(
            self._data_path,
            'ImageSets',
            self._image_set + '.txt')
        cachedir = os.path.join(self._data_path, self._image_set, 'annotations_cache')
        recs2d = []
        recs3d = []
        if output_dir is not None and not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_dataset_results_file_template().format(cls)
            rec2d, rec3d, twod_dists, threed_dists = dataset_eval(
                filename, annopath, imagesetfile, cls, cachedir)
            if type(rec2d) is dict:
                recs2d += [rec2d[5]]
                recs3d += [rec3d]
                if verbose:
                    print(('Reprojection Error : Pose Accuracy for {} = {:.4f}%, {:.4f}%'.format(cls, rec2d[5] * 100 , rec3d * 100)))
                    for px_threshold in [5, 7, 10, 12, 15, 20, 25, 30]:
                        # Print test statistics
                        print(' Acc using {} px 2D Projection = {:.2f}%'.format(px_threshold, rec2d[px_threshold] * 100))
            else:
                recs2d += [rec2d]
                recs3d += [rec3d]
                if verbose:
                    print(('Reprojection Error : Pose Accuracy for {} = {:.4f}, {:.4f}'.format(cls, rec2d, rec3d)))
        if verbose:
            print(('Mean Pose Accuracy = {:.4f}, {:.4f}'.format(np.mean(recs2d), np.mean(recs3d))))
            print('~~~~~~~~')
        return np.mean(recs2d), np.mean(recs3d), twod_dists, threed_dists

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
                   else self._comp_id)
        return comp_id

    @property
    def cache_path(self):
        cache_path = os.path.join(self._data_dir, 'cache')
        if not os.path.exists(cache_path):
            os.mkdir(cache_path)
        return cache_path

