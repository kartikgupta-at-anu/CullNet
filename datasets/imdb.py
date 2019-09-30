import os
import numpy as np
from multiprocessing import Pool
from functools import partial
import cfgs.config_yolo6d as cfg
import cv2


def mkdir(path, max_depth=3):
    parent, child = os.path.split(path)
    if not os.path.exists(parent) and max_depth > 1:
        mkdir(parent, max_depth-1)

    if not os.path.exists(path):
        os.mkdir(path)


class ImageDataset(object):
    def __init__(self, name, datadir, batch_size, im_processor,
                 shuffle=True, dst_size=None):
        self._name = name
        self._data_dir = datadir
        self._batch_size = batch_size
        self.dst_size = dst_size

        self._epoch = -1
        self._num_classes = 0
        self._classes = []

        # load by self.load_dataset()
        self._image_indexes = []
        self._image_names = []
        self._annotations = []
        # Use this dict for storing dataset specific config options
        self.config = {}

        self._shuffle = shuffle
        self._im_processor = im_processor

    def close(self):
        self.pool.terminate()
        self.pool.join()
        self.gen = None

    def load_dataset(self):
        raise NotImplementedError

    def evaluate_detections(self, all_boxes, output_dir=None):
        raise NotImplementedError

    def get_annotation(self, i):
        if self.annotations is None:
            return None
        return self.annotations[i]

    @property
    def name(self):
        return self._name

    @property
    def num_classes(self):
        return len(self._classes)

    @property
    def classes(self):
        return self._classes

    @property
    def image_names(self):
        return self._image_names

    @property
    def image_indexes(self):
        return self._image_indexes

    @property
    def annotations(self):
        return self._annotations

    @property
    def cache_path(self):
        cache_path = os.path.join(self._data_dir, 'cache')
        mkdir(cache_path)
        return cache_path

    @property
    def num_images(self):
        return len(self.image_names)

    @property
    def epoch(self):
        return self._epoch

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def batch_per_epoch(self):
        return self.num_images // self.batch_size
