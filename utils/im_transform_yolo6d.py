import numpy as np
import cv2
import random


def rand_scale(s):
    scale = random.uniform(1, s)
    if(random.randint(1,10000)%2):
        return scale
    return 1./scale


def random_distort_image(im):
    dhue = random.uniform(.95, 1.05)
    dsat = rand_scale(1.5)
    dexp = rand_scale(1.5)

    res = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)

    scale_h = lambda x: (x * dhue) % 181

    res[:, :, 0] = scale_h(res[:, :, 0])
    res[:, :, 1] = np.clip(res[:, :, 1] * dsat, 0, 255)
    res[:, :, 2] = np.clip(res[:, :, 2] * dexp, 0, 255)

    res = cv2.cvtColor(res, cv2.COLOR_HSV2RGB)
    return res


def imcv2_affine_trans(im):
    # Scale and translate
    h, w, c = im.shape
    scale = np.random.uniform() / 20. + 1.
    max_offx = (scale - 1.) * w
    max_offy = (scale - 1.) * h
    offx = int(np.random.uniform() * max_offx)
    offy = int(np.random.uniform() * max_offy)

    im = cv2.resize(im, (0, 0), fx=scale, fy=scale)
    im = im[offy: (offy + h), offx: (offx + w)]
    flip = np.random.uniform() > 0.5
    # if flip:
    #     im = cv2.flip(im, 1)

    return im, [scale, [offx, offy], flip]
