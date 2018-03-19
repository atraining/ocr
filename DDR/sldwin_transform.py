#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""some transforms about sliding windows which is used to train the detection net
"""
from cfg import Config as cfg
from PIL import Image, ImageDraw
import numpy as np
import cv2
import random

def generate_sldwins(im_shape):
    """generate sldwins on images"""

    width = im_shape[0]
    height = im_shape[1]
    srd_x = cfg.STRIDE[0]
    srd_y = cfg.STRIDE[1]
    sw_x = cfg.SLIDING_WINDOWS[0]
    sw_y = cfg.SLIDING_WINDOWS[1]

    # x_steps = int(np.ceil((width - sw_x) * 1.0/ srd_x)) + 1
    # y_steps = int(np.ceil((height - sw_y)  * 1.0/ srd_y)) + 1
    # rotate the original image so drop the edge sldwins
    x_steps = int((width - sw_x)/ srd_x) + 1
    y_steps = int((height - sw_y)/ srd_y) + 1

    #only consider left up
    left_x = np.arange(0, x_steps) * srd_x
    left_y = np.arange(0, y_steps) * srd_y

    left_x, left_y = np.meshgrid(left_x, left_y)
    sldwins = np.zeros(left_x.shape + (4,), dtype=np.int32)
    # pack the (left_x, left_y, left_x + sw_x, left_y + sw_y) to the meshgrid
    sldwins[:, :, 0] = left_x
    sldwins[:, :, 1] = left_y
    sldwins[:, :, 2] = left_x + sw_x
    sldwins[:, :, 3] = left_y + sw_y
    sldwins = sldwins.reshape((left_x.shape[0] * left_x.shape[1], 4))
    return sldwins

def clip_sldwins(sldwins, im_shape):
    """Clip sldwins to image boundaries."""

    sldwins[:, 0] = np.maximum(np.minimum(sldwins[:, 0], im_shape[0] - 1) ,0)
    sldwins[:, 1] = np.maximum(np.minimum(sldwins[:, 1], im_shape[1] - 1), 0)
    sldwins[:, 2] = np.maximum(np.minimum(sldwins[:, 2], im_shape[0] - 1), 0)
    sldwins[:, 3] = np.maximum(np.minimum(sldwins[:, 3], im_shape[1] - 1), 0)

def rectify_sldwins(sldwins):
    """exlarge clip_sldwins to (320*320)"""

    sldwins[:, 0] = sldwins[:, 2] - cfg.SLIDING_WINDOWS[0]
    sldwins[:, 1] = sldwins[:, 3] - cfg.SLIDING_WINDOWS[1]

def locate_sldwin(im_shape, x, y):
    """same as generate_bboxes, but return the generated sldwins's idx of give (x, y)"""

    width = im_shape[0]
    height = im_shape[1]
    srd_x = cfg.STRIDE[0]
    srd_y = cfg.STRIDE[1]
    sw_x = cfg.SLIDING_WINDOWS[0]
    sw_y = cfg.SLIDING_WINDOWS[1]

    # x_steps = int(np.ceil((width - sw_x) * 1.0 / srd_x)) + 1
    # y_steps = int(np.ceil((height - sw_y) * 1.0 / srd_y)) + 1

    x_steps = int((width - sw_x)/ srd_x) + 1
    y_steps = int((height - sw_y)/ srd_y) + 1

    # only consider left up
    left_x = np.arange(0, x_steps) * srd_x
    left_y = np.arange(0, y_steps) * srd_y

    assert x >=0 and y >= 0
    x_crd = np.searchsorted(left_x, x, 'right')
    y_crd = np.searchsorted(left_y, y, 'right')

    #locate in edge
    if x > left_x[-1] + sw_x or y > left_y[-1] + sw_y: return -1
    return (y_crd - 1) * len(left_x) + x_crd - 1


def display_sldwins(img, sldwins):
    """diplay sldwins"""

    img_draw = ImageDraw.Draw(img)
    print sldwins.shape
    for bbox in sldwins:
        img_draw.rectangle(list(bbox), outline=random.choice(['red', 'green', 'white', 'blue']))
        # img = img.resize((800, 800))
        cv_image = np.array(img)
        cv_image = cv_image[:, :, ::-1]
        cv2.imshow('', cv_image)
        cv2.waitKey(0)

def test_generate_sldwins():
    sldwins = generate_sldwins((1000, 1000))
    print sldwins.shape
    print sldwins
