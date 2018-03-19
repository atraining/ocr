#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""some misc utils"""

import os
import cv2
import numpy as np
from PIL import Image
from cfg import Config as cfg
from voc_helper import voc
import codecs

def get_offset(pixel, quad):
    """offset of a pixel and a quad"""
    offset = np.zeros((1, 8))
    offset[0, 0::2] = pixel[1] - quad.crds[:, 0]
    offset[0, 1::2] = pixel[0] - quad.crds[:, 1]
    return offset


def save_image(images, quadrilaterals, save_info):
    """save images and corresponding quadrilaterals to disk"""

    assert len(images) == len(quadrilaterals)
    num_images = len(images)
    for idx in xrange(num_images):
        img = images[idx]
        quads = quadrilaterals[idx]
        img.save(os.path.join(cfg.SAVE_IMAGE,
                              '{}_{}_{}_{}_{}.jpg'.format(cfg.DATASET, save_info[0], save_info[1], save_info[2],idx)))
        with codecs.open(os.path.join(cfg.SAVE_GTS,
                               '{}_{}_{}_{}_{}.txt'.format(cfg.DATASET, save_info[0], save_info[1], save_info[2],idx)), 'w', encoding="utf-8-sig") as gt_txt:
            for quad in quads:
                gt_txt.write('%d,%d,%d,%d,%d,%d,%d,%d,%s\n'%
                             (quad.crds[0, 0], quad.crds[0, 1], quad.crds[1, 0], quad.crds[1, 1],
                              quad.crds[2, 0], quad.crds[2, 1], quad.crds[3, 0], quad.crds[3, 1],
                              quad.label))

def save_cls_map(label_map, save_path):
    """save classification score map using palette in PASCAL VOC
    label_map (ndarray, (H, W))"""
    if cfg.HAVE_NOT_CARE_REGION_USE_PALETTE:
        vc = voc('/mnt/data/VOC/VOCdevkit/VOC2012')
        im = Image.fromarray(label_map, 'P')
        im.putpalette(vc.palette)
        im.save(save_path)
    else:
        im = Image.fromarray(label_map)
        im.save(save_path)

def display_reg_map(offsets, crd_idx):
    """display offset from pixels"""
    offsets = np.absolute(offsets)
    normalized_offset = (offsets[:, :, crd_idx] - np.min(offsets[:, :, crd_idx])) / \
                        (np.max(offsets[:, :, crd_idx]) - np.min(offsets[:, :, crd_idx]))
    brightness = normalized_offset * 255
    brightness = brightness.reshape(int(np.round(cfg.SLIDING_WINDOWS[0] * cfg.FCN_SCALE)),
                                     int(np.round(cfg.SLIDING_WINDOWS[1] * cfg.FCN_SCALE))).transpose()
    reg_img = np.zeros((int(np.round(cfg.SLIDING_WINDOWS[0] * cfg.FCN_SCALE)),
                        int(np.round(cfg.SLIDING_WINDOWS[1] * cfg.FCN_SCALE)), 3), np.uint8)
    reg_img[:, :, 0] = brightness
    reg_img[:, :, 1] = brightness
    reg_img[:, :, 2] = brightness
    cv2.imshow('', reg_img)
    cv2.waitKey(0)

def display_cls_map(label):
    """display cls map from pixels, containing -1 (NOT CARE), 1 (POSITIVE), 0(NEGATIVE)"""
    mask_img = np.zeros((int(np.round(cfg.SLIDING_WINDOWS[0] * cfg.FCN_SCALE)),
                         int(np.round(cfg.SLIDING_WINDOWS[1] * cfg.FCN_SCALE)), 3), np.float32)
    mask_img[np.where(label[:, :, 0] == 1)] = cfg.TEXT_REGION_COLOR
    mask_img[np.where(label[:, :, 0] == -1)] = cfg.NOT_CARE_COLOR
    cv2.imshow('', mask_img)
    cv2.waitKey(0)

# some utils for filter image and quad
def nonblack_ratio(image):
    """nonblack ratio of an image
     image: ndarray
     the last axis is channel
     very smart code!"""
    if type(image) is not np.ndarray:
        image = np.array(image)
    return image.any(axis=-1).sum() * 1.0 / (image.shape[0] * image.shape[1])

def size_nearly_parallelogram(quad, thresh=0.2):
    """1. size enought , height > 16 & width > 16
    2. judge nearly parallelogram to drop irregular quad"""
    dist1 = np.linalg.norm(quad.crds[0] - quad.crds[1])
    dist2 = np.linalg.norm(quad.crds[1] - quad.crds[2])
    dist3 = np.linalg.norm(quad.crds[2] - quad.crds[3])
    dist4 = np.linalg.norm(quad.crds[3] - quad.crds[0])
    if min(dist1, dist2, dist3, dist4) < 16: return False
    if np.abs(dist1-dist3) / min(dist1, dist3) > 0.2: return False
    if np.abs(dist2-dist4) / min(dist2, dist4) > 0.2: return False
    return True


