#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
class Config:
    DATA_ROOT = '/mnt/data/scenetext/dataset/icdar13-15/icdar2015/4.1.text_localization/'
    SUB_IMAGE = 'ch4_training_images'
    SUB_TXT = 'ch4_training_localization_transcription_gt'

    # DATA_ROOT = '/mnt/data/ICDARdevkit/ICDAR2015/augmentation2/'
    # DATA_ROOT = '/mnt/data/ICDARdevkit/ICDAR2015/debug'
    # DATA_ROOT = '/mnt/data/ICDARdevkit/ICDAR2015/augmentation3'
    # SUB_IMAGE = 'JPEGImages'
    # SUB_TXT = 'gts'

    DEBUG = False
    SAVE_ROOT = '/mnt/data/scenetext/dataset/DDR'
    SAVE_ROOT = '/mnt/data/ICDARdevkit/ICDAR2015/augmentation2/crop'
    SAVE_ROOT = '/mnt/data/ICDARdevkit/ICDAR2015/debug/crop'
    SAVE_ROOT = '/mnt/data/ICDARdevkit/ICDAR2015/augmentation3/crop'
    SAVE_ROOT = '/mnt/data/ICDARdevkit/ICDAR2015/augmentation4/crop'

    if DEBUG: SAVE_ROOT = os.path.join(SAVE_ROOT, 'debug')
    SAVE_IMAGE = os.path.join(SAVE_ROOT, 'images')
    SAVE_GTS = os.path.join(SAVE_ROOT, 'gts')
    SAVE_OFFSET = os.path.join(SAVE_ROOT, 'offsets')
    SAVE_LABEL_MAP = os.path.join(SAVE_ROOT, 'labels')
    SAVE_TXT = os.path.join(SAVE_ROOT, 'txts')

    for path in [SAVE_IMAGE, SAVE_GTS, SAVE_OFFSET, SAVE_LABEL_MAP, SAVE_TXT]:
        if not os.path.exists(path):
            os.mkdir(path)

    DATASET = '2015'
    SLIDING_WINDOWS = (320, 320) # (height, width)
    SLIDING_WINDOWS = (640, 360) # (width, height)
    STRIDE = (320, 320)
    STRIDE = (640, 360)
    # STRIDE = (320, 180)
    # SCALES = (1, 2, 3)
    SCALES = (1,)
    # SCALES = (2, )
    # THETAS = (0, 90, 180, 270)
    THETAS = (0,)
    #after shrink, the min size of a quad
    MIN_SIDE = 8
    # FCN_SCALE = 0.25
    SHRINK_SCALE = 0.3
    #color
    TEXT_REGION_COLOR = (255, 255, 255)
    NOT_CARE_COLOR = (0, 255, 0)

    #NOT CARE
    NOT_CARE_INTERVALS_EXP = [(-1.5, -1), (1, 1.5)]
    CARE_INTERVALS_EXP = [(-1, 1)]


    # HAVE_NOT_CARE_REGION_USE_PALETTE = True
    # FCN_SCALE = 0.25
    # SAVE_LABEL_MAP = os.path.join(SAVE_ROOT, 'labels')

    #MASK
    # have NOT_CARE region and use palette
    HAVE_NOT_CARE_REGION_USE_PALETTE = False
    FCN_SCALE = 1
    SAVE_LABEL_MAP = os.path.join(SAVE_ROOT, 'labels_bin')
