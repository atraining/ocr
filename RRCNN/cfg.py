#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
class Config:
    DATA_ROOT = '/mnt/data/scenetext/dataset/icdar13-15/icdar2015/4.1.text_localization/'
    DATA_ROOT = '/mnt/data/ICDARdevkit/ICDAR2015/aug-non-reorder'
    SUB_IMAGE = 'ch4_training_images'
    SUB_IMAGE = 'JPEGImages'
    SUB_TXT = 'ch4_training_localization_transcription_gt'
    SUB_TXT = 'gts'

    DEBUG = False
    SAVE_ROOT = '/mnt/data/scenetext/dataset/RRCNN'

    if DEBUG: SAVE_ROOT = os.path.join(SAVE_ROOT, 'debug')
    SAVE_IMAGE = os.path.join(SAVE_ROOT, 'images')
    SAVE_GTS = os.path.join(SAVE_ROOT, 'gts')
    SAVE_OFFSET = os.path.join(SAVE_ROOT, 'offsets')
    SAVE_LABEL_MAP = os.path.join(SAVE_ROOT, 'labels')
    SAVE_TXT = os.path.join(SAVE_ROOT, 'txts')

    DATASET = '2015'
    SLIDING_WINDOWS = (320, 320) # (height, width)
    STRIDE = (320, 320)
    SCALES = (1, 2, 3)
    # SCALES = (2, )
    THETAS = (-90, -75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 90)
