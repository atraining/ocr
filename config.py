#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""some common config"""
class Config:
    ICDAR15_DATA_ROOT = '/mnt/data/scenetext/dataset/icdar13-15/icdar2015/4.1.text_localization/'
    ICDAR15_SUB_IMAGE = 'ch4_training_images'
    # ICDAR15_SUB_IMAGE = 'ch4_test_images'
    ICDAR15_SUB_TXT = 'ch4_training_localization_transcription_gt'
    # ICDAR15_SUB_TXT = 'ch4_test_localization_transcription_gt'
    ICDAR13_DATA_ROOT = '/mnt/data/scenetext/dataset/icdar13-15/icdar2013/task2.1/'
    ICDAR13_SUB_IMAGE = 'Challenge2_Training_Task12_Images'
    ICDAR13_SUB_TXT = 'Challenge2_Training_Task1_GT'
    AUGMENTATION_DATA_ROOT = '/mnt/data/ICDARdevkit/ICDAR2015/augmentation4/'
    AUGMENTATION_SUB_IMAGE = 'JPEGImages'
    AUGMENTATION_SUB_TXT = 'gts'
