#!/usr/bin/env python
# -*- coding: utf-8 -*-
from utilities import *
from DDR import *

def gen_data():
    data_root = '/mnt/data/scenetext/dataset/icdar13-15/icdar2013/task2.1'
    data_root = '/mnt/data/scenetext/online_label.8.14'
    # data_root = '/mnt/data/scenetext/dataset/icdar13-15/icdar2015/4.1.text_localization/'
    # data_root = '/mnt/data/ICDARdevkit/ICDAR2015/augmentation4/crop/'
    # ast_root = '/mnt/data/scenetext/dataset/icdar13-15/icdar2013/task2.1/ast/test'
    # ast_root = '/mnt/data/scenetext/dataset/icdar13-15/icdar2013/task2.1/ast/train2/ast'
    # ast_root = '/mnt/data/scenetext/dataset/icdar13-15/icdar2015/4.1.text_localization/train/ast/'
    # ast_root = '/mnt/data/ICDARdevkit/ICDAR2015'
    ast_root = '/mnt/data/ICDARdevkit/ICDAR2015/augmentation4/crop/ast/'
    ast_root = '/mnt/data/ICDARdevkit/ICDAR2013'
    ast_root = '/mnt/data/ONLINEdevkit/8.15-6k'
    # image_path = 'Challenge2_Test_Task12_Images'
    # txt_path = 'Challenge2_Test_Task1_GT'
    image_path = 'Challenge2_Training_Task12_Images'
    txt_path = 'Challenge2_Training_Task1_GT'
    image_path = 'images'
    txt_path = 'gts'
    # txt_path = 'ch4_training_localization_transcription_gt'
    # image_path = 'ch4_training_images'
    # image_path = 'images'
    # txt_path = 'ch4_training_localization_transcription_gt'
    # txt_path = 'gts'
    os.chdir(data_root)
    # os.chdir(os.path.join(data_root, txt_path))
    # ICDAR2VOC(ast_root, image_path, txt_path)
    # for gt_txt in os.listdir(os.path.join(data_root, txt_path)):
    #     parse_rectangle(gt_txt)
    # scales = ((300, 300), (700, 700), (700, 500), (700, 300), (1600, 1600))
    # ICDAR2VOC_mutiscales(ast_root, image_path, txt_path, scales)
    ICDAR15ToVOC(ast_root, image_path, txt_path)