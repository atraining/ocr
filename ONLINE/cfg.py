#!/usr/bin/env python
# -*- coding: utf-8 -*-

class Config:
    data_root = '/media/data/scenetext/online/online_label.2.6/img_text/IMG_TEXT'
    # data_root = '/mnt/data/scenetext/online/online_label-2.6-8.14/1502764935087'
    # data_root = '/mnt/data/scenetext/online/online_label-8.14-8.23/1503478588091'
    # data_root = '/mnt/data/scenetext/online/online_label-8.23-10.26/online_label-8.23-10.26'
    # data_root = '/mnt/data/scenetext/online/online_label-10.26-11.27/images'
    # data_root = '/mnt/data/scenetext/online/online_label-11.27-12.20/images'
    data_root = '/mnt/data/scenetext/online/online_label-8.14/1234'
    # data_root = '/mnt/data/scenetext/online/online_label-2.6/test'
    # data_root = '/mnt/data/scenetext/online/online_label-2.6/recheck-1.4w'
    # data_root = '/mnt/data/ONLINEdevkit/test.9.15.1.2k'
    # data_root = '/media/data/scenetext/online/online_label.8.14/images'
    img_tages_dict_path = '/mnt/data/scenetext/online/online_label-2.6/download-recheck-1.4w.csv'
    # img_tages_dict_path = '/mnt/data/scenetext/online/online_label-2.6/image_content_2.6.csv'
    # img_tages_dict_path = '/mnt/data/scenetext/online/online_label-2.6-8.14/annos.xls'
    img_tages_dict_path = '/mnt/data/scenetext/online/online_label-8.14/annos.xls'
    # img_tages_dict_path = '/mnt/data/scenetext/online/online_label-8.14-8.23/annos.xls'
    # img_tages_dict_path = '/mnt/data/scenetext/online/download-8.14-10.26.csv'
    # img_tages_dict_path = '/mnt/data/scenetext/online/online_label-10.26-11.27/download.csv'
    # img_tages_dict_path = '/mnt/data/scenetext/online/online_label-11.27-12.20/download.csv'
    # img_tages_dict_path = '/mnt/data/ONLINEdevkit/test-origin.txt'
    # save_root = '/mnt/data/LINEdevkit/2.6/'
    # save_root = '/mnt/data/scenetext/online/online_label-2.6'
    # save_root = '/mnt/data/scenetext/online/online_label-8.23-10.26'
    # save_root = '/mnt/data/scenetext/online/online_label-10.26-11.27'
    # save_root = '/mnt/data/scenetext/online/online_label-11.27-12.20'
    # save_root = '/mnt/data/scenetext/online/online_label-8.14-8.23'
    # save_root = '/mnt/data/LINEdevkit/2.6-8.14/180'
    save_root = '/mnt/data/LINEdevkit/tmp/'
    # save_root = '/mnt/data/LINEdevkit/8.14-8.23/'
    # save_root = '/mnt/data/LINEdevkit/8.23-10.26/'
    # save_root = '/mnt/data/LINEdevkit/10.26-11.27/'
    # save_root = '/mnt/data/LINEdevkit/8.14/'
    # save_root = '/mnt/data/LINEdevkit/tmp/'

    SUB_IMAGE = 'images'
    SUB_ANNOS = 'annos.xls'
    SUB_SAVE = 'gts'
    SUB_SAVE = 'tmp'
    TRAIN_FILE = 'train.txt'

    DEBUG = True

    decode_utf8 = False
    #some config about ImageLineParser
    # width/height
    MIN_RATIO = 1.2
    BOXES_WIDTH = 5
    BOXES_HEIGHT = 10
    MIN_NUM_BOXES = 1
    MAX_HORIZONTAL_GAP = 140
    INNER_HORIZONTAL_GAP = 40
    MIN_V_OVERLAPS = 0.6
    MIN_SIZE_SIM = 0.6

    MAX_ANGLE = 30
