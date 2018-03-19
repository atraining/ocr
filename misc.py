#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""some misc utils run seperately"""

import random
import numpy as np
import os, cv2
import codecs
from config import Config as cfg
from DDR.quads_transform import parse_Quadrilateral, display_quadrilaterals_cv
from data_augmentation import is_ordered
from common_utilities import polygon_area


"""split major.txt inot test.txt and train.txt"""
def split_major():
    """split major.txt into major_tain.txt and major_test.txt"""
    f = open('/mnt/data/scenetext/online/txts/version2/file_path_all_2.txt', 'r')
    image_list = [f_name.strip() for f_name in f.readlines()]
    image_list = np.array(image_list)
    split(image_list, )
    f.close()

def split(image_list, ratio=0.9):
    num_image = len(image_list)
    num_train = int(num_image * ratio)
    rng = range(0, num_image)
    random.shuffle(rng)
    train_list = image_list[rng[:num_train]]
    test_list = image_list[rng[num_train:]]
    write_txt(train_list, test_list)

    return train_list, test_list

def write_txt(train_list, test_list):
    f_train = open('/mnt/data/scenetext/online/txts/version2/train.txt', 'w')
    f_test = open('/mnt/data/scenetext/online/txts/version2/test.txt', 'w')
    for train in train_list:
        f_train.write('%s\n' %train)
    for test in test_list:
        f_test.write('%s\n' %test)
    f_train.close()
    f_test.close()

def split_gt():
    """remove last field in ICDAR15 gt data"""
    icdar15_data_set = '/mnt/data/scenetext/dataset/icdar13-15/icdar2015/4.1.text_localization/ch4_training_localization_transcription_gt'
    os.chdir(icdar15_data_set)
    gt_txt_list = os.listdir('.')
    for gt_txt in gt_txt_list:
        f_save = open(os.path.join('../gt_non_difficult', gt_txt), 'w')
        with codecs.open(gt_txt, "r", encoding="utf-8-sig") as f_gt:
            frs = f_gt.readlines()
            for line in frs:
                line = line.strip()
                if line.split(',')[-1] == '###': continue # icdar15 test only for non_difficulty
                line = ''.join(s + ',' for s in line.split(',', 8)[:8])
                line = line[:-1]
                f_save.write(line+'\n')
        f_save.close()

def filter_quads(quads):
    non_ordered_quads = [quad for quad in quads if not is_ordered(quad.crds)]
    return non_ordered_quads

def filter_clockwise_quads(quads):
    anticlockwise_quads = [quad for quad in quads if not polygon_area(quad.crds) > 0]
    return anticlockwise_quads

def check_icdar15_ordered():
    """check if all the annotation is ordered"""
    os.chdir(cfg.ICDAR15_DATA_ROOT)
    image_list = os.listdir(cfg.ICDAR15_SUB_IMAGE)
    major_list = map(lambda f_name: f_name.split('.')[0], image_list)
    # os.chdir(cfg.AUGMENTATION_DATA_ROOT)
    # image_list = os.listdir(cfg.AUGMENTATION_SUB_IMAGE)

    for (ind, major) in enumerate(major_list):
        image_path = os.path.join(cfg.ICDAR15_DATA_ROOT, cfg.ICDAR15_SUB_IMAGE, '{}.jpg'.format(major))
        gt_path = os.path.join(cfg.ICDAR15_DATA_ROOT, cfg.ICDAR15_SUB_TXT, 'gt_{}.txt'.format(major))
        # image_path = os.path.join(cfg.AUGMENTATION_DATA_ROOT, cfg.AUGMENTATION_SUB_IMAGE, '{}.jpg'.format(major))
        # gt_path = os.path.join(cfg.AUGMENTATION_DATA_ROOT, cfg.AUGMENTATION_SUB_TXT, '{}.txt'.format(major))
        gt_quadrilaterals = parse_Quadrilateral(gt_path)
        non_ordered_quads = filter_clockwise_quads(gt_quadrilaterals)
        if len(non_ordered_quads) > 0:
            im = cv2.imread(image_path)
            display_quadrilaterals_cv(im, non_ordered_quads)

def result_to_submit():
    """
    1. use py-R-FCN to produce bbs , firstly conf=-np.conf, second nms
    2. then use this func to save submit-{conf}.zip
    3. assume that erery line is
    767,412,822,412,822,440,767,440,score"""

    result_dir = '/mnt/data/scenetext/dataset/icdar13-15/icdar2015/4.1.text_localization/result/model.0.0.1/submit'
    confs = np.arange(0.4, 1, 0.1)
    major_list = os.listdir(result_dir)
    import zipfile
    import shutil
    for conf in confs:
        zip_file_name = os.path.join(result_dir.rsplit('/', 1)[0], 'submit-{}'.format(conf))
        zip_file = zipfile.ZipFile(zip_file_name, 'w' )
        save_dir = os.path.join(result_dir.rsplit('/', 1)[0], 'submit{}'.format(conf))
        for major in major_list:
            gt_path = os.path.join(result_dir, major)
            filtered_lines = [line.strip() for line in open(gt_path).readlines() if
                              float(line.strip().split(',')[-1]) > conf]
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            f_txt = open(os.path.join(save_dir, major), 'w')
            for line in filtered_lines:
                f_txt.write(line.rsplit(',', 1)[0]+'\n')
            f_txt.close()
            zip_file.write(os.path.join(save_dir, major),  os.path.basename(os.path.join(save_dir, major)))
        zip_file.close()
        shutil.rmtree(save_dir)

def img_proc():
    with open('/home/netease/scenetext/crnn-git/data/char4/1_2.png', 'r') as f:
        imageBin = f.read()
        imageBuf = np.fromstring(imageBin, dtype=np.uint8)
        img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
        cv2.imshow('', img)


if __name__ == '__main__':
    # split_gt()
    # result_to_submit()
    # check_icdar15_ordered()
    # img_proc()
    split_major()