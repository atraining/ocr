#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gen_sldwins import *
from gen_tasks_gts import *
from cfg import Config as cfg
from quads_transform import parse_Quadrilateral, display_quadrilaterals
from utils import display_reg_map, display_cls_map

"""a interface to ROOT/test.py, containing
    1. test_gen_sldwins
    2. test_gen_task_gts
    3. test_sldwins
    4. test_task_gts"""

def test_gen_sldwins():
    """test the utils of DDR"""
    gen_sldwins()

def test_gen_task_gts():
    gen_tasks_gts()

def test_sldwins():
    """assuming cropped image have been saved in SAVE_ROOT/images and SAVE_ROOT/gts folder,
    this function random read image and its gts to display"""
    num_display = 20000

    images_list = os.listdir(cfg.SAVE_IMAGE)
    major_list = map(lambda f_name : f_name.split('.')[0], images_list)
    for i in xrange(num_display):
        major = random.choice(major_list)
        gt_path = os.path.join(cfg.SAVE_GTS, '{}.txt'.format(major))
        image_path = os.path.join(cfg.SAVE_IMAGE, '{}.jpg'.format(major))
        gt_quadrilaterals = parse_Quadrilateral(gt_path)
        img = Image.open(image_path)
        print 'displaying %s' %major
        display_quadrilaterals(img, gt_quadrilaterals)

def test_task_gts():
    """assuming task gts have been saved in SAVE_ROOT/labels & SAVE_ROOT/offsets & SAVE_ROOT/txts
    this function random read gts to dispaly"""
    num_display = 20000

    f = open(os.path.join(cfg.SAVE_TXT, 'major.txt'))
    major_list = [f_name.strip() for f_name in f.readlines()]
    for major in major_list:
        print major
        label = np.load(os.path.join(Config.SAVE_LABEL_MAP, '{}_label.npy'.format(major)))
        display_cls_map(label)
        offset = np.load(os.path.join(Config.SAVE_OFFSET, '{}_offset.npy'.format(major)))
        display_reg_map(offset, 0)
    f.close()
