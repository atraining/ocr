#!/usr/bin/env python
# -*- coding: utf-8 -*-

from DDR.quads_transform import parse_Quadrilateral, display_quadrilaterals_cv, Quadrilateral
from RRCNN.icdar import *
from DDR.utils import *
from RRCNN.cfg import Config as cfg
import os, cv2

DEBUG = True

from common_utilities import polygon_area
def find_order_boxes(inclined_rect, quad):
    """map first vertex in quad to first vertex in inclined rect according distance"""
    if inclined_rect.shape == (4, 1, 2):
        inclined_rect = inclined_rect.reshape((4, 2))

    if polygon_area(inclined_rect) > 0:# must clock-wise
        inclined_rect = inclined_rect[(0, 3, 2, 1), :]
        assert polygon_area(inclined_rect) < 0

    first_ind = -1
    min_dis =  np.inf
    for i in xrange(4):
        dis = np.linalg.norm(inclined_rect[i] - quad[0]) + \
              np.linalg.norm(inclined_rect[(i + 1) % 4] - quad[1]) + \
              np.linalg.norm(inclined_rect[(i + 2) % 4] - quad[2]) + \
              np.linalg.norm(inclined_rect[(i + 3) % 4] - quad[3])
        if dis < min_dis:
            first_ind = i
            min_dis = dis
    assert first_ind != -1
    return inclined_rect[(first_ind, (first_ind + 1) % 4,(first_ind + 2) % 4, (first_ind + 3) % 4 ), :]

def quads2inclinedRects(quads):
    inclined_rects = []
    for quad in quads:
        rect = cv2.minAreaRect(quad.crds)
        box = cv2.cv.BoxPoints(rect)
        box = np.int0(box)
        box = find_order_boxes(box, quad.crds)
        rect = Quadrilateral(label=quad.label)
        rect.crds[...] = box[...]
        inclined_rects.append(rect)
    return inclined_rects

def quads2axisRects(quads):
    axis_rects = []
    for quad in quads:
        x_min = np.min(quad.crds[:, 0])
        x_max = np.max(quad.crds[:, 0])
        y_min = np.min(quad.crds[:, 1])
        y_max = np.max(quad.crds[:, 1])

        rect = Quadrilateral(int(x_min), int(y_min),
                             int(x_max), int(y_min),
                             int(x_max), int(y_max),
                             int(x_min), int(y_max),
                             label=quad.label)
        axis_rects.append(rect)
    return axis_rects

def gen_tasks_gts():
    os.chdir(cfg.DATA_ROOT)
    image_list = os.listdir(cfg.SUB_IMAGE)
    major_list = map(lambda f_name: f_name.split('.')[0], image_list)

    major_file = open(os.path.join(cfg.SAVE_TXT, 'major.txt'), 'w')
    major_file_test = open(os.path.join(cfg.SAVE_TXT, 'major_test.txt'), 'w')
    major_file_train = open(os.path.join(cfg.SAVE_TXT, 'major_train.txt'), 'w')

    for (ind, major) in enumerate(major_list):
        image_path = os.path.join(cfg.DATA_ROOT, cfg.SUB_IMAGE, '{}.jpg'.format(major))
        gt_path = os.path.join(cfg.DATA_ROOT, cfg.SUB_TXT, '{}.txt'.format(major))
        gt_quadrilaterals = parse_Quadrilateral(gt_path)
        im = cv2.imread(image_path)

        gt_quads = np.zeros((len(gt_quadrilaterals), 4, 2), dtype=np.int32)
        for i in range(len(gt_quadrilaterals)):
            gt_quads[i] = gt_quadrilaterals[i].crds
        display_quadrilaterals_cv(im, gt_quads)
        gt_quads = gt_quads.reshape((-1, 4, 2))
        inclined_rects = generate_rbox(gt_quads)
        inclined_rects = inclined_rects.astype(np.int32)

        # inclined_rects = quads2inclinedRects(gt_quadrilaterals)
        # axis_rects = quads2axisRects(gt_quadrilaterals)
        if DEBUG:
            # display_quadrilaterals_cv(im, gt_quadrilaterals)
            display_quadrilaterals_cv(im, inclined_rects)
            # display_quadrilaterals_cv(im, axis_rects)
    major_file.close()
    major_file_test.close()
    major_file_train.close()
