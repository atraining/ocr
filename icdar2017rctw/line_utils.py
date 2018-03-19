#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""very similar to ONLINE/utils.py"""

import os
import cv2
import re
import numpy as np
from icdar2017rctw.cfg import Config as cfg
from icdar2017rctw.utilities import parse_txt
from ONLINE.line_parser import LineParser
from ONLINE.utils import split_lines
from ONLINE.text_proposal_connector import clip_boxes
from common_utilities import rotate_image, rotate_xml

def crop_line2png(data_root, save_root):
    """to train the CRNN, we need line annotation,
    1. crop line and save to png image,
    2. write the png filename and its label to txt file(label.txt)
    3. create the map.txt according the label.txt"""

    file_list = os.listdir(data_root)
    file_list = [file_name for file_name in file_list if file_name.rsplit('.', 1)[1] == 'jpg']
    print 'processing images number :%d' % len(file_list)
    s = set()
    counter = 0
    with open(os.path.join(save_root, 'label.txt'), 'w') as label_file:
        for i,img_name in enumerate(file_list):
            im = cv2.imread(os.path.join(data_root, img_name))
            if im is None: continue
            if np.min(im.shape[:2]) < 400 : continue
            gt_txt_path = os.path.join(cfg.data_root, img_name.rsplit('.', 1)[0] + '.txt')
            # gt_txt_path = os.path.join(cfg.gt_root, 'task2_' + img_name.rsplit('.', 1)[0] + '.txt')
            gts_line = parse_txt(gt_txt_path)
            gts_line = [line for line in gts_line if line.label != '###']
            for line in gts_line:
                # replace all non{char, alnum} to *
                line.label = re.sub(ur'[^\u4e00-\u9fa5a-zA-Z0-9]{1}', '*', line.label)
            gts_line = [line for line in gts_line if line.label.count('*') < len(line.label)]
            if gts_line is None or len(gts_line) == 0: continue
            # for convient, split Quad struct to (lines, lines_label)
            # line_parse = LineParser()
            # lines, lines_label = split_lines(gts_line)
            # lines = clip_boxes(lines, im.shape)
            # lines, lines_label = line_parse.filter_lines(lines, lines_label)

            if not os.path.exists(os.path.join(save_root, 'images')):
                os.makedirs(os.path.join(save_root, 'images'))
            for j,gt_line in enumerate(gts_line):
                rect = cv2.minAreaRect(gt_line.crds)
                box = cv2.cv.BoxPoints(rect)
                bbox = np.int0(box)
                if len(gt_line.label) == 1 and gt_line.label != '*':
                    quad = bbox.reshape((4, 2))
                    im_new = im
                elif np.linalg.norm(gt_line.crds[0] - gt_line.crds[1]) < np.linalg.norm(gt_line.crds[0] - gt_line.crds[3]) \
                    and (len(gt_line.label) >=2 or gt_line.label == '*'):
                    continue
                else:
                    im_copy = im.copy()
                    poly = bbox.reshape((4, 2))
                    # is vertical text
                    p_lowest = np.argmax(poly[:, 1])
                    p_lowest_right = (p_lowest - 1) % 4
                    p_lowest_left = (p_lowest + 1) % 4
                    if np.linalg.norm(poly[p_lowest] - poly[p_lowest_right]) > np.linalg.norm(
                                    poly[p_lowest] - poly[p_lowest_left]):
                        start_pt = p_lowest
                        end_pt = p_lowest_right
                    else:
                        start_pt = p_lowest_left
                        end_pt = p_lowest
                    try:
                        angle = np.rad2deg(
                            np.arctan((poly[start_pt][1] - poly[end_pt][1]) * 1.0 / (poly[start_pt][0] - poly[end_pt][0])))
                        im_new = rotate_image(im_copy, angle)
                        crds = list(bbox.reshape((-1)))
                        quad = rotate_xml(im_copy, crds, angle)
                        quad = quad.reshape((4, 2))
                    except:
                        continue
                x0 = np.min(quad[:, 0])
                y0 = np.min(quad[:, 1])
                x1 = np.max(quad[:, 0])
                y1 = np.max(quad[:, 1])
                # just for debug
                height = y1 - y0
                expand_ratio = 0.2

                new_image_name = '{0}_{1}.{2}'.format(img_name.rsplit('.', 1)[0], j, img_name.rsplit('.', 1)[1])
                s = s | set(gt_line.label)
                label_file.write('%s %s\n' %(new_image_name, gt_line.label.encode('utf-8')))

                expand_y0 = int(max(0, int(y0) - expand_ratio * height))
                expand_y1 = int(min(im_new.shape[0], int(y1) + expand_ratio * height))
                expand_x0 = int(max(0, int(x0) - expand_ratio * height))
                expand_x1 = int(min(im_new.shape[1], int(x1) + expand_ratio * height))
                line_image = im_new[expand_y0: expand_y1, expand_x0: expand_x1]
                if cfg.DEBUG:
                    print gt_line.label.encode('utf-8')
                    cv2.imshow('', line_image)
                    cv2.waitKey(0)
                if not os.path.exists(os.path.join(save_root, 'images')):
                    os.mkdir(os.path.join(save_root, 'images'))
                cv2.imwrite(os.path.join(save_root, 'images', new_image_name), line_image)
                counter += 1
            if i % 1000 == 0: print 'proceeded %d' % i

    print 'gen line images : %d' %counter
    with open(os.path.join(save_root, 'map.txt'), 'w') as map_file:
        for (ind, label) in enumerate(s):
            map_file.write('%d %s\n' % (ind, label.encode('utf-8')))
    label_file.close()
    map_file.close()


def gen_line_rctw():
    crop_line2png(cfg.data_root, cfg.save_root)