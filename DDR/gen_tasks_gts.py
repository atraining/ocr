#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""generate ground truth label for multi-task. 1. classification  2. regression
unlike pixel-element, we use the method proposed in EAST, that is shrink two endpoint of a line
in the end , the side is 0.4*original_size"""

from quads_transform import *
from  cfg import Config as cfg
import os, cv2
from utils import *

DEBUG = False


def gen_cls_map(ori_quads, shrunk_quads, scale=0.25):
    """generate classification socre map according quads"""
    assert len(ori_quads) == len(shrunk_quads)
    mask_quads_pixel = []
    scaled_ori_quads = scale_quadrilaterals(ori_quads, scale=scale)
    scaled_shrunk_quads = scale_quadrilaterals(shrunk_quads, scale=scale)
    # Create a mask image that contains the quad filled in
    mask_img = np.zeros((int(np.round(cfg.SLIDING_WINDOWS[0] * scale)),
                         int(np.round(cfg.SLIDING_WINDOWS[1] * scale)), 3))

    ori_contours = np.zeros((len(scaled_ori_quads), 4, 2), dtype=np.int32)
    shrunk_contours = np.zeros((len(scaled_shrunk_quads), 4, 2), dtype=np.int32)

    num_quads = len(scaled_shrunk_quads)
    for quad_idx in xrange(num_quads):
        shrunk_contours[quad_idx, :, :] = scaled_shrunk_quads[quad_idx].crds
        ori_contours[quad_idx, :, :] = scaled_ori_quads[quad_idx].crds

    positive_quads = []
    for contour_idx in xrange(num_quads):
        # a RGB mask image
        mask_blank = np.zeros((int(np.round(cfg.SLIDING_WINDOWS[0] * scale)),
                             int(np.round(cfg.SLIDING_WINDOWS[1] * scale))))

        ori_quad = ori_quads[contour_idx]
        if is_NOT_CARE_quadrilateral(ori_quad):
            cv2.drawContours(mask_img, ori_contours, contour_idx, color=cfg.NOT_CARE_COLOR, thickness=-1)
            # scaled_shrunk_quads.remove(scaled_shrunk_quads[contour_idx])
        else:
            cv2.drawContours(mask_img, ori_contours, contour_idx, color=cfg.NOT_CARE_COLOR, thickness=-1)
            cv2.drawContours(mask_img, shrunk_contours, contour_idx, color=cfg.TEXT_REGION_COLOR, thickness=-1)
            cv2.drawContours(mask_blank, shrunk_contours, contour_idx, color=255, thickness=-1)
            mask_quads_pixel.append(np.vstack((np.where(mask_blank == 255)[0],
                                     np.where(mask_blank == 255)[1])).transpose())
            positive_quads.append(scaled_shrunk_quads[contour_idx])
    if DEBUG:
        cv2.imshow("", mask_img)
        cv2.waitKey(0)

    # convert mask_img to label (W, H) to FCN
    label_map = np.zeros((int(np.round(cfg.SLIDING_WINDOWS[0] * scale)),
                         int(np.round(cfg.SLIDING_WINDOWS[1] * scale))), dtype=np.uint8)
    if cfg.HAVE_NOT_CARE_REGION_USE_PALETTE: # palette image
        label_map[np.where(mask_img[:, :, 1] == 255)] = 255
        label_map[np.where(mask_img[:, :, 0] == 255)] = 1
    else:# gray image
        label_map[np.where(mask_img[:, :, 1] == 255)] = 255
        label_map[np.where(mask_img[:, :, 0] == 255)] = 255

    return label_map, mask_img, mask_quads_pixel, positive_quads

def gen_reg_map(quads_pixels, quads, scale=0.25):
    """compute offset from every pixel in mask_pixels to four vertexs"""

    assert len(quads_pixels) == len(quads)
    offsets = np.zeros((int(np.round(cfg.SLIDING_WINDOWS[0]*scale)),
                        int(np.round(cfg.SLIDING_WINDOWS[1]*scale)), 8))
    num_quad = len(quads)
    for quad_idx in xrange(num_quad):
        mask_pixels = quads_pixels[quad_idx]
        quad = quads[quad_idx]
        for pixel in mask_pixels:
            offsets[pixel[1], pixel[0], :] = get_offset(pixel, quad)
    if DEBUG:
        # show left_upper
        display_reg_map(offsets, 0)
    return offsets

def gen_shrunk_quads(im, quadrilaterals):
    shrunk_quads = []
    ori_quads = []
    quads = []
    for quad in quadrilaterals:
        shrunk_quad = shrink_quadrilateral(quad)
        if is_valid_quadrilateral(shrunk_quad):
            shrunk_quads.append(shrunk_quad)
            ori_quads.append(quad)
        quads.append(quad)
        if DEBUG:
            display_quadrilaterals_cv(im, quads)
            display_quadrilaterals_cv(im, shrunk_quads)
    return ori_quads, shrunk_quads

def gen_tasks_gts():
    major_list = ['2015_img_0_2_1_4', '2015_img_0_2_1_1','2015_img_0_2_1_2', '2015_img_0_2_1_3']
    # major_list = ['2015_img_0_2_1_4', '2015_img_0_2_0_4']
    # major_list = ['2015_img_850_1_1_0']
    major_list = [major.split('.')[0] for major in os.listdir(cfg.SAVE_IMAGE)]

    major_file = open(os.path.join(cfg.SAVE_TXT, 'major.txt'), 'w')
    major_file_test = open(os.path.join(cfg.SAVE_TXT, 'major_test.txt'), 'w')
    major_file_train = open(os.path.join(cfg.SAVE_TXT, 'major_train.txt'), 'w')


    for (ind, major) in enumerate(major_list):
        image_path = os.path.join(cfg.SAVE_IMAGE, '{}.jpg'.format(major))

        im = cv2.imread(image_path)
        gt_path = os.path.join(cfg.SAVE_GTS, '{}.txt'.format(major))
        gt_quadrilaterals = parse_Quadrilateral(gt_path)
        ori_quads, shrunk_quads = gen_shrunk_quads(im, gt_quadrilaterals)
        label_map, mask_img, mask_quads_pixel, scaled_quads = gen_cls_map(ori_quads, shrunk_quads,scale=cfg.FCN_SCALE)
        assert len(mask_quads_pixel) == len(scaled_quads)
        if len(mask_quads_pixel) > 0:
            if ind % 1000 == 0:
                print 'pocessed %d/%d'%(ind+1, len(major_list))
            major_file.write('%s\n' % major)
            major_file_train.write('%s\n' % major) if random.uniform(0, 1) > 0.2 else major_file_test.write('%s\n' % major)
            offsets = gen_reg_map(mask_quads_pixel, scaled_quads,scale=cfg.FCN_SCALE)
            # save label and offset
            save_cls_map(label_map, os.path.join(cfg.SAVE_LABEL_MAP, '{}_label.png'.format(major)))

            # np.save(os.path.join(cfg.SAVE_OFFSET, '{}_offset'.format(major)), offsets)
    major_file.close()
    major_file_test.close()
    major_file_train.close()
