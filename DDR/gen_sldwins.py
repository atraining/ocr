#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""some preprocess about the DDR paper:
 2017-Liu-arxiv-Deep Direct Regression for Multi-Oriented Scene Text Detection"""


import os
from cfg import Config
from sldwin_transform import *
from quads_transform import *
from utils import *
DEBUG = False

def generate_cropped_gts(image, sldwins, quadrilaterals):
    """generate image and its gts, note only positive cropped image(contains gts)
    quadrilaterals: gts in original image
    """
    sldwin_imgs = []
    sldwin_gts = []

    sldwins_mp = {}
    for quad_idx, quad in enumerate(quadrilaterals):
        for crd_idx, crd in enumerate(quad.crds):
            # assume sldwin have no overlap ,so only a sldwin contain this quad
            sldwin_idx = locate_sldwin(image.size, crd[0], crd[1])
            if sldwin_idx == -1: continue
            # locate the vertex crds and create a map
            if sldwin_idx not in sldwins_mp:
                sldwins_mp[sldwin_idx] = dict()
                sldwins_mp[sldwin_idx][quad_idx] = list()
            else:
                if quad_idx not in sldwins_mp[sldwin_idx]:
                    sldwins_mp[sldwin_idx][quad_idx] = list()
            sldwins_mp[sldwin_idx][quad_idx].append(crd_idx)

    for sldwin_idx in sldwins_mp:
        cropped_image = image.crop(sldwins[sldwin_idx])
        quad_mp = sldwins_mp[sldwin_idx]
        clipped_quads = []
        for quad_idx in quad_mp:
            inside_idx_list = quad_mp[quad_idx]
            clipped_quad = rectify_quadrilateral(sldwins[sldwin_idx], quadrilaterals[quad_idx], inside_idx_list)
            # if clipped_quad is not None: clipped_quads.append(clipped_quad)
            if is_valid_quadrilateral(clipped_quad): clipped_quads.append(clipped_quad)
        # if len(clipped_quads) > 0:
        valid_quad = is_RRCNN_quad(cropped_image, clipped_quads)
        if valid_quad:
            sldwin_imgs.append(cropped_image)
            # sldwin_gts.append(clipped_quads)
            sldwin_gts.append(valid_quad)
            if DEBUG:
                # display_quadrilaterals(cropped_image, clipped_quads)
                display_quadrilaterals(cropped_image, valid_quad)

    return sldwin_imgs, sldwin_gts

def is_RRCNN_quad(im, quads):
    """this func is special for rotation data augmentation of RRCNN
    average 3 images for a angle, so I filter some
    1. black area ratio < 0.2
    2. IOU(quad, inclinedRects) > thresh (default = 0.9) (at last use nearly parallel)
    3. height > 16, width > 16 the same as faster-rcnn RPN
    4. then num_quad > 3"""
    # quads = [quad for quad in quads if quad.label != '###']
    # if nonblack_ratio(im) < 0.7: return None
    valid_quad = [quad for quad in quads if size_nearly_parallelogram(quad)]
    if len(valid_quad) < 2: return None
    return valid_quad

def generate_data(image, quadrilaterals, save_info):
    """use a sliding window to crop image and re-compute gt quadrilaterals"""

    sldwins = generate_sldwins(image.size)
    # if DEBUG:
    #     display_sldwins(image, sldwins)
    sldwin_imgs, sldwin_gts = generate_cropped_gts(image, sldwins, quadrilaterals)
    save_image(sldwin_imgs, sldwin_gts, save_info)

def gen_sldwins():
    """scale the image
       random rotate the image
       crop the above image with a sliding windows (320, 320, 160)"""

    os.chdir(Config.DATA_ROOT)
    image_list = os.listdir(Config.SUB_IMAGE)
    # major_list = map(lambda f_name: f_name.split('.')[0], image_list)

    # open a img_number.txt to map train.txt to ICDAR2015
    f_map = open('/mnt/data/ICDARdevkit/ICDAR2015/ImageSets/Main/img_number.txt')
    map = dict()
    for line in f_map.readlines():
        map[line.split(' ')[1].strip()] = line.split(' ')[0].split('/')[1][:-5]

    f_train = open('/mnt/data/ICDARdevkit/ICDAR2015/ImageSets/Main/train.txt')
    major_list = [line.strip() for line in f_train.readlines()]

    if DEBUG:
        num_images = len(image_list)
        print 'processing images number :%d' % num_images

    # major_list = ['img_196', 'img_197','img_198','img_199','img_200','img_201']
    # major_list = ['img_850']
    # major_list = ['000001_0']
    for major_idx,major in enumerate(major_list):
        print major_idx
        image_path = os.path.join(Config.DATA_ROOT, Config.SUB_IMAGE, '{}.jpg'.format(map[major]))
        gt_path = os.path.join(Config.DATA_ROOT, Config.SUB_TXT, 'gt_{}.txt'.format(map[major]))
        # gt_path = os.path.join(Config.DATA_ROOT, Config.SUB_TXT, '{}.txt'.format(major))
        gt_quadrilaterals = parse_Quadrilateral(gt_path)

        img = Image.open(image_path)

        width, height = img.size
        # if DEBUG:
        #     display_quadrilaterals(img, gt_quadrilaterals)
        for scale_idx, scale in enumerate(Config.SCALES):
            img = img.resize((width * scale, height * scale))
            scaled_quads = scale_quadrilaterals(gt_quadrilaterals, scale)
            # if DEBUG:
            #     display_quadrilaterals(img, scaled_quads)
            # theta = random.choice(Config.THETAS)
            theta = 0
            theta_idx = random.choice(range(0, len(Config.THETAS)))
            # for theta_idx, theta in enumerate(Config.THETAS):
            rotated_quads = rotate_quadrilaterals(scaled_quads, theta, (width * scale, height * scale))
            rot_img = img.rotate(theta, expand=1)
            # if DEBUG:
            #     display_quadrilaterals(rot_img, rotated_quads)
            save_info = [major, scale_idx, theta_idx]
            generate_data(rot_img, rotated_quads, save_info)
