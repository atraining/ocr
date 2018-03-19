#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""some utils of ICDAR completation"""

import os, shutil, random
from common_utilities import  *
import cv2
import math

def parse_rectangle(gt_txt):
    """in ICDAR13 ch2, gt data is
    158 128 411 181 "Footpath"
    """
    gts = []
    with codecs.open(gt_txt, "r", encoding="utf-8-sig") as f_gt:
        frs = f_gt.readlines()
        for line in frs:
            # line = line.strip().split(',') # for test
            # if "HEIGHT" in line:
            #     print gt_txt
            line = line.strip().split(' ')   # for train
            if len(line[4][1:-1]) == 1:
                print line[4][1:-1]
                print gt_txt
            gt = BoundingBox(int(line[0]),
                           int(line[1]),
                           int(line[2]),
                           int(line[3]),
                           line[4][1:-1])
            gts.append(gt)
    return gts

def ICDAR2VOC(ast_root, image_path, txt_path):
    """convert the ICDAR data to VOC format"""
    fs = os.listdir(image_path)
    fs = map(lambda f_name: f_name.split('.')[0], fs)
    n_image = len(fs)
    print 'processing images number :%d' % n_image
    image_sets, jpeg_images, annotations = gen_VOC_folders(ast_root)
    trainfile, testfile, mapfile = gen_txt_files(image_sets)
    cnt = 0
    ind = 0
    t1 = show_time()
    for f_name in fs:
        img_name = os.path.join(image_path, '{}.jpg'.format(f_name))
        txt_name = os.path.join(txt_path, 'gt_{}.txt'.format(f_name))
        gts = parse_rectangle(txt_name)
        im = cv2.imread(img_name)
        # display(im, gts)
        if im is None: continue
        img_h, img_w, depth = im.shape  # H * W * C
        annotation = ET.Element("annotation")
        size = ET.SubElement(annotation, "size")
        ET.SubElement(size, "width").text = str(img_w)
        ET.SubElement(size, "height").text = str(img_h)
        ET.SubElement(size, "depth").text = str(depth)

        for box in gts:
            label = box.label
            xmin = int(float(box.x1))
            ymin = int(float(box.y1))
            xmax = int(float(box.x2))
            ymax = int(float(box.y2))
            if xmin < 0 or ymin < 0 or xmax > img_w or ymax > img_h: continue  # attention the order , look as a atom operator
            component = ET.SubElement(annotation, "object")
            ET.SubElement(component, "name").text = "1"
            ET.SubElement(component, "txt").text = label
            bndbox = ET.SubElement(component, "bndbox")
            xml_bndbox(bndbox, xmin, ymin, xmax, ymax)
            cnt += 1

        image_id = "{:06d}".format(ind)
        ET.SubElement(annotation, "filename").text = "%s.jpg" % image_id
        tree = ET.ElementTree(annotation)
        tree.write(os.path.join(annotations, "%s.xml" % image_id), encoding="UTF-8")

        trainfile.write(image_id + '\n') if random.uniform(0, 1) > 0.2 else testfile.write(image_id + '\n')
        mapfile.write('%s: %s\n' % (img_name, image_id))

        shutil.copy(img_name, os.path.join(jpeg_images, '%s.jpg' % image_id))
        if ind % 1000 == 0:
            t1 = show_time(t1)
            print 'processed %d images' % ind
        ind += 1
    print 'image number: %d, image with bbs number: %d, all number of bbs %d' % (n_image, ind, cnt)

def ICDAR2VOC_mutiscales(ast_root, image_path, txt_path, scales):
    """convert the ICDAR data to VOC format with mutiscales"""
    fs = os.listdir(image_path)
    fs = map(lambda f_name: f_name.split('.')[0], fs)

    n_image = len(fs)
    print 'processing images number :%d' % n_image

    image_sets, jpeg_images, annotations = gen_VOC_folders(ast_root)
    trainfile, testfile, mapfile = gen_txt_files(image_sets)
    cnt = 0
    ind = 0

    t1 = show_time()

    for f_name in fs:
        img_name = os.path.join(image_path, '{}.jpg'.format(f_name))
        txt_name = os.path.join(txt_path, 'gt_{}.txt'.format(f_name))
        gts = parse_rectangle(txt_name)
        im = cv2.imread(img_name)
        # display(im, gts)
        if im is None: continue

        h, w, d = im.shape  # H * W * C
        for scale in scales:
            interp_mode = random.choice([cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4])
            im_rescale = cv2.resize(im, scale, interpolation=interp_mode)
            x_ratio = scale[0] * 1.0 / w
            y_ratio = scale[1] * 1.0 / h
            gts_rescale = [BoundingBox(
                            int(gt.x1 * x_ratio),
                            int(gt.y1 * y_ratio),
                            int(gt.x2 * x_ratio),
                            int(gt.y2 * y_ratio))
                           for gt in gts]
            # display(im_rescale, gts_rescale)

            img_h, img_w, depth = im_rescale.shape  # H * W * C
            annotation = ET.Element("annotation")
            size = ET.SubElement(annotation, "size")
            ET.SubElement(size, "width").text = str(img_w)
            ET.SubElement(size, "height").text = str(img_h)
            ET.SubElement(size, "depth").text = str(depth)

            # for box in gts:
            for box in gts_rescale:
                label = box.label
                xmin = int(float(box.x1))
                ymin = int(float(box.y1))
                xmax = int(float(box.x2))
                ymax = int(float(box.y2))
                if xmin < 0 or ymin < 0 or xmax > img_w or ymax > img_h: continue  # attention the order , look as a atom operator
                component = ET.SubElement(annotation, "object")
                ET.SubElement(component, "name").text = "1"
                ET.SubElement(component, "txt").text = label
                bndbox = ET.SubElement(component, "bndbox")
                xml_bndbox(bndbox, xmin, ymin, xmax, ymax)
                cnt += 1

            image_id = "{:06d}".format(ind)
            ET.SubElement(annotation, "filename").text = "%s.jpg" % image_id
            tree = ET.ElementTree(annotation)
            tree.write(os.path.join(annotations, "%s.xml" % image_id), encoding="UTF-8")

            trainfile.write(image_id + '\n') if random.uniform(0, 1) > 0.2 else testfile.write(image_id + '\n')
            mapfile.write('%s: %s\n' % (img_name, image_id))

            # shutil.copy(img_name, os.path.join(jpeg_images, '%s.jpg' % image_id))
            cv2.imwrite(os.path.join(jpeg_images, '%s.jpg' % image_id), im_rescale)
            if ind % 1000 == 0:
                t1 = show_time(t1)
                print 'processed %d images' % ind
            ind += 1
    print 'image number: %d, image with bbs number: %d, all number of bbs %d' % (n_image, ind, cnt)

def ICDAR15ToVOC(ast_root, image_path, txt_path):
    """convert ICDAR15 data to VOC format"""
    from DDR.quads_transform import Quadrilateral, parse_Quadrilateral, display_quadrilaterals_cv
    fs = os.listdir(txt_path)
    fs = map(lambda f_name: f_name.split('.')[0], fs)
    n_image = len(fs)
    print 'processing images number :%d' % n_image
    image_sets, jpeg_images, annotations , gts = gen_VOC_folders(ast_root)
    trainfile, testfile, mapfile = gen_txt_files(image_sets)
    cnt = 0
    ind = 0
    t1 = show_time()
    for f_name in fs:
        img_name = os.path.join(image_path, '{}.jpg'.format(f_name))
        txt_name = os.path.join(txt_path, '{}.txt'.format(f_name))
        gt_quads = parse_Quadrilateral(txt_name)
        # gt_quads = parse_rectangle(txt_name)
        im = cv2.imread(img_name)
        # display(im, gt_quads)
        # display_quadrilaterals_cv(im, gt_quads)
        if im is None: continue
        img_h, img_w, depth = im.shape  # H * W * C
        ar = img_w * 1.0 / img_h
        if not (ar > 0.462 and ar < 6.828):
            print 'aspect ratio is too small or too big', ar, img_name
            continue
        annotation = ET.Element("annotation")
        size = ET.SubElement(annotation, "size")
        ET.SubElement(size, "width").text = str(img_w)
        ET.SubElement(size, "height").text = str(img_h)
        ET.SubElement(size, "depth").text = str(depth)

        for quad in gt_quads:
            label = quad.label
            # if np.min(quad.crds) < 1 or np.max(quad.crds[:, 0]) > img_w - 1 or np.max(quad.crds[:, 1] > img_h - 1): continue
            component = ET.SubElement(annotation, "object")
            ET.SubElement(component, "name").text = "text"
            ET.SubElement(component, "txt").text = label
            quad_obj = ET.SubElement(component, "quad")
            xml_quad(quad_obj, quad)
            cnt += 1

        image_id = "{:06d}".format(ind)
        ET.SubElement(annotation, "filename").text = "%s.jpg" % image_id
        tree = ET.ElementTree(annotation)
        tree.write(os.path.join(annotations, "%s.xml" % image_id), encoding="UTF-8")

        trainfile.write(image_id + '\n') if random.uniform(0, 1) > 0.2 else testfile.write(image_id + '\n')
        mapfile.write('%s: %s\n' % (img_name, image_id))

        shutil.copy(img_name, os.path.join(jpeg_images, '%s.jpg' % image_id))
        if ind % 1000 == 0:
            t1 = show_time(t1)
            print 'processed %d images' % ind
        ind += 1