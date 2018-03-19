#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""some common utilties of dataset including ICDAR2017rctw, ICDAR2013, SynthText"""

import os, sys
import cv2
import codecs
import numpy as np
import math
from PIL import Image, ImageFont, ImageDraw, ImageOps
import matplotlib.pyplot as plt
from random import shuffle
import xml.etree.cElementTree as ET
import time

# two data structures

class BoundingBox:
    def __init__(self, x1, y1, x2, y2, label=None, type=None):
        self.label = label
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.type = 'Rectangle'

"""some utils about rotating the quadrilaterals"""

# some utilties about VOC format
def gen_txt_files(image_sets):
    txt_list = [open(os.path.join(image_sets, txt), 'w') for txt in ['train.txt', 'test.txt', 'img_number.txt']]
    return txt_list

def gen_VOC_folders(ast_root):
    folder_list = [os.path.join(ast_root, folder) for folder in ["ImageSets/Main", 'JPEGImages', 'Annotations', 'gts']]
    for d in folder_list:
        if not os.path.exists(d):
            os.makedirs(d)
    return folder_list

def xml_bndbox(bndbox, xmin, ymin, xmax, ymax):
    ET.SubElement(bndbox, "xmin").text = str(xmin)
    ET.SubElement(bndbox, "ymin").text = str(ymin)
    ET.SubElement(bndbox, "xmax").text = str(xmax)
    ET.SubElement(bndbox, "ymax").text = str(ymax)

def xml_quad(quad_obj, quad):
    if quad.type == 'Quadrilateral':
        ET.SubElement(quad_obj, "x0").text = str(quad.crds[0, 0])
        ET.SubElement(quad_obj, "y0").text = str(quad.crds[0, 1])
        ET.SubElement(quad_obj, "x1").text = str(quad.crds[1, 0])
        ET.SubElement(quad_obj, "y1").text = str(quad.crds[1, 1])
        ET.SubElement(quad_obj, "x2").text = str(quad.crds[2, 0])
        ET.SubElement(quad_obj, "y2").text = str(quad.crds[2, 1])
        ET.SubElement(quad_obj, "x3").text = str(quad.crds[3, 0])
        ET.SubElement(quad_obj, "y3").text = str(quad.crds[3, 1])
    elif quad.type == 'Rectangle':
        ET.SubElement(quad_obj, "x0").text = str(quad.x1)
        ET.SubElement(quad_obj, "y0").text = str(quad.y1)
        ET.SubElement(quad_obj, "x1").text = str(quad.x2)
        ET.SubElement(quad_obj, "y1").text = str(quad.y1)
        ET.SubElement(quad_obj, "x2").text = str(quad.x2)
        ET.SubElement(quad_obj, "y2").text = str(quad.y2)
        ET.SubElement(quad_obj, "x3").text = str(quad.x1)
        ET.SubElement(quad_obj, "y3").text = str(quad.y2)


def display(img, boxes):
    for box in boxes:
        if box.type is 'PolyLines':
            pts = np.array([[box.x1, box.y1], [box.x2, box.y2], [box.x3, box.y3], [box.x4, box.y4]], np.int32)
            if box.label.find('#') == -1:
                cv2.polylines(img, [pts], 1, (0, 0, 255), 1)
            else:
                cv2.polylines(img, [pts], 1, (255, 0, 0), 1)
            imS = cv2.resize(img, (960, 540))
            cv2.imshow("", imS)
        else:
            cv2.rectangle(img, (box.x1, box.y1), (box.x2, box.y2), (255, 0, 0), 2)
            cv2.putText(img, box.label, (box.x1, box.y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),2)
            imS = cv2.resize(img, (960, 540))
            cv2.imshow("", imS)
    cv2.waitKey(0)


def show_time(t1=None):
    if t1 == None:
        t1 = time.time()
        print 't1 time is {}'.format(time.asctime(time.localtime(t1)))
        return t1
    else:
        t2 = time.time()
        t = t2 - t1
        h = int(t / 3600)
        m = int((t % 3600) / 60)
        s = int(t % 60)
        sys.stdout.flush()
        print 'h:{},m:{},s:{}'.format(h, m, s)
        print 't2 time is {}'.format(time.asctime(time.localtime(t2)))
        return t2

def polygon_area(poly):
    '''
    compute area of a polygon
    :param poly:
    :return:
    '''
    edge = [
        (poly[1][0] - poly[0][0]) * (poly[1][1] + poly[0][1]),
        (poly[2][0] - poly[1][0]) * (poly[2][1] + poly[1][1]),
        (poly[3][0] - poly[2][0]) * (poly[3][1] + poly[2][1]),
        (poly[0][0] - poly[3][0]) * (poly[0][1] + poly[3][1])
    ]
    return np.sum(edge)/2.

def check_and_validate_quad(quad, (h, w)):
    quad[:, 0] = np.clip(quad[:, 0], 0, w-1)
    quad[:, 1] = np.clip(quad[:, 1], 0, h-1)
    p_area = polygon_area(quad)
    if abs(p_area) < 1:
        # print poly
        print 'invalid poly'
        return None
    if p_area > 0:
        # print 'poly in wrong direction'
        quad = quad[(0, 3, 2, 1), :]
    return quad

def display_boxes_plt(im, quads):
    """display gt quads using plt, support Chinese Font"""
    from matplotlib.font_manager import FontProperties
    ChineseFont = FontProperties(fname='/media/home/netease/PyProjects/test/fonts/SIMYOU.TTF')
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i, box in enumerate(quads):
        bbox = quads[i].crds.reshape(8)
        label = quads[i].label.decode('utf-8')
        ax.add_patch(
            plt.Polygon([(bbox[0], bbox[1]), (bbox[2], bbox[3]),
                         (bbox[4], bbox[5]), (bbox[6], bbox[7])], fill=False,
                        edgecolor='red', linewidth=3.5)
        )
        ax.text(bbox[0], bbox[1] - 2,
                u'{:s}'.format(label),
                bbox=dict(alpha=0.5),
                fontsize=14, color='white', fontproperties=ChineseFont)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    plt.show()

def rotate_image(src, angle, scale=1):
    # if angle > 0, the anti-clockwise
    w, h = src.shape[1], src.shape[0]
    r_angel = np.deg2rad(angle)
    new_w = (abs(h * np.sin(r_angel)) + abs(w * np.cos(r_angel)))*scale
    new_h = (abs(h * np.cos(r_angel)) + abs(w * np.sin(r_angel)))*scale

    rot_mat = cv2.getRotationMatrix2D((new_w/2, new_h/2), angle, scale)

    rot_move = np.dot(rot_mat, np.array([(new_w - w) * 0.5, (new_h - h) * 0.5, 0]))
    rot_mat[0, 2] += rot_move[0]
    rot_mat[1, 2] += rot_move[1]

    return cv2.warpAffine(src, rot_mat, (int(math.ceil(new_w)), int(math.ceil(new_h))), flags=cv2.INTER_LANCZOS4)
    # return cv2.warpAffine(src, rot_mat, (int(math.ceil(new_w)), int(math.ceil(new_h))),
    #                       borderMode=cv2.BORDER_TRANSPARENT,
    #                       borderValue=255)

def rotate_xml(src, crds, angle, scale=1.):
    w, h = src.shape[1], src.shape[0]
    r_angel = np.deg2rad(angle)
    new_w = (abs(h * np.sin(r_angel)) + abs(w * np.cos(r_angel)))*scale
    new_h = (abs(h * np.cos(r_angel)) + abs(w * np.sin(r_angel)))*scale

    rot_mat = cv2.getRotationMatrix2D((new_w * 0.5, new_h * 0.5), angle, scale)
    rot_move = np.dot(rot_mat, np.array([(new_w - w) * 0.5, (new_h - h) * 0.5, 0]))
    rot_mat[0, 2] += rot_move[0]
    rot_mat[1, 2] += rot_move[1]
    point1 = np.dot(rot_mat, np.array([crds[0], crds[1], 1])).reshape(1, 2)
    point2 = np.dot(rot_mat, np.array([crds[2], crds[3], 1])).reshape(1, 2)
    point3 = np.dot(rot_mat, np.array([crds[4], crds[5], 1])).reshape(1, 2)
    point4 = np.dot(rot_mat, np.array([crds[6], crds[7], 1])).reshape(1, 2)
    points = np.array([point1, point2, point3, point4], dtype=np.float32)
    # must accept a 3-d numpy array, (num_points, 1, 2)
    # rx, ry, rw, rh = cv2.boundingRect(points)
    # return reorder_vertexes(points)
    return points