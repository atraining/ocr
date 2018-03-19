#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""some transform of the gt quadrilaterals"""

import random
import math
import codecs
import cv2
import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import LineString
import copy
from cfg import Config as cfg


class Quadrilateral:
    def __init__(self, x1=0, y1=0, x2=0, y2=0, x3=0, y3=0, x4=0, y4=0, label=None, type=None):
        self.label = label
        self.crds = np.zeros((4, 2), dtype=np.int32) # note this !!!
        self.crds[0, 0] = x1
        self.crds[0, 1] = y1
        self.crds[1, 0] = x2
        self.crds[1, 1] = y2
        self.crds[2, 0] = x3
        self.crds[2, 1] = y3
        self.crds[3, 0] = x4
        self.crds[3, 1] = y4
        self.type = 'Quadrilateral'

    def get_adj_point(self, (x, y)):
        """get two idx of adj points given a point(x, y)"""

        cur_idx = self.locate_point((x, y))
        if cur_idx == -1: return None
        pre_idx = (cur_idx - 1 + 4) % 4
        next_idx = (cur_idx + 1) % 4
        return np.vstack(self.crds[pre_idx,: ], self.crds[next_idx, :])

    def locate_point(self, (x,y)):
        for idx in range(0, 4):
            if self.crds[idx, 0] == x and self.crds[idx, 1] == y: return idx
        return -1

def parse_Quadrilateral(gt_txt):
    """in ICDAR15 ch4, gt data is ﻿738,32,773,35,773,54,738,51,MAX"""
    gts = []
    with codecs.open(gt_txt, "r", encoding="utf-8-sig") as f_gt:
        frs = f_gt.readlines()
        for line in frs:
            label = ''.join(s + ',' for s in line.strip().split(',')[8:])
            label = label[:-1]
            line = line.strip().split(',')
            gt = Quadrilateral(int(np.round(float(line[0]))), int(np.round(float(line[1]))),
                               int(np.round(float(line[2]))), int(np.round(float(line[3]))),
                               int(np.round(float(line[4]))), int(np.round(float(line[5]))),
                               int(np.round(float(line[6]))), int(np.round(float(line[7]))),
                               label)
            gts.append(gt)
    return gts

def display_quadrilaterals(img, quadrilaterals):
    img_draw = ImageDraw.Draw(img)
    for quadrilateral in quadrilaterals:
        pts = ([(quadrilateral.crds[0, 0], quadrilateral.crds[0, 1]), (quadrilateral.crds[1, 0], quadrilateral.crds[1, 1]),
                (quadrilateral.crds[2, 0], quadrilateral.crds[2, 1]), (quadrilateral.crds[3, 0], quadrilateral.crds[3, 1])
                ])
        img_draw.polygon(pts, outline=(255, 0, 0))
    img = img.resize((800, 800))
    cv_image = np.array(img)
    cv_image = cv_image[:, :, ::-1]
    cv2.imshow('', cv_image)
    cv2.waitKey(0)

def display_quadrilaterals_cv(im, quadrilaterals):
    for quadrilateral in quadrilaterals:
        pts = quadrilateral
        pts = quadrilateral.crds
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(im, [pts], True, (0, 0, 255))
        cv2.putText(im, 'fir', (pts[0, 0, 0], pts[0, 0, 1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(im, 'sec', (pts[1, 0, 0], pts[1, 0, 1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(im, 'thi', (pts[2, 0, 0], pts[2, 0, 1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(im, 'for', (pts[3, 0, 0], pts[3, 0, 1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    im = cv2.resize(im, (800, 800))
    cv2.imshow('', im)
    cv2.waitKey(0)


def rotate_point(x1, y1, theta, width, height):
    """after rotating, the origin point is changed, so re-compute the coordinates"""
    rotated_x1 = 0
    rotated_y1 = 0
    if theta == 0:
        rotated_x1 = x1
        rotated_y1 = y1
    elif theta == 90:
        rotated_x1 = y1
        rotated_y1 = width - x1
    elif theta == 180:
        rotated_x1 = width - x1
        rotated_y1 = height - y1
    elif theta == 270:
        rotated_x1 = height - y1
        rotated_y1 = x1
    return rotated_x1, rotated_y1

def rotate_quadrilaterals(quadrilaterals, theta, size):
    """becasue the background is not fixed and there are only four angle(0, pi/2, pi, pi/2*3)
    so re-compute the quadrilaterals coordinates accoridng the new origin point"""
    width, height = size
    rotated_quads = []
    for quadrilateral in quadrilaterals:
        x1, y1 = rotate_point(quadrilateral.crds[0, 0], quadrilateral.crds[0, 1], theta, width, height)
        x2, y2 = rotate_point(quadrilateral.crds[1, 0], quadrilateral.crds[1, 1], theta, width, height)
        x3, y3 = rotate_point(quadrilateral.crds[2, 0], quadrilateral.crds[2, 1], theta, width, height)
        x4, y4 = rotate_point(quadrilateral.crds[3, 0], quadrilateral.crds[3, 1], theta, width, height)
        rotated_quads.append(Quadrilateral(x1, y1, x2, y2, x3, y3, x4, y4, label=quadrilateral.label))
    return rotated_quads

def scale_quadrilaterals(quadrilaterals, scale=1):
    """scale the quadrilaterals"""
    scale_quadrilaterals = []
    for quadrilateral in quadrilaterals:
        x1 ,y1 = quadrilateral.crds[0, 0] * scale, quadrilateral.crds[0, 1] * scale
        x2, y2 = quadrilateral.crds[1, 0] * scale, quadrilateral.crds[1, 1] * scale
        x3, y3 = quadrilateral.crds[2, 0] * scale, quadrilateral.crds[2, 1] * scale
        x4, y4 = quadrilateral.crds[3, 0] * scale, quadrilateral.crds[3, 1] * scale
        scale_quadrilaterals.append(Quadrilateral(int(np.round(x1)), int(np.round(y1)), int(np.round(x2)), int(np.round(y2)),
                                                  int(np.round(x3)), int(np.round(y3)), int(np.round(x4)), int(np.round(y4)),
                                                   label=quadrilateral.label))
    return scale_quadrilaterals

def rectify_quadrilateral(sldwin, quadrilateral, inside_idx_list):
    """crop original image also crop the gt quadrilaterals
    so re-compute the gt quadrilateral in the croped 320*320 sldwin"""


    #shallow copy and overwrite the vertex coordinates
    #for complex sub-object(such as nested range list1), copy.copy only copy a reference (A and B referce to list1), so modify
    #A.list1 will modify B.list1
    # but copy.deepcopy recreate a list ,so A.list1 and B.list1 is independent
    clipped_quad = copy.deepcopy(quadrilateral)
    num_crd = len(inside_idx_list)
    # only consider cropped gts is quad
    if num_crd == 4:
        clipped_quad.crds[:, 0] -= sldwin[0]
        clipped_quad.crds[:, 1] -= sldwin[1]
        return clipped_quad
    elif num_crd != 2:
        return None

    # two LineString represent two polygons
    line_sldwin = LineString([(sldwin[0], sldwin[1]), (sldwin[2],sldwin[1]),
                           (sldwin[2],sldwin[3]),(sldwin[0], sldwin[3]),
                           (sldwin[0], sldwin[1])])
    line_quad = LineString([(quadrilateral.crds[0, 0], quadrilateral.crds[0, 1]),
                            (quadrilateral.crds[1, 0], quadrilateral.crds[1, 1]),
                            (quadrilateral.crds[2, 0], quadrilateral.crds[2, 1]),
                            (quadrilateral.crds[3, 0], quadrilateral.crds[3, 1]),
                            (quadrilateral.crds[0, 0], quadrilateral.crds[0, 1])
                            ])
    ist_pts = line_sldwin.intersection(line_quad)
    #maybe intersection point is a line
    if type(ist_pts) == LineString:
        clipped_quad.crds[:, 0] -= sldwin[0]
        clipped_quad.crds[:, 1] -= sldwin[1]
        return clipped_quad

    # if cropped gt is not quad , drop it
    if len(ist_pts) != 2: return None
    # map intersection point idx to gt point idx
    ist_pts_idx_list = intersection_position(ist_pts, quadrilateral, inside_idx_list)
    if len(ist_pts_idx_list) != 2: return None

    clipped_quad.crds[ist_pts_idx_list[0], 0] = ist_pts[0].x
    clipped_quad.crds[ist_pts_idx_list[0], 1] = ist_pts[0].y

    if type(ist_pts[1]) == LineString: return None

    clipped_quad.crds[ist_pts_idx_list[1], 0] = ist_pts[1].x
    clipped_quad.crds[ist_pts_idx_list[1], 1] = ist_pts[1].y
    clipped_quad.crds[:, 0] -= sldwin[0]
    clipped_quad.crds[:, 1] -= sldwin[1]

    clipped_quad.crds[:, 0] = np.maximum(np.minimum(clipped_quad.crds[:, 0], cfg.SLIDING_WINDOWS[0] - 1), 0)
    clipped_quad.crds[:, 1] = np.maximum(np.minimum(clipped_quad.crds[:, 1], cfg.SLIDING_WINDOWS[1] - 1), 0)
    return clipped_quad


def intersection_position(ist_pts, quadrilateral, inside_idx_list):
    """map intersection point to gt point in order"""

    assert len(inside_idx_list) == 2
    assert len(ist_pts) == 2
    ist_pts_idx_list = []
    line_out_idx_list = []
    lines = []

    # construct two LineString
    adj_idx = [(inside_idx_list[0] - 1 + 4)%4, (inside_idx_list[0] + 1) % 4]
    adj_idx.remove(inside_idx_list[1])
    line_out_idx_list.append(adj_idx)
    lines.append(
        LineString([(quadrilateral.crds[adj_idx, 0], quadrilateral.crds[adj_idx, 1]),
                             (quadrilateral.crds[inside_idx_list[0], 0], quadrilateral.crds[inside_idx_list[0], 1]
                              )])
    )
    adj_idx = [(inside_idx_list[1] - 1 + 4)%4, (inside_idx_list[1] + 1) % 4]
    adj_idx.remove(inside_idx_list[0])
    line_out_idx_list.append(adj_idx)
    lines.append(
        LineString([(quadrilateral.crds[adj_idx, 0], quadrilateral.crds[adj_idx, 1]),
                             (quadrilateral.crds[inside_idx_list[1], 0], quadrilateral.crds[inside_idx_list[1], 1]
                              )])
    )
    for ist_pt_idx, ist_pt in enumerate(ist_pts):
        for line_idx, line in enumerate(lines):
            if line.distance(ist_pt) < 1e-8:
                ist_pts_idx_list.append(line_out_idx_list[line_idx])
                break

    return ist_pts_idx_list

def is_valid_quadrilateral(quadrilateral):
    if quadrilateral is None: return False
    return True
    # min_side = 2000
    # for crd_idx in xrange(4):
    #     next_idx = (crd_idx + 1) % 4
    #     min_side = min(min_side, np.linalg.norm(quadrilateral.crds[crd_idx] - quadrilateral.crds[next_idx]))
    # return False if min_side < cfg.MIN_SIDE else True

def is_NOT_CARE_quadrilateral(quadrilateral):
    """NOT CARE: -1
       CARE: 1
       OTHER: 0"""
    if quadrilateral is None: return None
    # find the shortest side
    min_side = 2000
    for crd_idx in xrange(4):
        next_idx = (crd_idx + 1) % 4
        min_side = min(min_side, np.linalg.norm(quadrilateral.crds[crd_idx] - quadrilateral.crds[next_idx]))
    # is NOT CARE?
    for interval in cfg.NOT_CARE_INTERVALS_EXP:
        min_value = 32 * pow(2, interval[0])
        max_value = 32 * pow(2, interval[1])
        if min_side > min_value and min_side < max_value:
            return True

    for interval in cfg.CARE_INTERVALS_EXP:
        min_value = 32 * pow(2, interval[0])
        max_value = 32 * pow(2, interval[1])
        if min_side > min and min_side < max:
            return False


def shrink_quadrilateral(quadrilateral, scale=0.3):
    """shrink quadrilateral accroding EAST"""

    r = [None] * 4
    for i in xrange(4):
        next_idx = (i + 1) % 4
        pre_idx = (i + 4 - 1) % 4
        r[i] =min(np.linalg.norm(quadrilateral.crds[i] - quadrilateral.crds[pre_idx]),
                   np.linalg.norm(quadrilateral.crds[i] - quadrilateral.crds[next_idx]))
    # find the longer side
    edges_pair = [None] * 2
    edges_pair[0] = [LineString([(quadrilateral.crds[0, 0], quadrilateral.crds[0, 1]),
                                 (quadrilateral.crds[1, 0], quadrilateral.crds[1, 1])]),
                     LineString([(quadrilateral.crds[3, 0], quadrilateral.crds[3, 1]),
                                 (quadrilateral.crds[2, 0], quadrilateral.crds[2, 1])]
                                )]
    edges_pair[1] = [LineString([(quadrilateral.crds[1, 0], quadrilateral.crds[1, 1]),
                                 (quadrilateral.crds[2, 0], quadrilateral.crds[2, 1])]),
                     LineString([(quadrilateral.crds[0, 0], quadrilateral.crds[0, 1]),
                                 (quadrilateral.crds[3, 0], quadrilateral.crds[3, 1])]
                                )]
    first_len = min(edges_pair[0][0].length, edges_pair[0][1].length)
    second_len = min(edges_pair[1][0].length, edges_pair[1][1].length)
    #shrink longer edges
    first_idx = 0 if first_len > second_len else 1
    if first_idx == 0:
        edge0 = edges_pair[0][0]
        edge2 = edges_pair[0][1]
        x0, y0 = shrink_crds(edge0, scale * r[0])
        x1, y1 = shrink_crds(edge0, scale * r[1], reverse=True)
        x3, y3 = shrink_crds(edge2, scale * r[3])
        x2, y2 = shrink_crds(edge2, scale * r[2], reverse=True)

        # modify edge3, edge4
        edge1 = LineString([(x1, y1), (x2, y2)])
        edge3 = LineString([(x0, y0), (x3, y3)])
        if edge1.length < cfg.MIN_SIDE or edge3.length < cfg.MIN_SIDE: return None
        x1, y1 = shrink_crds(edge1, scale * r[1])
        x2, y2 = shrink_crds(edge1, scale * r[2], reverse=True)
        x0, y0 = shrink_crds(edge3, scale * r[0])
        x3, y3 = shrink_crds(edge3, scale * r[3], reverse=True)
    else:  #shrink shorter edge
        edge1 = edges_pair[1][0]
        edge3 = edges_pair[1][1]
        x1, y1 = shrink_crds(edge1, scale * r[1])
        x2, y2 = shrink_crds(edge1, scale * r[2],reverse=True)
        x0, y0 = shrink_crds(edge3, scale * r[0])
        x3, y3 = shrink_crds(edge3, scale * r[3],reverse=True)

        # modify edge3, edge4
        edge0 = LineString([(x0, y0), (x1, y1)])
        edge2 = LineString([(x3, y3), (x2, y2)])
        if edge0.length < cfg.MIN_SIDE or edge2.length < cfg.MIN_SIDE: return None
        x0, y0 = shrink_crds(edge0, scale * r[0])
        x1, y1 = shrink_crds(edge0, scale * r[1], reverse=True)
        x3, y3 = shrink_crds(edge2, scale * r[3])
        x2, y2 = shrink_crds(edge2, scale * r[2], reverse=True)

    return Quadrilateral(x0, y0, x1, y1, x2, y2, x3, y3, label=quadrilateral.label)


def shrink_crds(edge, distance, reverse=False):
    """get new crds of line end-point,
       distance: shrink distance
       reverse=False: (start_point ---> end_point)"""
    assert distance >= 0
    left , right = edge.boundary
    left_x, left_y, right_x, right_y = left.x, left.y, right.x, right.y
    ratio = distance / edge.length
    if reverse:
        x = int(np.round(right_x - (right_x - left_x) * ratio))
        y = int(np.round(right_y - (right_y - left_y) * ratio))
    else:
        x = int(np.round(left_x + (right_x - left_x) * ratio))
        y = int(np.round(left_y + (right_y - left_y) * ratio))

    return x, y

