#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""some utilities of icdar2017rctw completation"""

import re
import random
from common_utilities import  *
from DDR.quads_transform import Quadrilateral

def parse_txt(gt_txt):
    gts = []
    with codecs.open(gt_txt, "r", encoding="utf-8-sig") as f_gt:
        frs = f_gt.readlines()
        for line in frs:
            try:
                line = line.strip().split(',')
                gt = Quadrilateral(int(line[0]), int(line[1]),
                                   int(line[2]), int(line[3]),
                                   int(line[4]), int(line[5]),
                                   int(line[6]), int(line[7]),
                                   # label = line[8]) # for rctw2017 test dataset
                                   label = line[9][1:-1])
                gts.append(gt)
            except:
                print 'parse_txt error', gt_txt
    return gts

def gradient(pt1, pt2):
    """compute the gradient of a line"""
    m = None
    # not vertical
    if pt1[0] != pt2[0]:
        m = (1./(pt1[0]-pt2[0]))*(pt1[1] - pt2[1])
    return m

def is_parallel(img, boxes):
    """judge if the box is parallel"""
    thresold = 0.2
    dbox = []
    for box in boxes:
        #horizon
        m1 = gradient((box.x1, box.y1), (box.x2, box.y2))
        m2 = gradient((box.x4, box.y4), (box.x3, box.y3))
        #vertical
        m3 = gradient((box.x1, box.y1), (box.x4, box.y4))
        m4 = gradient((box.x2, box.y2), (box.x3, box.y3))
        if abs(m1-m2) >= thresold and abs(m3-m2) >= thresold:
            print ('current gradient is m1 = %f, m2 = %f'%(m1, m2))
            print ('current gradient is m3 = %f, m4 = %f'%(m3, m4))
            dbox.append(box)
    if len(dbox) == 0: return 1
    else:
        display(img, dbox)
        return 0

def image_crop_polygon(data_root, save_root):
    """crop polygon in the image and save them to png image"""
    s = set()
    with open('/mnt/data/scenetext/icdar2017rctw/icdar2017rctw_train/label.txt','w') as label_file:  # format is 'utf-8'
        for i in range(0, 8345 + 1):
            im = Image.open(os.path.join(data_root, os.path.join('image_{}.jpg'.format(i))))
            gt_txt = os.path.join(data_root, os.path.join('image_{}.txt'.format(i)))
            gts = parse_txt(gt_txt)
            for (ind, box) in enumerate(gts):
                width = int(math.ceil(np.sqrt((box.x2 - box.x1) ** 2 + (box.y2 - box.y1) ** 2)))
                height = int(math.ceil(np.sqrt((box.x4 - box.x1) ** 2 + (box.y4 - box.y1) ** 2)))
                if width < height or box.label.find('#') != -1: continue
                pts = (box.x1, box.y1, box.x2, box.y2, box.x3, box.y3, box.x4, box.y4)
                newIm = im.transform((height, width), Image.QUAD, pts, Image.BICUBIC).transpose(
                    Image.ROTATE_90).transpose(Image.FLIP_TOP_BOTTOM)
                image_name = '{0}_{1}.png'.format(i, ind)
                s = s | set(box.label)
                label_file.write('%s %s\n' % (image_name, box.label.encode('utf-8')))
                newIm.save(os.path.join(save_root, image_name))
            if i%1000 == 0: print 'proceeded %d'%i
    print 'hello world'

    with open('/mnt/data/scenetext/icdar2017rctw/icdar2017rctw_train/map.txt', 'w') as map_file:
        for (ind, label) in enumerate(s):
            map_file.write('%d %s\n' % (ind, label.encode('utf-8')))
    label_file.close()
    map_file.close()

def is_regular_ch(u_ch):
    """judge the unnicode of the ch is a regular(Chinese char, alnum, space) """
    return (u'\u4e00' <= u_ch and u_ch <= u'\u9fa5') or u_ch.isalnum() or u_ch == u' '

def rectify_line(u_line):
    """given a line return the regular string"""
    new_line = u''
    for ch in u_line:
            new_line += u'*' if not is_regular_ch(ch) else ch
    return new_line

def rectify(data_root, sub_dir):
    """rectify map.txt and label.txt
    convert all the characters to * except
    1.Chinese character(u'\u4e00' cc <= u'\u9fa5')
    2. isalnum
    """
    with open(os.path.join(data_root, 'label_new.txt'), 'w') as label_new_txt:
        with open(os.path.join(data_root, 'label.txt'), 'r') as label_txt:
            frs = label_txt.readlines()
            for line in frs:
                line = line.strip().split(' ', 1)  # split only two elements , in case of there is space in line
                label = line[1]
                u_label = label.decode('utf-8')
                new_label = rectify_line(u_label)
                label_new_txt.write('%s %s\n'%(os.path.join(data_root, sub_dir, line[0]), new_label.encode('utf-8')))

    with open(os.path.join(data_root, 'map_new.txt'), 'w') as map_new_txt:
        with open(os.path.join(data_root, 'map.txt'), 'r') as map_txt:
            frs = map_txt.readlines()
            cnt = 0
            for line in frs:
                line = line.strip().split(' ', 1)
                if len(line) <=1: continue
                label = line[1]
                u_label = label.decode('utf-8')
                if is_regular_ch(u_label):
                    map_new_txt.write('%d %s\n'%(cnt,u_label.encode('utf-8')))
                    cnt += 1
            map_new_txt.write('%d %s'%(cnt, '*'))
            print 'all classes is %d'%cnt
    label_new_txt.close()
    label_txt.close()
    map_new_txt.close()
    map_txt.close()

def merge_map():
    """merge map_new.txt in rctw and online"""
    st = set()
    with open('/mnt/data/scenetext/online_label.2.6/line_crop/map_new.txt', 'r') as online_map:
        frs = online_map.readlines()
        for line in frs:
            line = line.strip().split(' ', 1)
            if len(line) <= 1: continue # can't tackle space
            st = st | set(line[1].decode('utf-8'))
    online_map.close()

    with open('/mnt/data/scenetext/icdar2017rctw/icdar2017rctw_train/map_new.txt', 'r') as rctw_map:
        frs = rctw_map.readlines()
        for line in frs:
            line = line.strip().split(' ', 1)
            if len(line) <= 1: continue
            st = st | set(line[1].decode('utf-8'))
    rctw_map.close()

    with open('/mnt/data/scenetext/icdar2017rctw/icdar2017rctw_train/merge/merge_map.txt', 'w') as merge_map_file:
        for (ind, label) in enumerate(st):
            print type(label), label
            merge_map_file.write('%d %s\n' % (ind, label.encode('utf-8')))
        merge_map_file.write('%d %s\n'%(ind+1, ' '))
    merge_map_file.close()

def merge_label():
    merge_lines = []
    with open('/mnt/data/scenetext/online_label.2.6/line_crop/label_new.txt', 'r') as online_label:
        frs = online_label.readlines()
        merge_lines.extend(frs)
    online_label.close()

    print len(merge_lines)
    with open('/mnt/data/scenetext/icdar2017rctw/icdar2017rctw_train/label_new.txt', 'r') as rctw_map:
        frs = rctw_map.readlines()
        merge_lines.extend(frs)

    print len(merge_lines)
    shuffle(merge_lines)

    with open('/mnt/data/scenetext/icdar2017rctw/icdar2017rctw_train/merge/merge_label.txt', 'w') as merge_label_file:
        for line in merge_lines:
            merge_label_file.write('%s'%line)

def filter_label():
    """
    a label list concatinate by hand
    read a raw label list contains non-except char , replace them and gen map.txt"""
    # first walk label_source folder to get label_merge.txt
    label_source_path = '/mnt/data/LINEdevkit/txt/label_source'
    label_merge = []
    for label_source in os.listdir(label_source_path):
        with open(os.path.join(label_source_path, label_source)) as f:
            lines = [line.strip() for line in f.readlines()]
            label_merge.extend(lines)
    print 'all image is %d'%len(label_merge)

    # label_path = '/media/data/LINEdevkit/txt/label_merge.txt'
    label_path_filter = '/mnt/data/LINEdevkit/txt/label_merge_filter.txt'
    label_filter_file = open(label_path_filter, 'w')
    label_filter_val_ls = []
    label_filter_train_ls = []
    st = set()
    for line in label_merge:
        try:
            line = line.decode('utf-8')
            label = line.split(' ', 1)[1]
        except:
            continue
            print line
        label = re.sub(ur'[^\u4e00-\u9fa5a-zA-Z0-9]{1}', '*', label)
        st = st | set(label)
        tmp = line.split(' ', 1)[0] + ' ' + label + '\n'
        label_filter_train_ls.append(tmp.encode('utf-8')) \
            if random.uniform(0, 1) < 0.9 else label_filter_val_ls.append(tmp.encode('utf-8'))
        label_filter_file.write(tmp.encode('utf-8'))
    label_filter_file.close()

    random.shuffle(label_filter_train_ls)
    random.shuffle(label_filter_val_ls)
    with open('/mnt/data/LINEdevkit/txt/label_train.txt', 'w') as f_train:
        for file_train in label_filter_train_ls:
            f_train.write(file_train)

    with open('/mnt/data/LINEdevkit/txt/label_val.txt', 'w') as f_val:
        for file_val in label_filter_val_ls:
            f_val.write(file_val)

    with open('/mnt/data/LINEdevkit/txt/merge_map.txt', 'w') as merge_map_file:
        for (ind, label) in enumerate(st):
            merge_map_file.write('%d %s\n' % (ind, label.encode('utf-8')))
    merge_map_file.close()


def gen_data():
    # data_root = '/mnt/data/scenetext/icdar2017rctw/icdar2017rctw_train/train'
    # save_root = '/mnt/data/scenetext/icdar2017rctw/icdar2017rctw_train/train_crop'
    # image_crop_polygon(data_root, save_root)
    data_root = '/mnt/data/scenetext/icdar2017rctw/icdar2017rctw_train'
    # data_root = '/mnt/data/scenetext/online_label.2.6/line_crop'
    # rectify(data_root, 'train_crop')
    # merge_map()
    # merge_label()
    filter_label()
