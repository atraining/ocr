#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ONLINE.cfg import Config
import re, os
from DDR.quads_transform import *
from cfg import Config as cfg
from line_parser import LineParser
from other import clip_boxes
from common_utilities import *
from line_utils import parse_csv
"""some utils about crop line png from original image
    maybe copied from ads-text project
    this is for online_label_8.14 which is labeled as quad"""

DEBUG = True
def parse_xls(xls_path):
    """xls file"""
    import xlrd
    bk = xlrd.open_workbook(xls_path)
    sh = bk.sheet_by_name("Sheet1")
    nrows = sh.nrows
    img_tages_dict = {}
    for i in range(1, nrows):
        row_data = sh.row_values(i)
        img_tages_dict[row_data[1]] = row_data[2]  # for /media/netease/ssd2/data/scenetext/OCR/ocr-6.9/result-copy.xls
    return img_tages_dict


def parse_txt(txt_path):
    """txt file"""
    img_tages_dict = {}
    with open(txt_path, 'r') as f:
        for line in f.readlines():
            line = line.strip().split(' ')
            img_tages_dict[line[0]] = line[1]
    return img_tages_dict

def parse_tags(img_tags_dict, img_name, decode_utf8=True):
    '''parse the tags of the img_name, and return the ground truth bbs of CC (not a line)'''
    gts = []
    if img_name not in img_tags_dict:
        print 'filter  not in dic', img_name
        return None
    tags = img_tags_dict[img_name]
    if decode_utf8:
        tags = tags.decode('utf-8')
    # attention: 'ur' is 'unicode' !!!
    # tags = re.sub(ur'##[^\u4e00-\u9fa5a-zA-Z0-9]{1,}##', '', tags)
    tags = re.sub(ur'##[^\u4e00-\u9fa5a-zA-Z0-9]{1}##', '*', tags)
    tags = re.sub(ur'##[^\u4e00-\u9fa5a-zA-Z0-9]{2}##', '**', tags)
    # tags = re.sub(ur'##[^\u4e00-\u9fa5a-zA-Z0-9]{3}##', '***', tags)
    # tags = re.sub(ur'##[^\u4e00-\u9fa5a-zA-Z0-9]{4}##', '****', tags)
    for tag in tags.split(';'):
        try:
            gt = Quadrilateral(int(float(tag.split(',', 8)[0])),
                                  int(float(tag.split(',', 8)[1])),
                                  int(float(tag.split(',', 8)[2])),
                                  int(float(tag.split(',', 8)[3])),
                                  int(float(tag.split(',', 8)[4])),
                                  int(float(tag.split(',', 8)[5])),
                                  int(float(tag.split(',', 8)[6])),
                                  int(float(tag.split(',', 8)[7])),
                                   label = tag.split(',', 8)[-1])
        except:
            print 'filter tag error', img_name, tag
            continue
        gts.append(gt)
    return gts

def write_gts_txt(gts, f_gt):
    for gt in gts:
        quad = gt.crds
        f_gt.write(str(int(quad[0, 0])) + ',' + str(int(quad[0, 1])) + ',' +
                   str(int(quad[1, 0])) + ',' + str(int(quad[1, 1])) + ',' +
                   str(int(quad[2, 0])) + ',' + str(int(quad[2, 1])) + ',' +
                   str(int(quad[3, 0])) + ',' + str(int(quad[3, 1])) + ',' + gt.label + '\n')

def parse_online():
    save_dir = os.path.join(Config.DATA_ROOT, Config.SUB_SAVE)
    train_list = open(os.path.join(Config.DATA_ROOT, Config.TRAIN_FILE), 'w')
    file_list = os.listdir(os.path.join(Config.DATA_ROOT, Config.SUB_IMAGE))
    file_list = [file_name for file_name in file_list if file_name.rsplit('.', 1)[1] != 'gif']
    img_tags_dict = parse_xls(os.path.join(Config.DATA_ROOT, Config.SUB_ANNOS))
    cnt = 0
    for file_name in file_list:
        gts = parse_tags(img_tags_dict, file_name)
        if gts is None:
            cnt += 1
            continue
        train_list.write('{}.txt\n'.format(file_name.rsplit('.', 1)[0]))
        gt_txt = os.path.join(save_dir, '{}.txt'.format(file_name.rsplit('.', 1)[0]))
        f_gt = codecs.open(gt_txt, "w", encoding="utf-8-sig")
        write_gts_txt(gts, f_gt)
        f_gt.close()
    print 'total images: %d, filter images: %d, contained images: %d'%(len(file_list), cnt, len(file_list) - cnt)
    train_list.close()


def split_lines(gt_lines):
    """given a list of Quad , split to
    lines: ndarray
    lines_label: list of label"""
    if len(gt_lines) == 0:
        return None, None
    lines = np.zeros((len(gt_lines), 4), dtype=np.float32)
    lines_label = []
    for ind, gt in enumerate(gt_lines):
        # rect = cv2.minAreaRect(gt.crds)
        # box = cv2.cv.BoxPoints(rect)
        # box = np.int0(box)
        lines[ind, 0] = min(gt.crds[:, 0])
        lines[ind, 1] = min(gt.crds[:, 1])
        lines[ind, 2] = max(gt.crds[:, 0])
        lines[ind, 3] = max(gt.crds[:, 1])
        lines_label.append(gt.label)
    return lines, lines_label

def crop_line2png(img_tages_dict, data_root, save_root):
    """to train the CRNN, we need line annotation,
    1. crop line and save to png image,
    2. write the png filename and its label to txt file(label.txt)
    3. create the map.txt according the label.txt"""

    file_list = os.listdir(data_root)
    file_list = [file_name for file_name in file_list if file_name.rsplit('.', 1)[1] != 'gif']
    # file_list = ['1480324751948_w5ya8rmffq2us889.jpeg']
    print 'processing images number :%d' % len(file_list)
    s = set()
    counter = 0
    line_parse = LineParser()
    with open(os.path.join(save_root, 'label.txt'), 'w') as label_file:
        for i,img_name in enumerate(file_list):
            im = cv2.imread(os.path.join(data_root, img_name))
            if im is None: continue
            # if np.min(im.shape[:2]) < 400 : continue
            gts_line = parse_tags(img_tages_dict, img_name, Config.decode_utf8)
            if gts_line is None: continue

            # for convient, split Quad struct to (lines, lines_label)
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

                    angle = np.rad2deg(
                        np.arctan((poly[start_pt][1] - poly[end_pt][1]) * 1.0 / (poly[start_pt][0] - poly[end_pt][0])))
                    im_new = rotate_image(im_copy, angle)
                    crds = list(bbox.reshape((-1)))
                    quad = rotate_xml(im_copy, crds, angle)
                    quad = quad.reshape((4, 2))
                x0 = np.min(quad[:, 0])
                y0 = np.min(quad[:, 1])
                x1 = np.max(quad[:, 0])
                y1 = np.max(quad[:, 1])
                # just for debug
                height = y1 - y0
                expand_ratio = 0.2


                # lines, lines_label = split_lines(gts_line)
                # lines = clip_boxes(lines, im.shape)
                # lines, lines_label = line_parse.filter_lines(lines, lines_label)

                new_image_name = '{0}_{1}.{2}'.format(img_name.rsplit('.', 1)[0], j, img_name.rsplit('.', 1)[1])
                s = s | set(gt_line.label)
                label_file.write('%s %s\n' %(new_image_name, gt_line.label.encode('utf-8')))

                expand_y0 = max(0, int(y0) - expand_ratio * height)
                expand_y1 = min(im_new.shape[0], int(y1) + expand_ratio * height)
                expand_x0 = max(0, int(x0) - expand_ratio * height)
                expand_x1 = min(im_new.shape[1], int(x1) + expand_ratio * height)
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

def crop_line2txt(img_tages_dict, data_root, save_root):
    """write line quad to txt file"""
    file_list = os.listdir(data_root)
    file_list = [file_name for file_name in file_list if file_name.rsplit('.', 1)[1] != 'gif']
    print 'processing images number :%d' % len(file_list)
    counter = 0
    # file_list = ['1477621480837_sbgifm20jejqyay6.jpeg']
    with open(os.path.join(save_root, 'file_path.txt'), 'w') as f_list:
        if not os.path.exists(os.path.join(save_root, 'txts')):
            os.makedirs(os.path.join(save_root, 'txts'))
        for i, img_name in enumerate(file_list):
            im = cv2.imread(os.path.join(data_root, img_name))
            if im is None: continue
            print img_name
            gts_line = parse_tags(img_tages_dict, img_name, Config.decode_utf8)
            if gts_line is None: continue
            txt_name = img_name.replace(img_name.split('.')[1], 'txt')
            txt_path = os.path.join(save_root, 'txts', txt_name)
            f_list.write('{}\n'.format(os.path.join(data_root, img_name)))
            with open(txt_path, 'w') as f_txt:
                for j, gt_line in enumerate(gts_line):
                    crds = gt_line.crds.astype(np.int)
                    print gt_line.label
                    f_txt.write('%d,%d,%d,%d,%d,%d,%d,%d\n'%(crds[0][0], crds[0][1],
                                                           crds[1][0], crds[1][1],
                                                           crds[2][0], crds[2][1],
                                                           crds[3][0], crds[3][1]))
                if cfg.DEBUG:
                    display_quadrilaterals_cv(im, gts_line)
            counter += 1
            if i % 1000 == 0: print 'proceeded %d' % i
    print 'gen line images : %d' %counter

def gen_line_quad():
    """the original interface to gen CRNN training data"""
    # img_tags_dict = parse_csv(cfg.img_tages_dict_path)
    img_tags_dict = parse_xls(cfg.img_tages_dict_path)
    # crop_line2png(img_tags_dict, cfg.data_root, cfg.save_root)
    crop_line2txt(img_tags_dict, cfg.data_root, cfg.save_root)

def gen_online_gts():
    """the test current RRCNN model, parse line annotation from test dataset
    1. get gts according annotations list
    2. gen a zip file to do test the same as IC15 task1 f-measure test
    3. must confirm the gts_quad is clock-wise
    4. gt_img_{}.txt is the same order in test_list.txt and
    detection model all read test_list.txt"""
    img_tags_dict = parse_txt(cfg.img_tages_dict_path)
    file_list = open('/mnt/data/ONLINEdevkit/test_list.txt', 'w')
    import zipfile
    zip_file_name = os.path.join(cfg.save_root, 'gts-online')
    zip_file = zipfile.ZipFile(zip_file_name, 'w')
    cnt = 1
    print 'len of tags', len(img_tags_dict)
    for img_name in img_tags_dict:
        im = cv2.imread(os.path.join(cfg.data_root, img_name))
        if im is None:
            print 'im is None', img_name
            continue
        quads = parse_tags(img_tags_dict, img_name)
        if quads is None:
            print 'quads is None', img_name
            continue
        file_list.write(img_name + '\n')
        gt_file_path = os.path.join(cfg.save_root, 'zip', 'gt_img_{}.txt'.format(cnt))

        with codecs.open(gt_file_path, 'w', encoding="utf-8-sig") as gt_txt:
            for quad in quads:
                if polygon_area(quad.crds) > 0:  # must clock-wise
                    print 'not clock-wise', quad.crds
                    quad.crds = quad.crds[(0, 3, 2, 1), :]
                    assert polygon_area(quad.crds) < 0
                gt_txt.write('%d,%d,%d,%d,%d,%d,%d,%d,%s\n'%
                             (quad.crds[0, 0], quad.crds[0, 1], quad.crds[1, 0], quad.crds[1, 1],
                              quad.crds[2, 0], quad.crds[2, 1], quad.crds[3, 0], quad.crds[3, 1],
                              quad.label))
        zip_file.write(gt_file_path, os.path.basename(gt_file_path))
        cnt += 1
    print 'num test images is %d\n'%cnt
    file_list.close()

