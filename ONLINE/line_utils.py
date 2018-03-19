#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""some utils about crop line png from original image
    maybe copied from ads-text project"""

import os, cv2
import random
import numpy as np
from ONLINE.cfg import Config
from line_parser import LineParser

def parse_xls(xls_path):
    """xls file"""
    import xlrd
    bk = xlrd.open_workbook(xls_path)
    sh = bk.sheet_by_name("Sheet1")
    nrows = sh.nrows
    img_tages_dict = {}
    for i in range(1, nrows):
        row_data = sh.row_values(i)
        # img_tages_dict[row_data['nos_path']] = row_data['tags']
        img_tages_dict[row_data[1]] = row_data[2]  # for /mnt/data/scenetext/online_label.4.10.id/download.xls
        # img_tages_dict[row_data[0]] = row_data[2]  # for /media/netease/ssd2/data/scenetext/OCR/ocr-6.9/result-copy.xls
    return img_tages_dict

def parse_csv(csv_path):
    """read the csv file and return a dict:{'img_name', 'tages'}"""
    import csv
    img_tages_dict = {}
    with open(csv_path, 'rb') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            img_tages_dict[row['nosPath']] = row['tags']
    return img_tages_dict


def crop_line2png(img_tages_dict, data_root, save_root):
    """to train the CRNN, we need line annotation,
    1. crop line and save to png image,
    2. write the png filename and its label to txt file(label.txt)
    3. create the map.txt according the label.txt"""

    file_list = os.listdir(data_root)
    # import random
    random.shuffle(file_list)
    # file_list = ['1480325136111_2gga323vslbuvays.png']
    print 'processing images number :%d' % len(file_list)
    s = set()
    counter = 0
    with open(os.path.join(save_root, 'label.txt'), 'w') as label_file:
        for i,img_name in enumerate(file_list):
            # if counter > 100: break
            if img_name not in img_tages_dict: continue
            im = cv2.imread(os.path.join(data_root, img_name))
            if im is None: continue
            if np.min(im.shape[:2]) < 400 : continue

            line_parse = LineParser(img_tages_dict[img_name], im.shape)
            lines, lines_label = line_parse.gen_line()
            if Config.DEBUG:
                char_boxes, char_labels = line_parse.get_all_boxes(line_parse.tags, decode_utf8=Config.decode_utf8)
                line_parse.display_boxes(im, char_boxes, char_labels)
                line_parse.display_boxes(im, lines, lines_label)

            if not os.path.exists(os.path.join(save_root, 'images')):
                os.makedirs(os.path.join(save_root, 'images'))
            for j,line in enumerate(lines):
                new_image_name = '{0}_{1}.{2}'.format(img_name.rsplit('.', 1)[0], j, img_name.rsplit('.', 1)[1])
                s = s | set(lines_label[j])
                label_file.write('%s %s\n' %(new_image_name, lines_label[j].encode('utf-8')))
                line_image = im[int(line[1]): int(line[3]), int(line[0]): int(line[2])]
                cv2.imwrite(os.path.join(save_root, 'images', new_image_name), line_image)

                #just for debug
                for k in xrange(5):
                    height = int(line[3]) - int(line[1])
                    shift_x = random.uniform(0, 0.4)
                    shift_y = random.uniform(0, 0.4)
                    line_image2 = im[max(0, int(line[1] - shift_y*height)):
                                    min(int(line[3] + (0.4-shift_y)*height), im.shape[0]),
                                    max(0, int(line[0] - shift_x*height)):
                                    min(int(line[2] + (0.4-shift_x)*height), im.shape[1])]
                    new_image_name = '{0}_{1}_{2}.{3}'.format(img_name.rsplit('.', 1)[0], j, k, img_name.rsplit('.', 1)[1])
                    cv2.imwrite(os.path.join(save_root, 'images', new_image_name), line_image2)
                counter += 1
            if i % 1000 == 0: print 'proceeded %d' % i

    print 'gen line images : %d' %counter
    with open(os.path.join(save_root, 'map.txt'), 'w') as map_file:
        for (ind, label) in enumerate(s):
            map_file.write('%d %s\n' % (ind, label.encode('utf-8')))

def save_line_annos(img_tages_dict, data_root, save_root):
    """to re-check the effect of original single char annos to lines,
    1. get lines
    2. write to csv file
        format:
        [[
            1477568421447_wzrceuthja20s1gm.jpeg,1:3:567:36:#;6:46:49:77:#;4:82:567:112:#;4:126:384:155:#;3:165:280:193:#;298:171:323:194:#;343:168:370:193:#;394:166:571:195:#;6:208:481:238:#
            IMAGE_NAME,line1_X0:line1_y0:line1_x1:line1_y1;line1_label:line2_X0:line2_y0:line2_x1:line2_y1:line2_label;
            there are no seprate symbol in label, such as [, : ;], onlye Chinese char, English char, number, *
        ]]
    """

    file_list = os.listdir(data_root)
    # import random
    # random.shuffle(file_list)
    # file_list = ['1480325230423_zqjcdg86z23vhpyk.jpeg']
    print 'processing images number :%d' % len(file_list)
    s = set()
    counter = 0
    counter_image = 0
    with open(os.path.join(save_root, 'line_annos.txt'), 'w') as line_file:
        for i,img_name in enumerate(file_list):
            if img_name not in img_tages_dict: continue
            im = cv2.imread(os.path.join(data_root, img_name))
            if im is None: continue
            # if np.min(im.shape[:2]) < 400 : continue

            line_parse = LineParser(img_tages_dict[img_name], im.shape)
            lines, lines_label = line_parse.gen_line()
            if Config.DEBUG:
                char_boxes, char_labels = line_parse.get_all_boxes(line_parse.tags, decode_utf8=Config.decode_utf8)
                line_parse.display_boxes(im, char_boxes, char_labels)
                line_parse.display_boxes(im, lines, lines_label)
            if len(lines) > 0:
                line_file.write('%s,'%img_name)
                for j,line in enumerate(lines):
                    if j == len(lines) - 1:
                        line_file.write('%d:%d:%d:%d:%s;\n'%(int(line[0]),int(line[1]),int(line[2]),int(line[3]),
                                                          lines_label[j].encode('utf-8')))
                    else:
                        line_file.write('%d:%d:%d:%d:%s;' % (int(line[0]), int(line[1]), int(line[2]), int(line[3]),
                                                             lines_label[j].encode('utf-8')))
                    counter += 1
            counter_image += 1
            if i % 1000 == 0: print 'proceeded %d' % i
    print 'image source number %d, contained image %d, gen line images : %d' \
          %(len(file_list), counter_image,counter)
def gen_line():
    # img_tages_dict = parse_xls(Config.img_tages_dict_path)
    img_tages_dict = parse_csv(Config.img_tages_dict_path)
    crop_line2png(img_tages_dict, Config.data_root, Config.save_root)

