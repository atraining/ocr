#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""some utils about line annotation re-check"""


import cv2, os
import shutil
class Config:
    def __init__(self, phase=None):
        self.error_list_path = '/media/data/LINEdevkit/error/log-{0}.txt'.format(phase)
        self.file_list = '/media/data/LINEdevkit/txt/label_{}.txt'.format(phase)
        self.except_file = '/media/data/LINEdevkit/lmdb/except_{0}.txt'.format(phase)
        self.save_root_dir = '/media/data/LINEdevkit/error/images'

        self.DEBUG = False

def get_file_list(file_list_path, except_list_path):
    with open(file_list_path, 'r') as file_list:
        item_list = [line.strip().decode('utf-8') for line in file_list.readlines()]
        # except_list is 1-index
        ret_item_list = []
        # print 'all item list num: %d'%len(item_list)
        with open(except_list_path, 'r') as except_file_list:
            except_ind_list = [int(line.strip().decode('utf-8')) - 1 for line in except_file_list.readlines()]
            for ind in xrange(len(item_list)):
                if ind not in except_ind_list:
                    ret_item_list.append(item_list[ind])
        # print 'after remove except item list num: %d'%len(ret_item_list)
        return ret_item_list

def parse_error_txt(error_list_path):
    with open(error_list_path, 'r') as error_f:
        error_list = [line.strip().decode('utf-8') for line in error_f.readlines()]
        print 'image that needed re-check num is %d'%len(list(set(error_list)))
        return list(set(error_list))

def analysis_annos():
    error_file_path = '/media/data/LINEdevkit/error/error_list.txt'
    with open(error_file_path, 'w') as error_file:
        for phase in ['train', 'val']:
            cfg = Config(phase)
            file_item_list = get_file_list(cfg.file_list, cfg.except_file)
            error_list = parse_error_txt(cfg.error_list_path)
            for error in error_list:
                error = error.split(' ')
                file_ind = int(error[0])
                pred_str = error[1]
                gt_str = error[2]
                file_path = file_item_list[file_ind - 1].split(' ')[0]
                if cfg.DEBUG or gt_str != file_item_list[file_ind - 1].split(' ')[1]:
                    im = cv2.imread(file_path)
                    print 'file ind is %d'%file_ind
                    print 'pre str: %s, gt str: %s, oringnal str: %s'%(pred_str, gt_str, file_item_list[file_ind - 1].split(' ')[1])
                    cv2.imshow('', im)
                    cv2.waitKey(0)
                shutil.copy(file_path, os.path.join(cfg.save_root_dir, '{}_{}.png'.format(phase, file_ind)))
                error_file.write('%s_%d.png %s\n'%(phase, file_ind, gt_str.encode('utf-8')))
