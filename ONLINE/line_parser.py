#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
from ONLINE.cfg import Config
from text_proposal_connector import TextProposalConnector
import matplotlib.pyplot as plt

class LineParser:
    """
        parse line in image
    """
    def __init__(self, tags='', im_shape=None):
        self.tags = tags
        self.boxes_connector = TextProposalConnector()
        self.im_shape = im_shape

    def get_all_boxes(self, tags='', decode_utf8=True):
        """given a label tages, parse all chars contained in it"""
        if decode_utf8:
            tags = tags.decode('utf-8')
        single_char = [char for char in tags.split(';') if char[-1] != '#']

        char_boxes = np.zeros((len(single_char), 4), dtype=np.float32)
        char_labels = []
        for ind, char in enumerate(single_char):
            try:
                char_boxes[ind] = np.array([float(char.split(':')[0]),
                                            float(char.split(':')[1]),
                                            float(char.split(':')[2]),
                                            float(char.split(':')[3])], dtype = np.float32)
                char_labels.append(char.split(':')[-1])
            except:
                char_labels.append('')

        return char_boxes, char_labels

    def filter_boxes(self, boxes, boxes_label):
        heights = boxes[:, 3] - boxes[:, 1] + 1
        widths = boxes[:, 2] - boxes[:, 0] + 1
        keep_inds = np.where((widths > Config.BOXES_WIDTH ) &
                             (heights > Config.BOXES_HEIGHT))[0]
        boxes = boxes[keep_inds]
        boxes_label = [boxes_label[ind] for ind in keep_inds]

        return boxes, boxes_label


    def filter_lines(self, lines, lines_label):
        heights = lines[:, 3] - lines[:, 1] + 1
        widths = lines[:, 2] - lines[:, 0] + 1
        keep_inds = np.where((widths / heights > Config.MIN_RATIO) &
                        (widths > (Config.BOXES_WIDTH * Config.MIN_NUM_BOXES)))[0]
        lines = lines[keep_inds]
        lines_label = [lines_label[ind] for ind in keep_inds]
        return lines, lines_label

    def gen_line(self):
        char_boxes, char_labels = self.get_all_boxes(self.tags, decode_utf8=Config.decode_utf8)
        char_boxes, char_labels = self.filter_boxes(char_boxes, char_labels)
        lines, lines_label = self.boxes_connector.get_text_lines(char_boxes, char_labels, self.im_shape[:2])
        lines, lines_label = self.filter_lines(lines, lines_label)

        return lines, lines_label

    def display_boxes(self, im, boxes, labels):
        from matplotlib.font_manager import FontProperties
        ChineseFont = FontProperties(fname='/media/home/netease/PyProjects/test/fonts/SIMYOU.TTF')
        im = im[:, :, (2, 1, 0)]
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(im, aspect='equal')
        for i, box in enumerate(boxes):
            bbox = boxes[i, :4]
            ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='red', linewidth=3.5)
            )
            ax.text(bbox[0], bbox[1] - 2,
                    u'{:s}'.format(labels[i]),
                    bbox=dict(alpha=0.5),
                    fontsize=14, color='white', fontproperties=ChineseFont)
        plt.axis('off')
        plt.tight_layout()
        plt.draw()
        plt.show()

