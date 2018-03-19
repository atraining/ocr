"""save (genometry_map, score_map, training_mask) to numpy array, thus can use GPU effectively"""
import sys
import cv2
import argparse
from utils import *

DEBUG = True
def parse_argments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_list_dir', type=str,
                        default='/mnt/data/scenetext/online/txts/version2/train.txt',
                        help='path to directory containning train.txt')
    parser.add_argument('--train_input_size', type=int,
                        default=640,
                        help='input size to fcn net')

    args = parser.parse_args(argv)
    return args

def main(args):
    image_fn_f = open(args.train_list_dir, 'r')
    image_fn_list = [line.strip() for line in image_fn_f.readlines()]
    print 'train images : {:2d}'.format(len(image_fn_list))
    image_fn_f.close()
    input_size = args.train_input_size
    for image_fn in image_fn_list:
        im = cv2.imread(image_fn)
        if im is None: return  None
        txt_fn = image_fn.rsplit('/', 2)[0] + '/txts/' + image_fn.rsplit('/', 2)[2].split('.')[0] + '.txt'
        text_polys, text_tags = load_annotation(txt_fn)
        text_polys, text_tags = check_and_valid_polys(text_polys, text_tags)

        #input image
        h, w = im.shape[:2]
        max_h_w = max(h, w)
        ratio = input_size * 1.0 / max_h_w
        im = cv2.resize(im, x_scale=ratio, y_scale=ratio)
        im_padding = np.zeros([input_size, input_size, 3], dtype=uint8)
        im_padding[:im.shape[0], :im.shape[1]] = im.copy()
        im = im_padding
        new_h, newe_w = im.shape[:2]
        angle_map, score_map, geometry_map, training_mask = gen_rbox(im, text_polys, text_tags)
        if DEBUG:
            cv2.imshow('', im)
            cv2.waitKey(0)
            cv2.imshow('', score_map)
            cv2.waitKey(0)
            cv2.imshow('', geometry_map)
            cv2.waitKey(0)
            cv2.imshow('', training_mask)
            cv2.waitKey(0)



def gen_rbox(im, text_polys, text_tages):
    for poly in text_polys:




if __name__ == '__main__':
    main(parse_argments(sys.argv[1:]))