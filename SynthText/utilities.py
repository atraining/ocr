#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""some utilities of SynthText dataset"""

import random, shutil
from common_utilities import *
import scipy.io


def get_image_words(txt):
    """get all the words containing in the image for txt which is grounded by the same font,color,distortiono etc
    txt containing space and \n"""
    words_list = []
    cnt = 0
    for t in txt:
        t = t.split()
        for tt in t:
            cnt += 1
            words_list.append(tt)
    return words_list

def get_image_bbs(wordBB, txt):
    """get all the bbs in a image, wordBB's shape is (2, 4, n_words)"""
    words_list = get_image_words(txt)
    if len(wordBB.shape) < 3: wordBB = wordBB.reshape((wordBB.shape[0], wordBB.shape[1], 1))
    _,_,n_words = wordBB.shape
    gts = []
    for ind in range(n_words):
        x = wordBB[0, :, ind]
        y = wordBB[1, :, ind]
        #find the Minimum Enclosing Rectangle
        label = words_list[ind]
        gt = BoundingBox(min(int(float(x[0])), int(float(x[3]))),
                         min(int(float(y[0])), int(float(y[1]))),
                         max(int(float(x[1])), int(float(x[2]))),
                         max(int(float(y[2])), int(float(y[3]))),
                         label,'Rectangle'
                         )
        gts.append(gt)
        # gt = Polylines(int(float(x[0])), int(float(y[0])),
        #                  int(float(x[1])), int(float(y[1])),
        #                  int(float(x[2])), int(float(y[2])),
        #                  int(float(x[3])), int(float(y[3])),
        #                  '', 'PolyLines')
        #
        # gts.append(gt)
    return gts

def SynthText2VOC(mat, ast_root):
    """conver SynthText data to VOC data Format"""
    image_sets, jpeg_images, annotations = gen_VOC_folders(ast_root)
    trainfile, testfile, mapfile = gen_txt_files(image_sets)
    cnt = 0
    ind = 0

    wordBB = mat['wordBB'] # shape is (1, 858750)
    imnames = mat['imnames']
    txt = mat['txt']
    _, n_image = wordBB.shape
    print 'processing images number :%d' % n_image

    t1 = show_time()

    for ind in range(n_image):
        gts = get_image_bbs(wordBB[0, ind], txt[0, ind])
        im = cv2.imread(imnames[0, ind][0])
        img_name = imnames[0, ind][0]
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

def gen_data():
    mat_file = '/mnt/data/scenetext/dataset/vgg/SynthText/SynthText/gt.mat'
    mat = scipy.io.loadmat(mat_file)
    data_root = '/mnt/data/scenetext/dataset/vgg/SynthText/SynthText'
    os.chdir(data_root)
    # scipy.io.savemat('small.mat', mdict={'charBB': mat['charBB'][0, :3],
    #                                      'imnames': mat['imnames'][0, :3],
    #                                      'txt': mat['txt'][0, :3],
    #                                      'wordBB': mat['wordBB'][0, :3]})
    ast_root = '/home/netease/data/scenetext/dataset/vgg/SynthText'
    SynthText2VOC(mat, ast_root)
    # get relative path
    rel_fs = []
    for (dirpath, dirnames, filenames) in os.walk(data_root):
        rel_fs.extend(map(lambda x: os.path.join(dirpath, x), filenames))
