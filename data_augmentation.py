#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""some data augmentation functions"""

from common_utilities import  *

class Config:
    DATA_ROOT = '/mnt/data/VOCdevkit/VOC2007/'
    DATA_ROOT = '/mnt/data/ICDARdevkit/ICDAR2015/'
    # DATA_ROOT = '/mnt/data/ICDARdevkit/ICDAR2015/augmentation4/crop/ast/'
    DATA_ROOT = '/mnt/data/ICDARdevkit/ICDAR2013/'
    DATA_ROOT = '/mnt/data/ONLINEdevkit/8.15-6k/'

    SAVE_ROOT = os.path.join(DATA_ROOT, 'augmentation')
    # SAVE_ROOT = os.path.join(DATA_ROOT, 'debug')
    # SAVE_ROOT = os.path.join(DATA_ROOT, 'augmentation-std')
    # SAVE_ROOT = os.path.join(DATA_ROOT, 'aug-non-reorder')
    # SAVE_ROOT = os.path.join(DATA_ROOT, 'augmentation4')

def rotate_image(src, angle, scale=1):
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

def roate_xml(src, crds, angle, scale=1.):
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

def reorder_vertexes(vertexes):
    """given four vertexes of almost rectangle, reorder vertexes
    (left-top, right-top, right-down, left-down)
    vertexes: ndarray (4, 2)"""
    if vertexes.shape == (4, 1, 2):
        vertexes = vertexes.reshape((4, 2))
    sorted_ind = np.argsort(vertexes[:, 0], axis=0)

    if vertexes[sorted_ind[1], 0] == vertexes[sorted_ind[2], 0]:
        if vertexes[sorted_ind[1], 1] > vertexes[sorted_ind[2], 1]:
            sorted_ind[1], sorted_ind[2] = sorted_ind[2], sorted_ind[1]

    if vertexes[sorted_ind[0], 1] > vertexes[sorted_ind[1], 1]:
        sorted_ind[0], sorted_ind[1] = sorted_ind[1], sorted_ind[0]

    if vertexes[sorted_ind[2], 1] > vertexes[sorted_ind[3], 1]:
        sorted_ind[2], sorted_ind[3] = sorted_ind[3], sorted_ind[2]

    return vertexes[[sorted_ind[0], sorted_ind[2], sorted_ind[3], sorted_ind[1]], :]


def is_ordered(vertexes):
    """given four vertexes of almost , jude if it is ordered"""
    if max(vertexes[0, 0], vertexes[3, 0]) > min(vertexes[1, 0], vertexes[2, 0]):
        return False
    if vertexes[0, 1] > vertexes[3, 1] or vertexes[1, 1] > vertexes[2, 1]:
        return False
    return True


def rotate(major_list, angle_list):
    # rotate image
    n_image = len(major_list)
    print 'processing images number :%d' % n_image
    image_sets, jpeg_images, annotations, gts= gen_VOC_folders(Config.SAVE_ROOT)
    trainfile, testfile, mapfile = gen_txt_files(image_sets)
    counter = 0
    for major in major_list:
        img_path = os.path.join(Config.DATA_ROOT, 'JPEGImages', '{}.jpg'.format(major))
        xml_path = os.path.join(Config.DATA_ROOT, 'Annotations', '{}.xml'.format(major))

        img = cv2.imread(img_path)
        cnt = 0
        for angle in angle_list:
            tree = ET.parse(xml_path)
            new_img = rotate_image(img, angle, scale=1)
            obj_size = tree.find('size')
            obj_size.find('width').text = str(new_img.shape[1])
            obj_size.find('height').text = str(new_img.shape[0])
            # cv2.imshow('', new_img)
            # cv2.waitKey(0)
            gt_txt = os.path.join(gts, '{}_{}.txt'.format(major, angle))
            f_gt = codecs.open(gt_txt, "w", encoding="utf-8-sig")
            cv2.imwrite(os.path.join(Config.SAVE_ROOT, 'JPEGImages', '{}_{}.jpg'.format(major, angle)), new_img)
            # rotate xml
            objs = tree.findall('object')
            for ind, obj in enumerate(objs):
                label = obj.find('txt').text
                if label is None: label = ''
                box = obj.find('quad')
                x0 = float(box.find('x0').text)
                y0 = float(box.find('y0').text)
                x1 = float(box.find('x1').text)
                y1 = float(box.find('y1').text)
                x2 = float(box.find('x2').text)
                y2 = float(box.find('y2').text)
                x3 = float(box.find('x3').text)
                y3 = float(box.find('y3').text)
                crds = [x0, y0, x1, y1, x2, y2, x3, y3]
                quad = roate_xml(img, crds, angle, scale=1)
                # clock-wise
                quad = quad.reshape((4, 2))
                quad = check_and_validate_quad(quad, (new_img.shape[0], new_img.shape[1]))
                if quad is None: continue
                assert polygon_area(quad.reshape(4, 2)) < 0
                box.find('x0').text = str(int(quad[0, 0]))
                box.find('y0').text = str(int(quad[0, 1]))
                box.find('x1').text = str(int(quad[1, 0]))
                box.find('y1').text = str(int(quad[1, 1]))
                box.find('x2').text = str(int(quad[2, 0]))
                box.find('y2').text = str(int(quad[2, 1]))
                box.find('x3').text = str(int(quad[3, 0]))
                box.find('y3').text = str(int(quad[3, 1]))
                f_gt.write( str(int(quad[0, 0])) + ',' + str(int(quad[0, 1])) + ',' +
                            str(int(quad[1, 0])) + ',' + str(int(quad[1, 1])) + ',' +
                            str(int(quad[2, 0])) + ',' + str(int(quad[2, 1])) + ',' +
                            str(int(quad[3, 0])) + ',' + str(int(quad[3, 1])) + ',' + label + '\n')
            f_gt.close()
            tree.write(os.path.join(Config.SAVE_ROOT, 'Annotations', '{}_{}.xml'.format(major, angle)))
            trainfile.write('{}_{}\n'.format(major, angle))
            cnt += 1
        if cnt != len(angle_list):
            print 'major:%s, wrote:%d\n'%(major, cnt)
        counter += 1
        if counter % 500 == 0:
            print 'processed %d'%counter
def test_rotate(major_list, angle_list):

    for major in major_list:
        for angle in angle_list:

            img_path = os.path.join(Config.SAVE_ROOT, 'JPEGImages', '{}_{}.jpg'.format(major, angle))
            img = cv2.imread(img_path)
            # rotate xml
            xml_path = os.path.join(Config.SAVE_ROOT, 'Annotations', '{}_{}.xml'.format(major, angle))
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for box in root.iter('quad'):
                x0 = float(box.find('x0').text)
                y0 = float(box.find('y0').text)
                x1 = float(box.find('x1').text)
                y1 = float(box.find('y1').text)
                x2 = float(box.find('x2').text)
                y2 = float(box.find('y2').text)
                x3 = float(box.find('x3').text)
                y3 = float(box.find('y3').text)
                pts = np.array([x0, y0, x1, y1, x2, y2, x3, y3], dtype=np.int32)
                pts = pts.reshape((-1, 1, 2))
                # print pts
                cv2.polylines(img, [pts], True, (0, 0, 255), 1)
            cv2.imshow('', img)
            cv2.waitKey(0)

if __name__ == '__main__':
    img_list = os.listdir(os.path.join(Config.DATA_ROOT, 'JPEGImages'))
    major_list = map(lambda img : img.split('.')[0], img_list)
    # data augmentation only for train data
    # f_train = open(os.path.join(Config.DATA_ROOT, 'ImageSets/Main/train.txt'))
    # major_list = [line.strip() for line in f_train.readlines()]
    # major_list[:] = [ind for ind in major_list if ind not in invalid_list]
    # major_list = ['000000','000001', '000002', '000003', '000004']
    angle_list = [-90, -75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 90]
    rotate(major_list, angle_list)
    # test_rotate(major_list, angle_list)
