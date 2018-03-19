import os
from captcha.image import ImageCaptcha
import pickle
import numpy as np
import string
import random
from multiprocessing import Pool

def id_generator(size=6, chars=string.ascii_letters + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


data_root = '/mnt/data/scenetext/captcha/python-gen'
label_root = '/mnt/data/LINEdevkit/tfRecord_captcha/txt'
fonts_root = '/media/home/netease/PyProjects/test/fonts'

fonts = os.listdir(fonts_root)
fonts = [os.path.join(fonts_root, font) for font in fonts]
image = ImageCaptcha(fonts=fonts)

def gen_captcha(params):
    cnt = params
    id = id_generator(random.randint(4, 6))
    img_path = os.path.join(data_root, '{0}_{1}.jpg'.format(id, cnt))
    image.write(id, img_path)
    if cnt % 10000 == 0:
        print 'processed %d'%cnt

def split(image_list, ratio=0.9):
    image_list = np.array(img_list)
    num_image = len(image_list)
    num_train = int(num_image * ratio)
    rng = range(0, num_image)
    random.shuffle(rng)
    train_list = image_list[rng[:num_train]]
    test_list = image_list[rng[num_train:]]

    return train_list, test_list

def write_txt(train_list, test_list, save_dir):
    f_train = open(os.path.join(save_dir, 'label_train.txt'), 'w')
    f_test = open(os.path.join(save_dir, 'label_test.txt'), 'w')
    for train in train_list:
        f_train.write('%s\n' %train)
    for test in test_list:
        f_test.write('%s\n' %test)
    f_train.close()
    f_test.close()

def write_map(label_root):
    # a raw map containing alphanum upper-insensitive"
    with open(os.path.join(label_root, 'alphanum_36.txt'), 'w') as f:
        f.write('%d %s\n' % (0, '<nul>'))
        for ind, char in enumerate(string.ascii_lowercase + string.digits):
            f.write('%d %s\n'%(ind+1, char.decode('utf-8')))


if __name__ == '__main__':
    # pool = Pool(8)
    # pool.map(gen_captcha, range(0, 3000000))
    img_list = os.listdir(data_root)
    img_list = [os.path.join(data_root, img)  for img in img_list]
    #split train/test
    train_list, test_list = split(img_list, 0.9)
    # write to txt
    write_txt(train_list, test_list, label_root)
    # save to map
    write_map(label_root)
