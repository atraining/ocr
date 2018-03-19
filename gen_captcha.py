# -*- encoding: utf-8 -*-
from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib import text as mtext
from matplotlib import colors as mcolors
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

import numpy as np
import math
from color_util import *

import pickle
import os
import cv2

fonts_root = "/mnt/data/fonts/ch_font/huawen"
fs = []
for (dirpath, dirnames, filenames) in os.walk(fonts_root):
    fs.extend(map(lambda x: os.path.join(dirpath, x.decode('utf-8')), filenames))
from matplotlib.font_manager import FontProperties

class CurvedText(mtext.Text):
    """
    A text object that follows an arbitrary curve.
    """
    def __init__(self, x, y, text, axes, image, font, **kwargs):
        super(CurvedText, self).__init__(x[0],y[0],' ', axes, **kwargs)

        axes.add_artist(self)

        ##saving the curve:
        self.__x = x
        self.__y = y
        self.__zorder = self.get_zorder()

        ##creating the text objects
        self.__Characters = []
        self.image = image
        self.pos_list = []
        self.char_list = []

        for c in text:
            if c == ' ':
                ##make this an invisible 'a':
                t = mtext.Text(0,0,'a')
                t.set_alpha(0.0)
            else:
                # t = mtext.Text(0,0,c, **kwargs)
                if c == u'领' or np.random.randint(0, 9) > 5:
                    t = mtext.Text(0,0,u'{:s}'.format(c),color=colors['darkred'], fontproperties=font, weight='heavy',size=24)
                else:
                    t = mtext.Text(0, 0, u'{:s}'.format(c), color=colors['darkred'], fontproperties=font)
            #resetting unnecessary arguments
            t.set_ha('center')
            t.set_rotation(0)
            t.set_zorder(self.__zorder +1)

            self.__Characters.append((c,t))
            axes.add_artist(t)

    ##overloading some member functions, to assure correct functionality
    ##on update
    def set_zorder(self, zorder):
        super(CurvedText, self).set_zorder(zorder)
        self.__zorder = self.get_zorder()
        for c,t in self.__Characters:
            t.set_zorder(self.__zorder+1)

    def draw(self, renderer, *args, **kwargs):
        """
        Overload of the Text.draw() function. Do not do
        do any drawing, but update the positions and rotation
        angles of self.__Characters.
        """
        self.update_positions(renderer)

    def update_positions(self,renderer):
        """
        Update positions and rotations of the individual text elements.
        """

        #preparations

        ##determining the aspect ratio:
        ##from https://stackoverflow.com/a/42014041/2454357

        ##data limits
        xlim = self.axes.get_xlim()
        ylim = self.axes.get_ylim()
        ## Axis size on figure
        figW, figH = self.axes.get_figure().get_size_inches()
        ## Ratio of display units
        _, _, w, h = self.axes.get_position().bounds
        ##final aspect ratio
        aspect = ((figW * w)/(figH * h))*(ylim[1]-ylim[0])/(xlim[1]-xlim[0])

        #points of the curve in figure coordinates:
        x_fig,y_fig = (
            np.array(l) for l in zip(*self.axes.transData.transform([
            (i,j) for i,j in zip(self.__x,self.__y)
            ]))
        )

        #point distances in figure coordinates
        x_fig_dist = (x_fig[1:]-x_fig[:-1])
        y_fig_dist = (y_fig[1:]-y_fig[:-1])
        r_fig_dist = np.sqrt(x_fig_dist**2+y_fig_dist**2)

        #arc length in figure coordinates
        l_fig = np.insert(np.cumsum(r_fig_dist),0,0)

        #angles in figure coordinates
        rads = np.arctan2((y_fig[1:] - y_fig[:-1]),(x_fig[1:] - x_fig[:-1]))
        degs = np.rad2deg(rads)


        rel_pos = 10
        ind = 0
        for c,t in self.__Characters:
            #finding the width of c:
            t.set_rotation(0)
            t.set_va('center')
            bbox1  = t.get_window_extent(renderer=renderer)
            w = bbox1.width * 1.2
            h = bbox1.height * 1.2

            #ignore all letters that don't fit:
            if rel_pos+w/2 > l_fig[-1]:
                t.set_alpha(0.0)
                rel_pos += w
                continue

            elif c != ' ':
                t.set_alpha(1.0)

            #finding the two data points between which the horizontal
            #center point of the character will be situated
            #left and right indices:
            il = np.where(rel_pos+w/2 >= l_fig)[0][-1]
            ir = np.where(rel_pos+w/2 <= l_fig)[0][0]

            #if we exactly hit a data point:
            if ir == il:
                ir += 1

            #how much of the letter width was needed to find il:
            used = l_fig[il]-rel_pos
            rel_pos = l_fig[il]

            #relative distance between il and ir where the center
            #of the character will be
            fraction = (w/2-used)/r_fig_dist[il]

            ##setting the character position in data coordinates:
            ##interpolate between the two points:
            x = self.__x[il]+fraction*(self.__x[ir]-self.__x[il])
            y = self.__y[il]+fraction*(self.__y[ir]-self.__y[il])

            #getting the offset when setting correct vertical alignment
            #in data coordinates
            t.set_va(self.get_va())
            bbox2  = t.get_window_extent(renderer=renderer)
            width = bbox2.width * 1.2
            height = bbox2.height * 1.2

            bbox1d = self.axes.transData.inverted().transform(bbox1)
            bbox2d = self.axes.transData.inverted().transform(bbox2)
            dr = np.array(bbox2d[0]-bbox1d[0])

            #the rotation/stretch matrix
            if ind > 1:
                rad = rads[il] + np.pi  * np.random.random_sample()
            else:
                rad = rads[il]
            # rad = 2 * np.pi * np.random.random_sample()
            rot_mat = np.array([
                [math.cos(rad), math.sin(rad)*aspect],
                [-math.sin(rad)/aspect, math.cos(rad)]
            ])

            ##computing the offset vector of the rotated character
            drp = np.dot(dr,rot_mat)

            #setting final position and rotation:
            t.set_position(np.array([x,y])+drp)
            start = np.array([x,y])+drp
            # print max(0, int(start[1])), int(start[1] + height), max(0, int(start[0])), int(start[0] + width)
            text_region = self.image[max(0, int(start[1])):int(start[1] + height),
                          max(0, int(start[0])):int(start[0] + width), :3]
            mean_b, mean_g, mean_r = np.mean(text_region, axis=(0, 1))
            # mean_b, mean_g, mean_r = 0, 0, 0
            color = gen_contrast_color((mean_r, mean_g, mean_b))

            self.pos_list.append(np.array([x + drp[0],y + drp[1], width, height]))
            self.char_list.append(c)
            color = [c / 255.0 for c in color]

            # t.set_rotation(degs[il])
            t.set_rotation(np.rad2deg(rad))
            # t.set_backgroundcolor(color)
            # t.set_color(colors['darkblue'])
            t.set_color(color)

            t.set_va('center')
            t.set_ha('center')

            #updating rel_pos to right edge of character
            rel_pos += w-used

            ind += 1

def get_curves():
    N = 150
    curves = [
        [
            np.sin(np.linspace(0, 2 * np.pi, N)) + 0.2,
            -np.cos(np.linspace(0, 2 * np.pi, N)) + 1 + 0.2,
        ],
        [
            np.sin(np.linspace(0, 2 * np.pi, N)) + 0.2,
            -np.cos(np.linspace(0, 2 * np.pi, N)) + 1 + 0.6,
        ],
        [
            -np.cos(np.linspace(0, 2 * np.pi, N)) + 1 + 0.3,
            np.sin(np.linspace(0, 2 * np.pi, N)),
        ],
        [
            np.cos(np.linspace(0, 2 * np.pi, N)) - 0.3,
            -np.sin(np.linspace(0, 2 * np.pi, N)) + 1,
        ],
    ]
    return curves
def get_texts(poetry_name):
    text_dir = '/mnt/data/scenetext/poetry/texts'
    text = os.path.join(text_dir, random.choice(os.listdir(text_dir)))
    texts = []
    with open(text, 'r') as text_f:
        lines = text_f.readlines()
        line = lines[0].decode('utf-8').strip()
        line = line.split(u'。')
        for line_split in line:
            try:
                texts.append([line_split.rsplit(u'，', 1)[0], line_split.rsplit(u'，', 1)[1]])
            except:
                print line_split
    return texts
def dict2txt(pos_list,char_list, fname):
    """write pos_char_pos to txt"""
    assert len(pos_list) == len(char_list)
    with open(fname, 'w') as f:
        for i in xrange(0, len(pos_list)):
            f.write('%d,%d,%d,%d,%s\n'%(int(pos_list[i][0]), int(pos_list[i][1]), int(pos_list[i][2])
                                      , int(pos_list[i][3]), char_list[i].encode('utf-8')))
if __name__ == '__main__':
    curves = get_curves()
    # ax = Axes.reshape(-1)[3]
    image_dir = '/mnt/data/scenetext/poetry/images'
    for i in xrange(2):
        font = np.random.choice(fs)
        ChineseFont = FontProperties(fname=font)
        poetry_name = random.choice([name for name in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, name))])
        image_name = os.path.join(image_dir, poetry_name, random.choice(os.listdir(os.path.join(image_dir, poetry_name))))
        texts = get_texts(poetry_name)
        # random.shuffle(texts)
        # Figure, Axes = plt.subplots(1, 1, figsize=(6.4, 3.2))
        Figure, Axes = plt.subplots(1, dpi=100)
        ax = Axes
        ax.axis('off')
        im = plt.imread(image_name)
        ax.imshow(im)
        text_list = []
        for curve, text in zip(curves[:2], random.choice(texts)):
            #adding the text
            xlim = ax.get_xlim()
            w = xlim[1] - xlim[0]
            ylim = ax.get_ylim()
            h = ylim[1] - ylim[0]
            text = CurvedText(
                x = curve[0] * np.abs(w),
                y = curve[1] * np.abs(h),
                text=text,#'this this is a very, very long text',
                va = 'bottom',
                axes = ax, ##calls ax.add_artist in __init
                image = im,
                font=ChineseFont,# __
            )
            plt.plot(curve[0] * np.abs(w), curve[1] * np.abs(h))
            text_list.append(text)
        fname = './data/texts/{}.txt'.format(i)
        plt.savefig('./data/images/{}.png'.format(i))
        char_list = []
        pos_list = []
        for text in text_list:
            pos_list.extend(text.pos_list)
            char_list.extend(text.char_list)
        dict2txt(pos_list, char_list, fname)
        plt.axis('off')
        plt.show()

