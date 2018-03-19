# -*- encoding: utf-8 -*-
import random
import colorsys


def complement_color(color):
    color = [x / 255.0 for x in color]
    # print color
    h, l, s = colorsys.rgb_to_hls(*color)
    # print 'HLS', h, l, s
    h = h - 0.5 if h > 0.5 else h + 0.5
    l = 1 - l
    # s = 1 - s
    # print 'HLS', h, l, s
    color = colorsys.hls_to_rgb(h, l, s)
    # print 'color: ', color
    # rand_color_func = lambda x: max(0, min(1, random.gauss(x, 0.1)))
    # color = [rand_color_func(x) for x in color]
    # print 'color with noise: ', color
    return tuple([int(x * 255) for x in color])


def choose_black_or_white(color):
    """return white or black according to the perceptive luminance of the given color for good contrast"""
    r, g, b = color
    a = 1 - (0.299 * r + 0.587 * g + 0.114 * b) / 255
    d = 0 if a < 0.5 else 1
    d += (1 - 2 * d) * max(0, min(1, random.gauss(0, 0.1)))
    # d += max(0, min(1, random.gauss(0, 0.1)))
    return tuple([int(d * 255) for _ in range(3)])


def gen_color_pair():
    color1 = (random.randint(0, 31) * 8, random.randint(0, 31) * 8, random.randint(0, 31) * 8)
    # 80% of images will have white/black color for background/foreground
    if random.uniform(0, 1) < 0.8:
        color2 = choose_black_or_white(color1)
    else:
        color2 = complement_color(color1)
    # 50% probability to swap the background and foreground color
    if random.uniform(0, 1) < 0.5:
        return color1, color2
    else:
        return color2, color1


def gen_contrast_color(color1):
    # 80% of images will have white/black color for background/foreground
    if random.uniform(0, 1) < 1:
        color2 = choose_black_or_white(color1)
    else:
        color2 = complement_color(color1)
    return color2
