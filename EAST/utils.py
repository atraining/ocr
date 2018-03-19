import numpy as np

def load_annotation(txt_fn):
    text_polys = []
    text_tags = []
    with open(txt_fn, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            splits = line.split(',')
            text_polys.append(np.array(splits[:8]).reshape((4, 2)))
            if len(splits) > 8:
                text_tags.append(1) if splits[8] == '###' else text_tags.append(0)

        return  np.array(text_polys), np.array(text_tags)


def check_and_valid_polys(text_ploys):
    for poly in text_ploys:
        if poly_area(poly) > 0:
            poly = poly[(0, 3, 2, 1)]
            if poly_area(poly) > 0:
                print 'after reverse, point direction is also anti-clockwise'
        if poly_area(poly) < 1:
            print poly
            print 'invalid ploy, area less than 1'

def poly_area(poly):
    """Shoelace formula
    A_poly = 1/2* abs(sum_{i=1}^n (x_i + x_{i+1}) * (y_i - y{i+1}))
    if A_poly > 0 : anti-clockwise"""

    return ((poly[0][0] + poly[1][0]) * (poly[0][1] - poly[1][1]) + \
           (poly[1][0] + poly[2][0]) * (poly[1][1] - poly[2][1]) + \
           (poly[2][0] + poly[3][0]) * (poly[2][1] - poly[3][1]) + \
           (poly[3][0] + poly[0][0]) * (poly[3][1] - poly[0][1])) * 1.0 /2

