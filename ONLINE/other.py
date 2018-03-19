import cv2
import numpy as np
from matplotlib import cm

def threshold(coords, min_, max_):
    return np.maximum(np.minimum(coords, max_), min_)


def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """
    #from index 0 and step = 2, so that boxes[:, 0 and 2]
    boxes[:, 0::2]=threshold(boxes[:, 0::2], 0, im_shape[1]-1)
    #from index 1 and step = 2, so that boxes[:, 1 and 3]
    boxes[:, 1::2]=threshold(boxes[:, 1::2], 0, im_shape[0]-1)
    return boxes

class Graph:
    def __init__(self, graph):
        self.graph=graph

    def sub_graphs_connected(self):
        sub_graphs=[]
        for index in xrange(self.graph.shape[0]):
            #if no precursors and have succession,then this is head
            if not self.graph[:, index].any() and self.graph[index, :].any():
                v=index
                sub_graphs.append([v])
                while self.graph[v, :].any():
                    #when have succession, find the first succession, note the dia of graph is False
                    v=np.where(self.graph[v, :])[0][0]
                    sub_graphs[-1].append(v)
        return sub_graphs

