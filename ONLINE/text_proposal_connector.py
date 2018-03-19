import numpy as np
from math import pi
from other import clip_boxes
from cfg import Config as cfg
from text_proposal_graph_builder import TextProposalGraphBuilder

class TextProposalConnector:
    """
        Connect text proposals into text lines
    """
    def __init__(self):
        self.graph_builder=TextProposalGraphBuilder()

    def group_text_proposals(self, char_boxes, char_labels, im_size):
        graph=self.graph_builder.build_graph(char_boxes, char_labels, im_size)
        return graph.sub_graphs_connected()

    def fit_y(self, X, Y, x1, x2):
        len(X)!=0
        # if X only include one point, the function will get line y=Y[0]
        if np.sum(X==X[0])==len(X):
            return Y[0], Y[0]
        p=np.poly1d(np.polyfit(X, Y, 1))
        return p(x1), p(x2)

    def get_text_lines(self, char_boxes, char_labels, im_size):
        # tp=text proposal
        # the subgraph of graph, type is 'list of list '
        tp_groups = self.group_text_proposals(char_boxes, char_labels, im_size)
        text_lines = np.zeros((len(tp_groups), 4), np.float32)

        text_lines_label = [''] * len(text_lines)
        for index, tp_indices in enumerate(tp_groups):
            text_line_boxes = char_boxes[list(tp_indices)]

            x0=np.min(text_line_boxes[:, 0])
            x1=np.max(text_line_boxes[:, 2])

            offset=(text_line_boxes[0, 2]-text_line_boxes[0, 0])*0.5

            #rectify the text proposals to a line which uses ployfit's method, then enlarge
            lt_y, rt_y=self.fit_y(text_line_boxes[:, 0], text_line_boxes[:, 1], x0+offset, x1-offset)
            lb_y, rb_y=self.fit_y(text_line_boxes[:, 0], text_line_boxes[:, 3], x0+offset, x1-offset)

            # the score of a text line is the average score of the scores
            # of all text proposals contained in the text line

            y0 = min(lt_y, rt_y)
            y1 = max(lb_y, rb_y)
            if np.arctan((text_line_boxes[-1, 1] - y0)/(text_line_boxes[-1, 2] - x0)) > pi*cfg.MAX_ANGLE/180:
                print 'current angle is', np.arctan((text_line_boxes[-1, 3] - y0) / (text_line_boxes[-1, 2] - x0)) / pi * 180
                x0, y0, x1, y1 = 0, 0, 0, 0
            text_lines[index, 0]=x0
            text_lines[index, 1]=y0
            text_lines[index, 2]=x1
            text_lines[index, 3]=y1

            for tp_indice in tp_indices:
                text_lines_label[index] += char_labels[tp_indice]

        text_lines=clip_boxes(text_lines, im_size)

        return text_lines, text_lines_label
