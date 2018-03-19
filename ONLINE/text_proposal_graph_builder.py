from cfg import Config as cfg
import numpy as np
from other import Graph


class TextProposalGraphBuilder:
    """
        Build Text proposals into a graph.
    """
    def get_successions(self, index):
        """find the nearest proposals, and the search range is [1, MAX_HORIZONTAL_GAP=50 - 1]
        exclude the current x-accordinate, if find then return immediately
        note: the meet_v_iou of two proposals is compared not only overlapped area but also size """
        box=self.text_proposals[index]
        results=[]
        # MAX_HORIZONTAL_GAP=50
        for left in range(max(int(box[0]) + 1, int(box[2]) - cfg.INNER_HORIZONTAL_GAP), min(int(box[2]) + cfg.MAX_HORIZONTAL_GAP + 1, self.im_size[1])):
            adj_box_indices=self.boxes_table[left]
            for adj_box_index in adj_box_indices:
                if self.meet_v_iou(adj_box_index, index):
                    results.append(adj_box_index)
            if len(results) == 1:
                return results
            elif len(results) > 1:
                print 'box has multi succession', box[0], box[1], box[2], box[3]
        return results

    def meet_v_iou(self, index1, index2):
        """compare two text proposals not only overlapped area but also size_similarity"""
        def overlaps_v(index1, index2):
            h1=self.heights[index1]
            h2=self.heights[index2]
            y0=max(self.text_proposals[index2][1], self.text_proposals[index1][1])
            y1=min(self.text_proposals[index2][3], self.text_proposals[index1][3])
            return max(0, y1-y0+1)/min(h1, h2)

        def size_similarity(index1, index2):
            h1=self.heights[index1]
            h2=self.heights[index2]
            return min(h1, h2)/max(h1, h2)

        return overlaps_v(index1, index2)>=cfg.MIN_V_OVERLAPS and \
               size_similarity(index1, index2)>=cfg.MIN_SIZE_SIM

    def build_graph(self, char_boxes, char_labels, im_size):
        self.text_proposals = char_boxes
        self.char_labels = char_labels
        self.im_size=im_size
        self.heights=char_boxes[:, 3] - char_boxes[:, 1] + 1

        # for x-cords
        boxes_table=[[] for _ in range(self.im_size[1])]
        for index, box in enumerate(char_boxes):
            # in order to have no -1, is have, point to end pixel
            anchor = min(max(0, int(box[0])), im_size[1] - 1)
            boxes_table[anchor].append(index)
        self.boxes_table=boxes_table

        graph=np.zeros((char_boxes.shape[0], char_boxes.shape[0]), np.bool)

        for index, box in enumerate(char_boxes):
            # expect only one succession
            succession_index=self.get_successions(index)
            if len(succession_index)==0:
                continue
            graph[index, succession_index]=True
        return Graph(graph)
