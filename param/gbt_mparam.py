import numpy as np
from param.train_param import TrainParam


class GBTreeModelParam:
    def __init__(self):
        self.num_trees = 0
        self.size_leaf_vector = 0
        self.num_roots = 1
        self.num_feature = 0
        self.num_pbuffer = 0
        self.num_output_group = 1

        self.reserved = np.zeros(31)

    def set_param(self, name, val):
        if name == 'bst:num_pbuffer':
            self.num_pbuffer = val
        elif name == 'bst:num_output_group':
            self.num_output_group = val
        elif name == 'bst:num_roots':
            self.num_roots = val
        elif name == 'bst:num_feature':
            self.num_feature = val
        elif name == 'bst:size_leaf_vector':
            self.size_leaf_vector = val

    def pred_buffer_size(self):
        """ size of needed preduction buffer """
        return self.num_output_group * self.num_pbuffer * \
               (self.size_leaf_vector + 1)

    def buffer_offset(self, buffer_index, bst_group):
        """ get the buffer offset given a buffer index and group id """
        if buffer_index < 0:
            return -1
        return (buffer_index + self.num_pbuffer * bst_group) * \
               (self.size_leaf_vector + 1)
