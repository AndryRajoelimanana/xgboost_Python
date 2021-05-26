from param.model_param import LearnerModelParam
import numpy as np


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


class GBTreeModel:
    def __init__(self, learner_model):
        self.learner_model_param = LearnerModelParam(learner_model, 0.5)
        self.param = GBTreeModelParam()
        self.trees_to_update = []
        self.trees = []
        self.tree_info = []

    def configure(self, cfg):
        if len(self.trees) == 0:
            for k, v in cfg.items():
                if hasattr(self.param, k):
                    setattr(self.param, k, v)

    def init_trees_to_update(self):
        if len(self.trees_to_update) == 0:
            for tree in self.trees:
                self.trees_to_update.append(tree)
            self.trees = []
            self.param.num_trees = 0
            self.tree_info = []

    def commit_model(self, new_trees, bst_group):
        for tree in new_trees:
            self.trees.append(tree)
            self.tree_info.append(bst_group)
        self.param.num_trees += len(new_trees)

