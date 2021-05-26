from param.model_param import LearnerModelParam, XGBoostParameter

import numpy as np


class GBTreeModelParam(XGBoostParameter):
    def __init__(self):

        self.num_trees = 0
        self.size_leaf_vector = 0


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

