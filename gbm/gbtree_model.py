from tree.gbtree import GBTreeModelParam
from utils.util import LearnerModelParam


class GBTreeModel:
    def __init__(self, learner_model):
        # TODO
        self.learner_model_param = LearnerModelParam()
        # end to do
        self.param = GBTreeModelParam()
        self.trees_to_update = []
        self.trees = []
        self.tree_info = []

    def configure(self, cfg):
        if len(self.trees) == 0:
            for k, v in cfg:
                setattr(self.param, k, v)

    def init_trees_to_update(self):
        if len(self.trees_to_update):
            for tree in self.trees:
                self.trees_to_update.append(tree)
            self.trees = []
            self.param.num_trees = 0
            self.tree_info = []

