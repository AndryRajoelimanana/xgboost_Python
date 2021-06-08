from param.model_param import LearnerModelParam
from param.gbtree_params import GBTreeModelParam


class GBTreeModel:
    def __init__(self, learner_model=None):
        self.learner_model_param = LearnerModelParam(learner_model, 0.5)
        self.param = GBTreeModelParam()
        self.trees_to_update = []
        self.trees = []
        self.tree_info = []

    def configure(self, cfg):
        if len(self.trees) == 0:
            self.param.update_allow_unknown(cfg)

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


