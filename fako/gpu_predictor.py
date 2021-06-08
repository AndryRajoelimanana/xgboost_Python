from predictor.predictors import *
from fako.cpu_predictor import *


class GPUPredictor(Predictor):
    def __init__(self):
        super(GPUPredictor, self).__init__()

    @staticmethod
    def predict_dmatrix(data, model, tree_begin, tree_end):
        return predict_by_all_trees(data, model, tree_begin, tree_end)

    def predict_batch(self, fmat, model, tree_begin, tree_end=0):
        if tree_end == 0:
            tree_end = len(model.trees)
        predt = self.predict_dmatrix(fmat, model, tree_begin, tree_end)
        return predt

    def predict_instance(self, inst, model, tree_end):
        return self.predict_dmatrix(inst, model, 0, tree_end)

    @staticmethod
    def predict_leaf(data, model, ntree_limit=0):
        n_sample = data.shape[0]
        n_trees = len(model.trees)
        if ntree_limit == 0 or ntree_limit > n_trees:
            ntree_limit = n_trees

        preds = np.zeros((n_sample, ntree_limit))
        for i in range(n_sample):
            for tree_id in range(n_trees):
                preds[i, tree_id] = pred_leaf_by_one_tree(data[i], model.trees[
                    tree_id])
        return preds

    def predict_contribution(self, data, model, tree_begin, tree_end,
                             approximate):
        return 0