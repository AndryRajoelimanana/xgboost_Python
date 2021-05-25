import numpy as np


class Predictor:
    def __init__(self, generic_param):
        self.generic_param_ = generic_param


def predict_by_all_trees(fmat, model, tree_begin, tree_end):
    n_group = model.learner_model_param.num_output_group
    n_sample = fmat.shape[0]
    preds = np.zeros((n_sample, n_group))
    for tree_id in range(tree_begin, tree_end):
        gid = model.tree_info[tree_id]
        for i in range(n_sample):
            preds[i, gid] += pred_value_by_one_tree(fmat[i], model.trees[
                tree_id])
    return preds


def pred_value_by_one_tree(feats, tree):
    has_missing = np.isnan(feats).any()
    lid = tree.get_leaf_index(feats, has_missing)
    return tree[lid].leaf_value()


def predict_batch_by_block_of_rows_kernel(batch, model, tree_begin, tree_end):
    num_group = model.learner_model_param.num_output_group
    num_feature = model.learner_model_param.num_feature
    nsize = batch
    preds = predict_by_all_trees(batch, model, tree_begin, tree_end)
    return preds


class CPUPredictor(Predictor):
    def __init__(self):
        pass

    def predict_dmatrix(self, fmat, model, tree_begin, tree_end):
        return predict_batch_by_block_of_rows_kernel(fmat, model, tree_begin,
                                                     tree_end)

    def predict_batch(self, fmat, model, tree_begin, tree_end=0):
        if tree_end == 0:
            tree_end = len(model.trees)
        predt = self.predict_dmatrix(fmat, model, tree_begin, tree_end)
        print(predt)
        return predt