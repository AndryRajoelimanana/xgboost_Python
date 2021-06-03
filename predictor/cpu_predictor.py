from predictor.predictors import Predictor
import numpy as np
import numbers




def predict_by_all_trees(fmat, model, tree_begin, tree_end):
    n_group = model.learner_model_param.num_output_group
    n_sample = fmat.shape[0]
    preds = np.full((n_sample, n_group), model.learner_model_param.base_score)
    for tree_id in range(tree_begin, tree_end):
        gid = model.tree_info[tree_id]
        for i in range(n_sample):
            preds[i, gid] += pred_value_by_one_tree(fmat[i], model.trees[
                tree_id])
    return preds


def pred_value_by_one_tree(feats, tree):
    has_missing = np.isnan(feats).any()
    lid = tree.get_leaf_index(feats, has_missing)
    print(f'leaf: {lid} , value: {tree[lid].leaf_value()}')
    return tree[lid].leaf_value()


def pred_leaf_by_one_tree(feats, tree):
    has_missing = np.isnan(feats).any()
    lid = tree.get_leaf_index(feats, has_missing)
    return lid


def predict_batch_by_block_of_rows_kernel(batch, model, tree_begin, tree_end):
    num_group = model.learner_model_param.num_output_group
    num_feature = model.learner_model_param.num_feature
    nsize = batch
    preds = predict_by_all_trees(batch, model, tree_begin, tree_end)
    return preds


class CPUPredictor(Predictor):
    def __init__(self):
        super(CPUPredictor, self).__init__()

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