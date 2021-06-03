import numpy as np
import numbers


class Predictor:
    def __init__(self, generic_param):
        self.__generic_param_ = generic_param

    def configure(self, cfg):
        pass


def create_predictor(name, generic_param):
    if name == 'cpu_predictor':
        pred = CPUPredictor(generic_param)
    elif name == 'gpu_prediction':
        pred = GPUPredictor(generic_param)
    else:
        raise ValueError(f"Unknown GradientBooster: {name}")
    return pred


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
    def __init__(self, generic_param):
        super(CPUPredictor, self).__init__(generic_param)

    @staticmethod
    def init_prediction(model, nrow=1, base_margin=None):
        ngroup = model.learner_model_param.num_output_group
        n = ngroup * nrow
        if isinstance(base_margin, np.ndarray):
            assert n == base_margin.shape[0]
            return base_margin
        elif isinstance(base_margin, numbers.Number):
            init_pred = np.full((nrow, ngroup), base_margin)
            return init_pred
        elif base_margin is None:
            base_score = model.learner_model_param.base_score
            base_margin = np.full((nrow, ngroup), base_score)
            return base_margin
        else:
            raise Exception(f"Invalid base_margin: {base_margin}")


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


class GPUPredictor(Predictor):
    def __init__(self, generic_param):
        super(GPUPredictor, self).__init__(generic_param)

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