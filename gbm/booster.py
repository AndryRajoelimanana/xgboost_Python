from sklearn import datasets
from param.params import *
from objective.regression_obj import LinearSquareLoss
from gbm.learner import Learner
from utils.util import one_hot_encoding

kRtEps = 1e-6


def train(params, dtrain, num_boost_round=10, evals=(), obj=None, feval=None,
          xgb_model=None, callbacks=None, evals_result=None, maximize=None,
          verbose_eval=None, early_stopping_rounds=None, labels=None,
          weights=None):
    booster = Booster(params, dtrain, labels, weights)
    if xgb_model is not None:
        booster = Booster(params, dtrain, labels, weights, model_file=xgb_model)
    if callbacks is not None or maximize is not None or \
            verbose_eval is not None or early_stopping_rounds is not None:
        pass          # todos
    if evals_result is not None or evals is not None or feval is not None:
        pass          #

    # boosting iterations
    for i in range(num_boost_round):
        booster.update(dtrain, i, obj)
    return booster


class Booster:
    def __init__(self, param=None, cache=(), labels=None, weights=None,
                 model_file=None, seed=0):

        param = param or {}
        self.data = cache
        if weights is None:
            weights = np.ones_like(labels)

        if model_file is not None:
            # todos
            pass

        if 'num_class' in param.keys():
            param['num_output_group'] = param['num_class']
            if param['num_class'] > 1:
                label = one_hot_encoding(labels)
            else:
                label = labels[..., np.newaxis]
        elif 'num_output_group' in param.keys():
            param['num_class'] = param['num_output_group']
            label = one_hot_encoding(labels)
        else:
            unique_label = len(np.unique(labels))
            if unique_label < 25:
                param['num_class'] = unique_label
                label = one_hot_encoding(labels)
            else:
                param['num_class'] = 1
                label = labels[..., np.newaxis]

        self.learner = Learner.create(cache, label, weights, seed)

        if 'booster' in param.keys():
            self.booster = param['booster']
        else:
            self.booster = 'gbtree'
        self.set_param(param)

    def set_param(self, params):
        for k, v in params.items():
            self.learner.set_param(k, v)

    def set_gpair(self, pred=0.5, obj_=None):
        obj_fn = obj_ if obj_ is not None else LinearSquareLoss()
        if isinstance(pred, float) or isinstance(pred, int):
            pred = np.full(self.data.shape[0], pred)
        grad = obj_fn.gradient(pred, y)
        hess = obj_fn.hessian(pred, y)
        return grad, hess

    def update(self, dtrain, i_iter, fobj=None):
        if fobj is None:
            self.learner.update_one_iter(i_iter, dtrain)
        else:
            ngroup = self.learner.learner_model_param_.num_output_group
            if i_iter == 0:
                base_score = self.learner.mparam_.base_score
                pred = np.full((dtrain.shape[0], ngroup), base_score)
            else:
                pred = self.predict(dtrain, output_margin=True, training=True)

            gpair = np.zeros((dtrain.shape[0], 2, ngroup))
            for i in range(ngroup):
                gpair[:, :, i] = fobj.get_gradient(pred[:, i],
                                                   self.learner.labels_[:, i],
                                                   self.learner.weights_, i)

            self.boost(dtrain, gpair, i_iter)

    def predict(self, dtrain, output_margin=False, ntree_limit=0,
                pred_leaf=False, pred_contribs=False,
                approx_contribs=False, pred_interactions=False,
                validate_features=True, training=False,
                iteration_range=(0, 0), strict_shape=False):

        layer_begin, layer_end = iteration_range
        if ntree_limit != 0:
            layer_end = layer_begin + ntree_limit * \
                        self.learner.learner_model_param_.num_output_group
        preds = self.learner.predict(dtrain, output_margin, layer_begin,
                                     layer_end, training, pred_leaf,
                                     pred_contribs, approx_contribs,
                                     pred_interactions)
        if validate_features:
            pass
        if strict_shape:
            pass
        return preds

    def boost(self, dtrain, gpair, i_iter):
        self.learner.configure()
        self.learner.boost_one_iter(i_iter, dtrain, gpair)

    def predict_raw(self, dmat, training, layer_begin, layer_end):
        return self.learner.predict_raw(dmat, training, layer_begin, layer_end)

    def print_tree(self):
        tree_dict = {}
        trees = self.learner.gbm_.model_.trees
        for tid, tree in enumerate(trees):
            for nid, node in enumerate(tree.nodes_):
                if node.info_.split_cond is not None:
                    tree_dict[
                        nid] = f'f{node.sindex_} < {node.info_.split_cond}'
                    lid = node.left_child()
                    rid = node.right_child()
                    print(f'{tid}-{nid}  {lid}-{rid}   f{node.sindex_}'
                          f' {node.info_.split_cond}')
                else:
                    print(f'{tid}-{nid}  Leaf')

    def get_tree(self, tid=0, nid=0):
        tree = self.learner.gbm_.model_.trees[tid]
        return serializable_tree(tree, nid)


def serializable_tree(tree, nid=0):
    node = tree.nodes_[nid]
    if not node.info_.split_cond:
        return "Leaf"
    lid = node.left_child()
    rid = node.right_child()

    obj = {f'cond-{nid}': f'f{node.sindex_} < '
                          f'{node.info_.split_cond}',
           f'left-{lid}': serializable_tree(tree, lid),
           f'right-{rid}': serializable_tree(tree, rid)}

    return obj


if __name__ == "__main__":
    boston = datasets.load_boston()
    data0 = boston['data']
    X = data0[:, :-1]
    y = data0[:, -1]

    # parms = {'updater': 'shotgun',
    #         'booster': 'gblinear', 'learning_rate': 1.0}

    parms = {'learning_rate': 1.0}

    bst = train(parms, X, labels=y, num_boost_round=10)
    predictions = bst.predict(X)

    print(predictions[:, 0])

    # x1 = np.array([[1.36, 15.35, 10.26, 8.16, 2.67, 12.26, 1.02, 40.20,
    #                 51.66, 59.26],
    #     [13.36, 5.35, 0.26, 84.16, 24.67, 22.26, 18.02, 14.20, 61.66, 57.26]])
    # x1 = x1.T
    # y1 = np.array(
    #     [37.54, 14.54, -0.72, 261.19, 76.90, 67.15, 53.89, 43.48, 182.60,
    #

    # params = {'booster': 'gblinear', 'updater': 'shotgun',
    # 'learning_rate': 1.0,
    #           'num_class': 1}
    # bst = train(params, x1[:, -1:], labels=y1, num_boost_round=1)
    # pred = bst.predict(x1[:, -1:], output_margin=True, training=True)
    # print(pred[:, 0])

    # print(json.dumps(tree0_dict, indent=4))
