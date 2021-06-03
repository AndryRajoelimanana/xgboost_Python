from sklearn import datasets
from params import *
from objective.regression_obj import LinearSquareLoss
from gbm.learner import LearnerImpl
from utils.util import one_hot_encoding
import json


kRtEps = 1e-6


def train(params, dtrain, num_boost_round=10, evals=(), obj=None, feval=None,
          xgb_model=None, callbacks=None, evals_result=None, maximize=None,
          verbose_eval=None, early_stopping_rounds=None, labels=None,
          weights=None):
    bst = Booster(params, dtrain, labels, weights)
    for i in range(num_boost_round):
        bst.update(dtrain, i, obj)
    return bst


class Booster:
    def __init__(self, param=None, cache=(), labels=None, weights=None,
                 model_file=None, seed=0):

        param = param or {}
        self.data = cache
        if weights is None:
            weights = np.ones_like(labels)

        if 'num_class' in param.keys():
            param['num_output_group'] = param['num_class']
            label = one_hot_encoding(labels)
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

        self.learner = LearnerImpl(cache, label, weights, seed)

        if 'booster' in param.keys():
            self.booster = param['booster']
        else:
            self.booster = 'gbtree'
        self.set_param(param)

    def set_param(self, params, value=None):
        for k, v in params.items():
            self.learner.set_param(k, v)

    def set_gpair(self, pred=0.5, obj_=None):
        obj_fn = obj_ if obj_ is not None else LinearSquareLoss()
        if isinstance(pred, float) or isinstance(pred, int):
            pred = np.full(self.data.shape[0], pred)
        grad = obj_fn.gradient(pred, y)
        hess = obj_fn.hessian(pred, y)
        return grad, hess

    def update(self, train, i_iter, fobj=None):
        if fobj is None:
            self.learner.update_one_iter(i_iter, train)
        else:
            ngroup = self.learner.learner_model_param_.num_output_group
            if i_iter == 0:
                base_score = self.learner.mparam_.base_score
                pred = np.full((train.shape[0], ngroup), base_score)
            else:
                pred = self.predict(train, output_margin=True, training=True)

            gpair = np.zeros((train.shape[0], 2, ngroup))
            for i in range(ngroup):
                gpair[:, :, i] = fobj.get_gradient(pred[:, i],
                                                   self.learner.labels_[:, i],
                                                   self.learner.weights_, i)
            self.boost(train, gpair, i_iter)

    def predict(self, train, output_margin=False, ntree_limit=0,
                pred_leaf=False, pred_contribs=False,
                approx_contribs=False, pred_interactions=False,
                validate_features=True, training=False,
                iteration_range=(0, 0), strict_shape=False):

        layer_begin, layer_end = iteration_range
        preds = self.learner.predict(train, output_margin, layer_begin,
                                     layer_end, training, pred_leaf,
                                     pred_contribs, approx_contribs,
                                     pred_interactions)
        return preds

    def boost(self, dtrain, gpair, i_iter):
        self.learner.configure()
        # gpair = (grad, hess)
        self.learner.boost_one_iter(i_iter, dtrain, gpair)

    def predict_raw(self, dmat, training, layer_begin, layer_end):
        return 0

    def print_tree(self):
        tree_dict = {}
        trees = self.learner.gbm_.model_.trees
        for tid, tree in enumerate(trees):
            for nid, node in enumerate(tree.nodes_):
                if node.info_.split_cond is not None:
                    tree_dict[nid] = f'f{node.sindex_} < {node.info_.split_cond}'
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
           f'right-{rid}':  serializable_tree(tree, rid)}

    return obj


if __name__ == "__main__":
    boston = datasets.load_boston()
    data0 = boston['data']
    X = data0[:, :-1]
    y = data0[:, -1]

    params = {'num_parallel_tree': 5, 'updater': 'grow_colmaker',
              'colsample_bylevel': 0.5}
    bst = train(params, X, labels=y, obj=LinearSquareLoss(), num_boost_round=5)

    tree0_dict = bst.get_tree(0)

    pred = bst.predict(X, output_margin=True, training=True)
    # print(pred[:, 0])
    # print(json.dumps(tree0_dict, indent=4))
