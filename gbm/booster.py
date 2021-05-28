from sklearn import datasets
from params import *
from objective.loss_function import LinearSquareLoss
from gbm.learner import LearnerImpl
from utils.util import one_hot_encoding


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
                 model_file=None):

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

        self.learner = LearnerImpl(cache, label, weights)

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

    def update(self, train, i, fobj=None):
        if fobj is None:
            self.learner.update_one_iter(i, train)
        else:
            ngroup = self.learner.learner_model_param_.num_output_group
            if i == 0:
                base_score = self.learner.mparam_.base_score
                pred = np.full((train.shape[0], ngroup), base_score)
            else:
                pred = self.predict(train, output_margin=True, training=True)

            grad = fobj.gradient(pred, self.learner.labels_)
            hess = fobj.hessian(pred, self.learner.labels_)
            self.boost(train, grad, hess)

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

    def boost(self, dtrain, grad, hess):
        self.learner.configure()
        gpair = (grad, hess)
        self.learner.boost_one_iter(0, dtrain, gpair)

    def predict_raw(self, dmat, training, layer_begin, layer_end):
        return 0


if __name__ == "__main__":
    boston = datasets.load_boston()
    data0 = boston['data']
    X = data0[:, :-1]
    y = data0[:, -1]
    # gg = LinearSquareLoss()
    # base_scores = 0.5

    # base_s = np.full(y.shape, base_scores)
    # grad0 = gg.gradient(base_s, y)
    # hess0 = gg.hessian(base_s, y)
    # trees0 = [RegTree()]

    # bst = Booster({'num_parallel_tree': 5, 'updater': 'grow_colmaker'}, X, y)
    # bst.update(X, 0, LinearSquareLoss())
    params = {'num_parallel_tree': 5, 'updater': 'grow_colmaker',
              'colsample_bylevel': 0.5}
    bst = train(params, X, labels=y, obj=LinearSquareLoss(), num_boost_round=4)

    # colmaker = ColMaker()
    # colmaker.update(grad0, hess0, X, trees0[0])

    # gpair = []
    # stats = GradStats()
    #
    # for i in range(len(grad)):
    #     gpair.append(GradientPair(grad[i], hess[i]))
    #     stats.add(GradientPair(grad[i], hess[i]))
    #
    # bb = DMat(csc_matrix(X))
    # nnn = bb.get_col(1)
    #
    # # pos = positions(gpair)
    #
    # p = TrainParam()
    # ev = TreeEvaluator(p, 12, -1).get_evaluator()
    # snode_stats = get_snode(pos, gpair)
    # snode_weight = ev.calc_weight(-1, p, snode_stats)
    # snode_root_gain = ev.calc_gain(-1, p, snode_stats)
    #
    # bbb = get_loss(X, p, grad, hess)

    # w = calc_weight(p, gpair)
    # gain = calc_gain(p, stats)

    print(0)
