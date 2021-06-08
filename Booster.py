import numpy as np
from data_i.data_mat import bst_gpair
from objective.loss_function import LinearSquare
from fako.gbtree import GBTree
from data_i.simple_matrix import DMatrix
import sys
from sklearn import datasets
from objective.evaluation import EvalSet


class Booster:
    """
    xgboost.py
    """

    def __init__(self, params={}, cache=[], model_file=None):
        dmats = [d.handle for d in cache]
        self.handle = self.booster_create(dmats, len(cache))
        self.set_param(params)

    def set_param(self, params, v=None):
        if isinstance(params, dict):
            for k, v in params.items():
                self.handle.set_param(k, v)
        elif isinstance(params, str) and v is not None:
            self.handle.set_param(params, v)
        else:
            for k, v in params:
                self.handle.set_param(k, v)

    def update(self, dtrain, it, fobj=None):
        if fobj is None:
            self.update_one_iter(it, dtrain.handle)
        else:
            pred = self.predict(dtrain)
            grad, hess = fobj(pred, dtrain)
            self.boost(dtrain, grad, hess)

    def boost(self, dtrain, grad, hess):
        self.boost_one_iter(dtrain, grad, hess, len(grad))

    def predict(self, data, output_margin=False, ntree_limit=0):
        pred = self.booster_predict(data, output_margin, ntree_limit)
        return pred

    def update_one_iter(self, i, dtrain):
        """
        XGBoosterUpdateOneIter : xgboost_wrapper.cpp
        """
        self.handle.check_init_model()
        self.handle.check_init(dtrain)
        self.handle.update_one_iter(i, dtrain)

    def boost_one_iter(self, dtrain, grad, hess, length):
        """
        XGBoosterBoostOneIter : xgboost_wrapper.cpp
        """
        self.handle.check_init_model()
        self.handle.check_init(dtrain.handle)
        self.handle.boost_one_iter(dtrain, grad, hess, length)

    @staticmethod
    def booster_create(dmats, length):
        """
        xgboost_wrapper: XGBoosterCreate
        """
        mats = []
        for i in range(length):
            mats.append(dmats[i])
        return Booster_Learn(mats)

    def booster_predict(self, data, output_margin, ntree_limit):
        """
        XGBoosterPredict xgboost_wrapper.cpp
        """
        return self.handle.pred(data, output_margin, ntree_limit)


class BoostLearner:
    """
    learner-inl.hpp (xgboost.learner.BoostLearner)
    """

    def __init__(self):
        self.obj_ = None
        self.gbm_ = None
        self.name_obj_ = "reg:linear"
        self.name_gbm_ = "gbtree"
        self.silent = 0
        self.prob_buffer_row = 1.0
        self.cache_ = []
        self.mparam = BoostLearner.ModelParam()
        self.preds_ = []
        self.cfg_ = []
        self.evaluator_ = EvalSet()

    def set_cache_data(self, mats):
        num_feature = 0
        buffer_size = 0
        assert len(self.cache_) == 0, "can only call cache data_ once"
        for i in range(len(mats)):
            for j in range(i):
                if mats[i] == mats[j]:
                    continue
            mats[i].cache_learner_ptr_ = self
            self.cache_.append(BoostLearner.CacheEntry(mats[i], buffer_size,
                                                       mats[i].info.num_row()))
            buffer_size += mats[i].info.num_row()
            num_feature = np.maximum(num_feature, mats[i].info.num_col())
        if num_feature > self.mparam.num_feature:
            self.set_param('bst:numfeature', num_feature)
        self.set_param('num_pbuffer', buffer_size)
        if not self.silent:
            print(f'buffer_size = {buffer_size}')

    def set_param(self, name, val):

        if name == 'silent':
            self.silent = val
        elif name == 'prob_buffer_row':
            self.prob_buffer_row = val
        elif name == 'eval_metric':
            self.evaluator_.add_eval(val)
        elif name == 'num_class':
            self.num_output_group = val

        if name[:4] != 'bst:':
           name = 'bst:'+name

        if self.gbm_ is None:
            if name == 'objective':
                self.name_obj_ = val
            elif name == 'booster':
                self.name_gbm_ = val
            self.mparam.set_param(name, val)
        if self.gbm_ is not None:
            self.gbm_.set_param(name, val)
        if self.obj_ is not None:
            self.obj_.set_param(name, val)
        if self.gbm_ is None or self.obj_ is None:
            self.cfg_.append((name, val))

    def init_model(self):
        self.init_obj_gbm()
        self.mparam.base_score = self.obj_.prob_to_margin(
            self.mparam.base_score)
        self.gbm_.init_model()

    def check_init(self, p_train):
        p_train.fmat().init_col_access(self.prob_buffer_row)

    def update_one_iter(self, iters, train):
        self.preds_ = self.predictraw(train, self.preds_)
        gpair = self.obj_.get_gradient(self.preds_, train.info, iters)
        self.gbm_.do_boost(train.fmat(), train.info.info, gpair)

    def eval_one_iter(self, iters, evals, evname):
        pass

    def predict(self, data, output_margin, out_preds, ntree_limit=0):
        out_pred = self.predictraw(data, out_preds, ntree_limit)
        if not output_margin:
            self.obj_.pred_transform(out_pred)
        return out_pred

    def init_obj_gbm(self):
        if self.obj_ is not None:
            return
        self.obj_ = LinearSquare()
        self.gbm_ = GBTree()
        for name, val in self.cfg_:
            self.obj_.set_param(name, val)
            self.gbm_.set_param(name, val)
        if self.evaluator_.size == 0:
            self.evaluator_.add_eval(self.obj_.default_eval_metric())

    def predictraw(self, data, out_pred, ntree_limit=0):
        fmat = data.fmat()
        preds = self.gbm_.predict(fmat, self.find_buffer_offset(data),
                                  data.info.info, out_pred, ntree_limit)
        ndata = len(preds)
        if len(data.info.base_margin_) != 0:
            assert ndata == len(data.info.base_margin_), "base margin"
            for j in range(ndata):
                preds[j] += data.info.base_margin_[j]
        else:
            for j in range(ndata):
                preds[j] += self.mparam.base_score
        return preds

    class ModelParam:
        def __init__(self, base_score=0.5, num_feature=0, num_class=0):
            self.base_score = base_score
            self.num_feature = num_feature
            self.num_class = num_class
            self.reserved = np.zeros(32)

        def set_param(self, name, val):
            if name == 'bst:num_feature':
                self.num_feature = val
            elif name == "num_class":
                self.num_class = val
            elif name == 'base_score':
                self.base_score = val

    def find_buffer_offset(self, mat):
        for i in range(len(self.cache_)):
            if self.cache_[i].mat_ == mat and mat.cache_learner_ptr_ == self:
                if self.cache_[i].num_row_ == mat.info.num_row():
                    return self.cache_[i].buffer_offset_
        return -1

    class CacheEntry:
        def __init__(self, mat, buffer_offset, num_row):
            self.mat_ = mat
            self.buffer_offset_ = buffer_offset
            self.num_row_ = num_row


class Booster_Learn(BoostLearner):
    """
    xgboost_wrapper.cpp xgboost.wrapper
    """

    def __init__(self, mats):
        super().__init__()
        self.silent = 1
        self.initmodel = False
        self.length = None
        self.set_cache_data(mats)
        print('ll')

    def pred(self, dmat, output_margin, ntree_limit):
        self.check_init_model()
        self.preds_ = self.predict(dmat, output_margin != 0, ntree_limit)
        self.length = len(self.preds_)
        return self.preds_

    def boost_one_iter(self, train, grad, hess, length):
        gpair_ = np.empty(length, dtype=object)
        ndata = length
        for j in range(ndata):
            gpair_[j] = bst_gpair(grad[j], hess[j])
        self.gbm_.do_boost(train.fmat(), train.info.info, gpair_)

    def check_init_model(self):
        if not self.initmodel:
            self.init_model()
            self.initmodel = True


def train(params, dtrain, num_boost_round=10, evals=[], obj=None, feval=None):
    """ train a booster with given paramaters
    """
    bst = Booster(params, [dtrain] + [d[0] for d in evals])
    for i in range(num_boost_round):
        bst.update(dtrain, i, obj)
        if len(evals) != 0:
            bst_eval_set = bst.eval_set(evals, i, feval)
            if isinstance(bst_eval_set, str):
                sys.stderr.write(bst_eval_set + '\n')
            else:
                sys.stderr.write(bst_eval_set.decode() + '\n')
    return bst


if __name__ == '__main__':

    diabetes = datasets.load_diabetes()
    X, y = diabetes.data, diabetes.target
    dmat = DMatrix(X, label=y)
    dmat.handle.fmat().init_col_access()
    bst = Booster(params={'bst:num_feature': 10, 'num_feature': 10,
                          'max_depth': 5, 'eta': 1, 'num_class': 1},
                  cache=[dmat])

    for it in range(3):
        bst.update(dmat, it)
        print('k')
        # bb = BoostLearner()
        # bb.set_cache_data(mattt)
    nn = bst.predict(dmat.handle, 0, 0)
    print(nn)
