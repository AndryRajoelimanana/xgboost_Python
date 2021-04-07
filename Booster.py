import numpy as np
from data_mat import DMatrix, bst_gpair, CacheEntry
from params import ModelParam
from objective.loss_function import SquareErrorLoss
from reg_tree import GBTree


class Booster:
    def __init__(self, params={}, cache=[], model_file = None):
        dmats = [d.handle for d in cache]
        self.handle = self.booster_create(dmats, len(cache))

    def booster_create(self, dmats, length):
        mats = np.empty(0, dtype=object)
        for i in range(length):
            mats = np.append(mats, dmats[i])
        return Booster_m(mats)


class BoostLearner:
    """
    learner-inl.hpp
    """
    def __init__(self):
        self.obj_ = None
        self.gbm_ = None
        self.name_obj_ = "reg:linear"
        self.name_gbm_ = "gbtree"
        self.silent = 0
        self.prob_buffer_row = 1.0
        self.cache_ = np.empty(0)
        self.mparam = ModelParam()
        self.preds_ = None

    def set_cache_data(self, mats):
        num_feature = 0
        buffer_size = 0
        for i in range(mats.size):
            duplicate = False
            for j in range(i):
                if mats[i] == mats[j]:
                    duplicate = True
            if duplicate:
                continue
            mats[i].cache_learner_ptr_ = self
            new_cache = CacheEntry(mats[i], buffer_size,
                                   mats[i].info.num_row())
            self.cache_ = np.append(self.cache_, new_cache)
            buffer_size += mats[i].info.num_row()
            num_feature = np.max(num_feature, mats[i].info.num_col())
        if num_feature > self.mparam.num_feature:
            setattr(self, 'bst:numfeature', num_feature)
        setattr(self, 'num_pbuffer', buffer_size)
        if not self.silent:
            print(f'buffer_size = {buffer_size}')

    def set_param(self, name, val):
        setattr(self, name, val)

    def init_model(self):
        self.init_obj_gbm()
        self.mparam.base_score = self.obj_.prob_to_margin(self.mparam.base_score)
        self.gbm_.init_model()

    def check_init(self, p_train):
        p_train.fmat().init_col_access(self.prob_buffer_row)

    def update_one_iter(self, iter, train):
        preds_ = self.predictraw(train)
        gpair = self.obj_.get_gradient(preds_, train.info, iter)
        self.gbm_.do_boost(train.fmat(), train.info.info, gpair)

    def predict(self, data, output_margin, ntree_limit=0):
        out_preds = self.predictraw(data, ntree_limit)
        if not output_margin:
            self.obj_.pred_transform(out_preds)
        return out_preds

    def init_obj_gbm(self):
        if self.obj_ is not None:
            return
        self.obj_ = SquareErrorLoss()
        self.gbm_ = GBTree()
        # for i in range(self.cfg_.size):
        #    self.obj_
        if self.evaluator_.size == 0:
            self.evaluator_.add_eval(self.obj_.default_eval_metric())

    def predictraw(self, data, ntree_limit=0):
        preds = self.gbm_.predict(data.fmat(), self.find_buffer_offset(data),
                                  data.info.info, ntree_limit)
        ndata = preds.size
        if data.info.base_margin.size != 0:
            assert preds.size == data.info.base_margin.size, "base margin"
            for j in range(ndata):
                preds[j] += data.info.base_margin[j]
        else:
            for j in range(ndata):
                preds[j] += self.mparam.base_score
        return preds

    def find_buffer_offset(self, mat):
        for i in range(self.cache_.size):
            if self.cache_[i].mat_ == mat and mat.cache_learner_ptr_ == self:
                if self.cache_[i].num_row_ == mat.info.num_row():
                    return self.cache_[i].buffer_offset_
        return -1


class Booster_m(BoostLearner):
    """
    xgboost_wrapper.cpp
    """
    def __init__(self, mats):
        self.silent = 1
        self.init_model = False
        self.set_cache_data(mats)

    def pred(self, dmat, output_margin, ntree_limit):
        self.check_init_model()
        self.predict(dmat, output_margin != 0, self.preds_, ntree_limit)
        self.length = self.preds_.size()
        return self.preds_

    def boost_one_iter(self, train, grad, hess, length):
        gpair_ = np.empty(length, dtype=object)
        ndata = length
        for j in range(ndata):
            gpair_[j] = bst_gpair(grad[j], hess[j])





if __name__ == '__main__':
    nn = np.array([[1, 0, 3, 4],
                   [0, 1, 3, 0],
                   [1, 0, 0, 0],
                   [0, 0, 1, 5]])

    mattt = DMatrix(nn)
    # bb = BoostLearner()
    # bb.set_cache_data(mattt)
    print('jj')
