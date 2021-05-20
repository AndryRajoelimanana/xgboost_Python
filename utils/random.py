import numpy as np
from utils.util import resize

kRtEps = 1e-6


class ColumnSampler:
    def __init__(self, seed=0):
        self.rng_ = np.random.RandomState(seed)
        self.feature_weights_ = []
        self.feature_set_tree_ = []
        self.colsample_bylevel_ = 1.0
        self.colsample_bytree_ = 1.0
        self.colsample_bynode_ = 1.0
        self.feature_set_level_ = {}

    def init(self, num_col, feature_weights, colsample_bynode,
             colsample_bylevel, colsample_bytree, skip_index_0=False):
        self.feature_weights_ = feature_weights
        self.colsample_bylevel_ = colsample_bylevel
        self.colsample_bytree_ = colsample_bytree
        self.colsample_bynode_ = colsample_bynode
        feature_set_tree = list(range(int(skip_index_0), num_col))

        self.feature_set_tree_ = self.col_sample(feature_set_tree,
                                           self.colsample_bytree_)

    def weighted_sampling_without_replacement(self, feat, weights, n):
        assert len(feat) == len(weights)
        keys = [None] * len(weights)
        for i in range(len(feat)):
            w = np.maximum(weights[i], kRtEps)
            u = self.rng_.uniform()
            k = np.log(u) / w
            keys[i] = k
        ind = list(np.argsort(keys))
        ind.reverse()
        resize(ind, n)
        results = [None] * len(ind)
        for k in range(len(ind)):
            results[k] = feat[ind[k]]
        return results

    def col_sample(self, p_features, colsample):
        if colsample == 1:
            return p_features
        features = p_features
        assert len(features) > 0
        n = np.maximum(1, colsample*len(features))
        if len(self.feature_weights_) != 0:
            new_features = self.weighted_sampling_without_replacement(
                p_features, self.feature_weights_, n)
        else:
            new_features = features.copy()
            self.rng_.shuffle(new_features)
            resize(new_features, n)
        return sorted(new_features)

    def get_feature_set(self, depth):
        if self.colsample_bylevel_ == 1.0 and self.colsample_bynode_ == 1.0:
            return self.feature_set_tree_
        if depth not in self.feature_set_level_.keys():
            self.feature_set_level_[depth] = self.col_sample(
                self.feature_set_tree_, self.colsample_bylevel_)
        if self.colsample_bynode_ == 1.0:
            return self.feature_set_level_[depth]
        return self.col_sample(self.feature_set_level_[depth],
                               self.colsample_bynode_)
