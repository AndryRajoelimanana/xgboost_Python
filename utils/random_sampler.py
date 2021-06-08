import numpy as np
from utils.util import resize, check_random_state

kRtEps = 1e-6


class ColumnSampler:
    def __init__(self, random_state=0):
        self.rng_ = check_random_state(random_state)
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

        self.feature_set_tree_ = []
        self.feature_set_level_ = {}

        feature_set_tree_ = list(range(int(skip_index_0), num_col))

        self.feature_set_tree_ = self.col_sample(feature_set_tree_,
                                                 self.colsample_bytree_)

    def col_sample(self, p_features, colsample):
        if colsample == 1.0:
            return p_features
        features = p_features
        assert len(features) > 0
        n = np.maximum(1, np.floor(colsample * len(features)))
        fweights = self.feature_weights_
        if len(fweights) != 0:
            new_features = weighted_sampling_without_replacement(self.rng_,
                                                                 p_features,
                                                                 fweights,
                                                                 n)
        else:
            new_features = features.copy()
            new_features = shuffle_std(new_features, self.rng_)
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


def weighted_sampling_without_replacement(rng, feat, weights, n):
    assert len(feat) == len(weights)
    keys = [None] * len(weights)
    for i in range(len(feat)):
        w = np.maximum(weights[i], kRtEps)
        u = uniform_real_distribution(rng)
        k = np.log(u) / w
        keys[i] = k
    ind = list(np.argsort(keys))
    ind.reverse()
    resize(ind, n)
    return [feat[i] for i in ind]


def uniform_real_distribution(rng, param=(0, 2147483647), num_bit=24):
    b = np.minimum(24, num_bit)
    r = 2 ** 32
    log2r = int(np.log(r) / np.log(2))
    k = np.maximum(1, int((b + log2r - 1) / log2r))
    summ = 0
    tmpp = 1
    while k > 0:
        summ += (rng.randint(2 ** 32)) * tmpp
        tmpp *= r
        k -= 1
    return (summ / tmpp) * (param[1] - param[0]) + param[0]


def uniform_int_distribution(rng, min_value=0, max_value=2147483647):
    """
    https://gcc.gnu.org/onlinedocs/libstdc++/libstdc++-api-4.5/a00987_source.html
    :param max_value:
    :param min_value:
    :param_ rng:
    :return:
    """
    rng = check_random_state(rng)
    urange = max_value - min_value
    urnrange = 2 ** 32 - 1
    tmp = 0
    ret = 2 ** 34
    if urnrange > urange:
        uerange = urange + 1
        scaling = int(urnrange / uerange)
        past = uerange * scaling
        while ret >= past:
            ret = rng.randint(2 ** 32)
        ret = int(ret / scaling)
    elif urnrange < urange:
        while ret > urange or ret < tmp:
            uerngrange = urnrange + 1
            new_max = urange / uerngrange
            tmp = uerngrange * uniform_int_distribution(rng, 0, new_max)
            ret - tmp + rng.randint(2 ** 32)
    else:
        ret = rng.randint(2 ** 32)
    return ret + min_value


def shuffle_std(data, rng, crange=(0, 0)):
    first = crange[0]
    if crange[1] == 0:
        last = len(data) - 1
    else:
        last = crange[1]
    urnrange = 2 ** 32 - 1
    urange = last - first + 1
    if (urnrange / urange) >= urange:
        i = first + 1
        if urange % 2 == 0:
            i_rnd = first + uniform_int_distribution(rng, 0, 1)
            swap(data, i, i_rnd)
            i += 1
        while i <= last:
            swap_range = i - first + 1
            a1, a2 = gen_two_uniform_ints(rng, swap_range, swap_range + 1)
            swap(data, i, first + a1)
            i += 1
            swap(data, i, first + a2)
            i += 1
    else:
        for i in range(first + 1, last):
            i_rnd = uniform_int_distribution(rng, 0, i - first)
            swap(data, i, first + i_rnd)
    return data


def swap(data, i1, i2):
    tmp = data[i1]
    data[i1] = data[i2]
    data[i2] = tmp


def gen_two_uniform_ints(rng, a, b):
    x = uniform_int_distribution(rng, 0, int(a * b - 1))
    return x // b, x % b


if __name__ == '__main__':
    nn1 = np.random.RandomState(0)
    ppp = []
    for _ in range(5):
        ppp.append(uniform_real_distribution(nn1, (0, 5)))
    print(ppp)

    print('numpy')
    nn1 = np.random.RandomState(0)
    dd = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    nn1.shuffle(dd)
    print(dd)

    # same as std::shuffle
    nn1 = np.random.RandomState(0)
    print(shuffle_std([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], nn1))
