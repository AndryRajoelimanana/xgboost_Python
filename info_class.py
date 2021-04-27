import numpy as np
from utils.util import resize


class BoosterInfo:
    def __init__(self, num_row=0, num_col=0):
        self.num_row = num_row
        self.num_col = num_col
        self.root_index = []
        self.fold_index = []

    def get_root(self, i):
        if len(self.root_index) == 0:
            return 0
        else:
            return self.root_index[i]


class MetaInfo:
    kNumField = 11

    def __init__(self, num_row=0, num_col=0):
        self.info = BoosterInfo()
        self.num_row_ = num_row
        self.num_col_ = num_col
        self.num_nonzero_ = 0
        self.labels_ = self.group_ptr_ = self.weights_ = self.base_margin_ = []
        self.labels_lower_bound_ = self.labels_upper_bound_ = []
        self.feature_type_names = self.feature_names = self.feature_types = []
        self.feature_weigths = self.label_order_cache_ = []
        self.clear()

    def num_row(self):
        return self.info.num_row

    def num_col(self):
        return self.info.num_col

    def clear(self):
        self.labels_.clear()
        self.group_ptr_.clear()
        self.weights_.clear()
        self.base_margin_.clear()
        self.info = BoosterInfo(0, 0)

    def get_weight(self, i):
        if len(self.weights_) != 0:
            return self.weights_[i]
        else:
            return 1

    def label_abs_sort(self):
        if self.label_order_cache_ == len(self.labels_):
            return self.label_order_cache_
        self.label_order_cache_ = [i[0] for i in sorted(enumerate(
            self.labels_), key=lambda x: abs(x[1]))]
        return self.label_order_cache_

    def get_param(self, field):
        return getattr(self, field)

    def set_param(self, field, val):
        setattr(self, field, val)
        return self

    def set_info(self, key, val):
        if key == 'label':
            self.labels_ = val
        elif key == 'weight':
            self.weights_ = val
            assert np.all([w >= 0 for w in
                           self.weights_]), "Weights must be positive values."
        elif key == 'base_margin':
            self.base_margin_ = val
        elif key == 'group':
            self.group_ptr_ = np.cumsum(val)
        elif key == 'qid':
            is_sorted = lambda a: np.all(a[:-1] <= a[1:])
            assert is_sorted(val), "`qid` must be sorted in non-decreasing " \
                                   "order along with data. "
            self.group_ptr_ = list(np.unique(val))
            if self.group_ptr_[-1] != len(val):
                self.group_ptr_.append(len(val))
        elif key == 'label_lower_bound':
            self.labels_lower_bound_ = val
        elif key == 'label_upper_bound':
            self.labels_upper_bound_ = val
        elif key == 'feature_weights':
            assert np.all(val >= 0)
            self.feature_weigths = val
        else:
            raise NameError('Unknown key for MetaInfo: ')

    def get_info(self, key):
        if key == "label":
            return self.labels_
        elif key == "weight":
            return self.weights_
        elif key == "base_margin":
            return self.base_margin_
        elif key == "label_lower_bound":
            return self.labels_lower_bound_
        elif key == "label_upper_bound":
            return self.labels_upper_bound_
        elif key == "feature_weights":
            return self.feature_weigths
        elif key == "group_ptr":
            return self.group_ptr_
        else:
            raise NameError("Unknown float field name: {key}")

    def slice(self, ridxs):
        out = MetaInfo(len(ridxs), self.num_col_)
        out.labels_ = gather(self.labels_, ridxs)
        out.labels_upper_bound_ = gather(self.labels_upper_bound_, ridxs)
        out.labels_lower_bound_ = gather(self.labels_lower_bound_, ridxs)
        if len(self.weights_) + 1 == len(self.group_ptr_):
            h_weights = out.weights_
        else:
            out.weights_ = gather(self.weights_, ridxs)
        if len(self.base_margin_) != self.num_row_:
            assert len(self.base_margin_) % self.num_row_ == 0, "Incorrect " \
                                                                "size of base" \
                                                                " margin " \
                                                                "vector. "
            stride = len(self.base_margin_) / self.num_row_
            out.base_margin_ = gather(self.base_margin_, ridxs, stride)
        else:
            out.base_margin_ = gather(self.base_margin_, ridxs)

        out.feature_weigths = self.feature_weigths
        out.feature_names = self.feature_names
        out.feature_types = self.feature_types
        out.feature_type_names = self.feature_type_names
        return out


def gather(ins, ridxs, stride=1):
    if len(ins) == 0:
        return []
    size = len(ridxs)
    out = [] * size * stride
    for i in range(size):
        ridx = ridxs[i]
        for j in range(stride):
            out[i * stride + j] = ins[ridx * stride + j]
    return out
