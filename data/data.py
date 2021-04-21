import numpy as np
from utils.util import resize


# src/data/data.c

# namespace: xgboost

class MetaInfo:
    kNumField = 11

    def __init__(self):
        self.num_row_ = self.num_col_ = self.num_nonzero_ = 0
        self.labels_ = []
        self.root_index_ = []
        self.group_ptr_ = []
        self.weights_ = []
        self.base_margin_ = []
        self.labels_lower_bound_ = []
        self.labels_upper_bound_ = []

    def get_weight(self, i):
        if len(self.weights_) != 0:
            return self.weights_[i]
        else:
            return 1

    def clear(self):
        self.num_row_ = self.num_col_ = self.num_nonzero_ = 0
        self.labels_.clear()
        self.root_index_.clear()
        self.group_ptr_.clear()
        self.weights_.clear()
        self.base_margin_.clear()

    def set_info(self, key, val):
        if key == "root_index":
            self.root_index_ = val
        elif key == "label":
            self.labels_ = val
        elif key == "weight":
            self.weights_ = val
        elif key == "base_margin":
            self.base_margin_ = val
        elif key == "group":
            size = len(val)
            self.group_ptr_ = [0]*size
            self.group_ptr_ += val
            for i in range(size):
                self.group_ptr_[i] = self.group_ptr_[i-1] + self.group_ptr_[i]


class Entry:
    def __init__(self, index, fvalue):
        """ Entry of sparse vector"""
        self.index = index
        self.fvalue = fvalue

    @staticmethod
    def cmp_value(a, b):
        return a.fvalue < b.fvalue

    def __eq__(self, other):
        return (self.index == other.index) and (self.fvalue == other.fvalue)


class BatchParam:
    def __init__(self, device, max_bin, gpu_page_size=0):
        self.gpu_id = device
        self.max_bin = max_bin
        self.gpu_page_size = gpu_page_size

    def __ne__(self, other):
        return (self.gpu_id != other.gpu_id) or (
                self.max_bin != other.max_bin) or (
                self.gpu_page_size != other.gpu_page_size)


class HostSparsePageView:
    def __init__(self):
        self.offset = None
        self.data = None
    def __getitem__(self, item):
        size = self.offset.data() + i + 1
