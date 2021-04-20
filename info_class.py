import numpy as np


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
    def __init__(self):
        self.info = BoosterInfo()
        self.labels = self.group_ptr = self.weights = self.base_margin = []
        self.clear()

    def num_row(self):
        return self.info.num_row

    def num_col(self):
        return self.info.num_col

    def clear(self):
        self.labels.clear()
        self.group_ptr.clear()
        self.weights.clear()
        self.base_margin.clear()
        self.info = BoosterInfo(0, 0)

    def get_weight(self, i):
        if len(self.weights) != 0:
            return self.weights[i]
        else:
            return 1

    def get_param(self, field):
        return getattr(self, field)

    def set_param(self, field, val):
        setattr(self, field, val)
        return self
