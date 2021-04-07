import numpy as np


class BoosterInfo:
    def __init__(self, num_row=0, num_col=0):
        self.num_row = num_row
        self.num_col = num_col
        self.root_index = np.empty(0)
        self.fold_index = np.empty(0)

    def get_root(self, i):
        if self.root_index.size == 0:
            return 0
        else:
            return self.root_index[i]


class MetaInfo:
    def __init__(self):
        self.info = BoosterInfo()
        self.clear()

    def num_row(self):
        return self.info.num_row

    def num_col(self):
        return self.info.num_col

    def clear(self):
        self.labels = np.empty(0)
        self.group_ptr = np.empty(0)
        self.weights = np.empty(0)
        self.base_margin= np.empty(0)
        self.info = BoosterInfo()

    def get_weight(self, i):
        if self.weights.size != 0:
            return self.weights[i]
        else:
            return 1

    def get_param(self, field):
        return getattr(self, field)

    def set_param(self, field, val):
        setattr(self, field, val)
        return self
