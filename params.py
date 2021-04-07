import numpy as np


class ModelParam:
    def __init__(self, base_score=0.5, num_feature=0, num_class=0):
        self.base_score = base_score
        self.num_feature = num_feature
        self.num_class = num_class
        self.reserved = np.zeros(32)

    def set_param(self, name, val):
        setattr(self, name, val)


def thresholdL1(w, alpha):
    if w > + alpha:
        return w - alpha
    if w < - alpha:
        return w + alpha
    return 0.0


class TrainParam:
    """
    param.h
    """
    def __init__(self):
        self.learning_rate = 0.3
        self.min_child_weight = 1.0
        self.max_depth = 6
        self.reg_lambda = 1.0
        self.reg_alpha = 0.0
        self.default_direction = 0
        self.subsample = 1.0
        self.colsample_bytree = 1.0
        self.colsample_bylevel = 1.0
        self.opt_dense_col = 1.0
        self.nthread = 0
        self.size_leaf_vector = 0
        self.min_split_loss = 0

    def set_param(self, name, value):
        setattr(self, name, value)

    def calc_gain(self, sum_grad, sum_hess):
        if sum_hess < self.min_child_weight:
            return 0
        if self.reg_alpha == 0:
            return (sum_grad*sum_grad)/(sum_hess + self.reg_lambda)
        else:
            return (thresholdL1(sum_grad, self.reg_alpha))**2 / (sum_hess +
                                                                 self.reg_lambda)

    def calc_gain_cost(self, sum_grad, sum_hess, test_grad, test_hess):
        w = self.calc_weight(sum_grad, sum_hess)
        ret = test_grad * w + 0.5 * (test_hess + self.reg_lambda)*(w**2)
        if self.reg_alpha == 0:
            return -1 * ret
        else:
            return -2 * (ret + self.reg_alpha*np.abs(w))

    def calc_weight(self, sum_grad, sum_hess):
        if sum_hess < self.min_child_weight:
            return 0
        if self.reg_alpha == 0:
            return - sum_grad/ (sum_hess + self.reg_lambda)
        else:
            return thresholdL1(sum_grad, self.reg_alpha)/(sum_hess +
                                                          self.reg_lambda)

    def need_prune(self, loss_chg, depth):
        return loss_chg < self.min_split_loss

    def cannot_split(self, sum_hess, depth):
        return sum_hess < self.min_child_weight*2


class GradStats:
    """
    param.h
    """
    def __init__(self, param=None):
        if param is None:
            self.param = TrainParam()
        self.sum_grad = self.sum_hess = 0

    @staticmethod
    def check_info(info):
        pass

    def add(self, grad, hess):
        self.sum_grad += grad
        self.sum_hess += hess

    def add_stats(self, gpair, info, ridx):
        b = gpair[ridx]
        self.add_pair(b)

    def add_pair(self, b):
        self.add(b.sum_grad, b.sum_hess)

    def calc_gain(self, param):
        return param.calc_gain(self.sum_grad, self.sum_hess)

    def calc_weight(self, param):
        return param.calc_weight(self.sum_grad, self.sum_hess)

    def set_substract(self, a, b):
        self.sum_grad = a.sum_grad - b.sum_grad
        self.sum_hess = a.sum_hess - b.sum_hess

    def empty(self):
        return self.sum_hess == 0


class SplitEntry:
    """
    param.h
    """
    def __init__(self):
        self.loss_chg = 0.0
        self.sindex = 0
        self.split_value = 0.0

    def need_replace(self, new_loss_change, split_index):
        if self.split_index() <= split_index:
            return new_loss_change > self.loss_chg
        else:
            return not (self.loss_chg > new_loss_change)

    def update(self, e):
        if self.need_replace(e.loss_chg, e.split_index()):
            self.loss_chg = e.loss_chg
            self.sindex = e.sindex
            self.split_value = e.split_value
            return True
        else:
            return False

    def update(self, new_loss_chg, split_index, new_split_value,
               default_left):
        if self.need_replace(new_loss_chg, split_index):
            self.loss_chg = new_loss_chg
            if default_left:
                split_index |= (1 << 31)
            self.sindex = split_index
            self.split_value = new_split_value
            return True
        else:
            return False

    def split_index(self):
        return self.sindex & ((1 << 31) - 1)

    def default_left(self):
        return (self.sindex >> 31) != 0