import numpy as np


def thresholdL1(w, alpha):
    if w > + alpha:
        return w - alpha
    if w < - alpha:
        return w + alpha
    return 0.0


def calc_gain_given_weight(p, sum_grad, sum_hess, w):
    if sum_hess <= 0:
        return 0
    return -(2.0 * sum_grad * w + (sum_hess + p.reg_lambda) * (w * w))


def calc_weight(p, sum_grad, sum_hess):
    if sum_hess < p.min_child_weight or sum_hess <= 0:
        return 0
    dw = - thresholdL1(sum_grad, p.reg_alpha)/(sum_hess + p.reg_lambda)
    if p.max_delta_step != 0.0 and np.abs(dw) > p.max_delta_step:
        dw = np.copysign(p.max_delta_step, dw)
    return dw


def calc_gain(p, sum_grad, sum_hess):
    if sum_hess < p.min_child_weight:
        return 0
    if p.max_delta_step == 0.0:
        if p.reg_alpha == 0.0:
            return (sum_grad * sum_grad) / (sum_hess + p.reg_lambda)
        else:
            dw = thresholdL1(sum_grad, p.reg_alpha)
            return (dw*dw) / (sum_hess + p.reg_lambda)
    else:
        w = calc_weight(p, sum_grad, sum_hess)
        ret = calc_gain_given_weight(p, sum_grad, sum_hess, w)
        if p.reg_alpha == 0:
            return ret
        else:
            return ret + p.reg_alpha * np.abs(w)


def calc_gain_stat(p, stat):
    return calc_gain(p, stat.get_grad(), stat.get_hess())


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
        if name == 'gamma':
            self.min_split_loss = value
        elif name == 'eta':
            self.learning_rate = value
        elif name == 'lambda':
            self.reg_lambda = value
        elif name == 'learning_rate':
            self.learning_rate = value
        elif name == 'min_child_weight':
            self.min_child_weight = value
        elif name == 'min_split_loss':
            self.min_split_loss = value
        elif name == 'reg_lambda':
            self.reg_lambda = value
        elif name == 'reg_alpha':
            self.reg_alpha = value
        elif name == 'subsample':
            self.subsample = value
        elif name == 'colsample_bylevel':
            self.colsample_bylevel = value
        elif name == 'colsample_bytree':
            self.colsample_bytree = value
        elif name == 'opt_dense_col':
            self.opt_dense_col = value
        elif name == 'size_leaf_vector':
            self.size_leaf_vector = value
        elif name == 'max_depth':
            self.max_depth = value
        elif name == 'nthread':
            self.nthread = value
        elif name == 'default_direction':
            if value == 'learn':
                self.default_direction = 0
            elif value == 'left':
                self.default_direction = 1
            elif value == 'right':
                self.default_direction = 2

    def need_prune(self, loss_chg, depth):
        return loss_chg < self.min_split_loss or (
                self.max_depth != 0 and depth > self.max_depth)

    def calc_gain(self, sum_grad, sum_hess):
        if sum_hess < self.min_child_weight:
            return 0
        if self.reg_alpha == 0:
            return (sum_grad * sum_grad) / (sum_hess + self.reg_lambda)
        else:
            return (thresholdL1(sum_grad, self.reg_alpha)) ** 2 / (sum_hess +
                                                                   self.reg_lambda)

    def need_forward_search(self, col_density=0.0):
        return (self.default_direction == 2) or ((self.default_direction == 0)
                                                 and (
                                                             col_density < self.opt_dense_col))

    def need_backward_search(self, col_density=0.0):
        return self.default_direction != 2

    def calc_gain_cost(self, sum_grad, sum_hess, test_grad, test_hess):
        w = calc_weight(p, sum_grad, sum_hess)
        ret = test_grad * w + 0.5 * (test_hess + self.reg_lambda) * (w ** 2)
        if self.reg_alpha == 0:
            return -1 * ret
        else:
            return -2 * (ret + self.reg_alpha * np.abs(w))

    def calc_gain_given_weight(self, p, sum_grad, sum_hess, w):
        if sum_hess <= 0:
            return 0
        # avoiding - 2 (G*w + 0.5(H+lambda)*w^2 (using obj = G^2/(H+lambda))
        if not self.has_constraint:
            return (thresholdL1(sum_grad, p.reg_alpha) ** 2) / (sum_hess +
                                                                p.reg_lambda)
        return -(2.0 * sum_grad * w + (sum_hess + p.reg_lambda) * (w * w))

    def cannot_split(self, sum_hess, depth):
        return sum_hess < self.min_child_weight * 2


class GradStats:
    """
    param.h
    """

    def __init__(self, sum_grad=0, sum_hess=0, param=None):
        if param is None:
            self.param = TrainParam()
        self.sum_grad = sum_grad
        self.sum_hess = sum_hess

    def clear(self):
        self.sum_grad = 0
        self.sum_hess = 0

    @staticmethod
    def check_info(info):
        pass

    def get_grad(self):
        return self.sum_grad

    def get_hess(self):
        return self.sum_hess

    def add(self, grad, hess):
        self.sum_grad += grad
        self.sum_hess += hess

    def __add__(self, other):
        self.sum_grad += other.sum_grad
        self.sum_hess += other.sum_hess
        return self

    @staticmethod
    def reduce(a, b):
        return a + b

    @staticmethod
    def set_substract(a, b):
        return a - b

    def empty(self):
        return self.sum_hess == 0

    def add_stats(self, gpair, info, ridx):
        b = gpair[ridx]
        self.add(b.grad, b.hess)

    def add_pair(self, b):
        self.add(b.sum_grad, b.sum_hess)

    def calc_gain(self, param):
        return param.calc_gain(self.sum_grad, self.sum_hess)

    def calc_weight(self, param):
        return param.calc_weight(self.sum_grad, self.sum_hess)

    def set_leaf_vec(self, param, vec):
        pass


class SplitEntry:
    """
    param.h
    """

    def __init__(self, loss_chg=0, sindex=0, split_value=0):
        self.loss_chg = loss_chg
        self.sindex = sindex
        self.split_value = split_value
        self.left_sum = 0
        self.right_sum = 0

    def split_index(self):
        return self.sindex & ((1 << 31) - 1)

    def default_left(self):
        return (self.sindex >> 31) != 0

    def need_replace(self, new_loss_chg, split_index):
        if np.isinf(new_loss_chg):
            return False
        if self.split_index() <= split_index:
            return new_loss_chg > self.loss_chg
        else:
            return not (self.loss_chg > new_loss_chg)

    def update_e(self, e):
        if self.need_replace(e.loss_chg, e.split_index()):
            self.loss_chg = e.loss_chg
            self.sindex = e.sindex
            self.split_value = e.split_value
            self.left_sum = e.left_sum
            self.right_sum = e.right_sum
            return True
        else:
            return False

    def update(self, new_loss_chg, split_index, new_split_value,
               default_left, left_sum, right_sum):
        if self.need_replace(new_loss_chg, split_index):
            self.loss_chg = new_loss_chg
            if default_left:
                split_index |= (1 << 31)
            self.sindex = split_index
            self.split_value = new_split_value
            self.left_sum = left_sum
            self.right_sum = right_sum
            return True
        else:
            return False

    @staticmethod
    def reduce(dst, src):
        dst.update_e(src)


class GradientPair:
    def __init__(self, grad = 0, hess =0 ):
        self.grad_ = grad
        self.hess_ = hess

    def set_grad(self, g):
        self.grad_ = g

    def set_hess(self, h):
        self.hess_ = h

    def get_grad(self, g):
        return self.grad_

    def get_hess(self, h):
        return self.hess_

    def add(self, grad, hess):
        self.grad_ += grad
        self.hess_ += hess

    def __add__(self, other):
        g = GradientPair(self.grad_, self.hess_)
        g.grad_ += other.grad_
        g.hess_ += other.hess_
        return g

    def __iadd__(self, other):
        self.grad_ += other.grad_
        self.hess_ += other.hess_
        return self

    def __sub__(self, other):
        g = GradientPair(self.grad_, self.hess_)
        g.grad_ -= other.grad_
        g.hess_ -= other.hess_
        return g

    def __isub__(self, other):
        self.grad_ -= other.grad_
        self.hess_ -= other.hess_
        return self

    def __mul__(self, other):
        g = GradientPair(self.grad_, self.hess_)
        g.grad_ *= other.grad_
        g.hess_ *= other.hess_
        return g

    def __imul__(self, other):
        self.grad_ *= other.grad_
        self.hess_ *= other.hess_
        return self

    def __truediv__(self, other):
        g = GradientPair(self.grad_, self.hess_)
        g.grad_ /= other.grad_
        g.hess_ /= other.hess_
        return g

    def __idiv__(self, other):
        self.grad_ /= other.grad_
        self.hess_ /= other.hess_
        return self

    def __eq__(self, other):
        return self.grad_ == other.grad_ and self.hess_ == other.hess_

    @staticmethod
    def reduce(self, a, b):
        a += b


def ParseInteractionConstraint(constraint_str, out):
    pass


if __name__ == '__main__':
    nn = GradientPair(2, 3)
    nn1 = GradientPair(12, 13)
    mm = nn / nn1
    nn2 = GradientPair(2, 3)
    nn /= nn2
    print(0)


