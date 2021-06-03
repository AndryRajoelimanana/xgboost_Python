import numpy as np
from enum import Enum
import sys


def thresholdL1(w, alpha):
    if w > alpha:
        return w - alpha
    if w < - alpha:
        return w + alpha
    return 0.0


def calc_gain_given_weight(p, sum_grad, sum_hess, w):
    if sum_hess <= 0:
        return 0
    return -(2.0 * sum_grad * w + (sum_hess + p.reg_lambda) * (w * w))


def calc_weight(p, sum_grad_in, sum_hess=0):
    if isinstance(sum_grad_in, GradientPair):
        sum_grad = sum_grad_in.get_grad()
        sum_hess = sum_grad_in.get_hess()
    else:
        sum_grad = sum_grad_in
    if sum_hess < p.min_child_weight or sum_hess <= 0:
        return 0
    dw = - thresholdL1(sum_grad, p.reg_alpha) / (sum_hess + p.reg_lambda)
    if p.max_delta_step != 0.0 and np.abs(dw) > p.max_delta_step:
        dw = np.copysign(p.max_delta_step, dw)
    return dw


def calc_gain(p, sum_grad_in, sum_hess=0):
    if isinstance(sum_grad_in, GradStats):
        sum_grad = sum_grad_in.get_grad()
        sum_hess = sum_grad_in.get_hess()
    else:
        sum_grad = sum_grad_in
    if sum_hess < p.min_child_weight:
        return 0
    if p.max_delta_step == 0.0:
        if p.reg_alpha == 0.0:
            return (sum_grad * sum_grad) / (sum_hess + p.reg_lambda)
        else:
            dw = thresholdL1(sum_grad, p.reg_alpha)
            return (dw * dw) / (sum_hess + p.reg_lambda)
    else:
        w = calc_weight(p, sum_grad, sum_hess)
        ret = calc_gain_given_weight(p, sum_grad, sum_hess, w)
        if p.reg_alpha == 0:
            return ret
        else:
            return ret + p.reg_alpha * np.abs(w)


class TrainParam:
    """
    param_.h
    """

    def __init__(self):
        self.learning_rate = 0.3
        self.min_split_loss = 0.0
        self.max_depth = 6
        self.max_leaves = 0
        self.max_bin = 256
        self.grow_policy = self.TreeGrowPolicy.kDepthWise
        self.min_child_weight = 1.0
        self.reg_lambda = 1.0
        self.reg_alpha = 0.0
        self.default_direction = self.Direction.learn
        self.max_delta_step = 0.0
        self.subsample = 1.0
        self.sampling_method = self.SamplingMethod.kUniform
        self.colsample_bynode = 1.0
        self.colsample_bylevel = 1.0
        self.colsample_bytree = 1.0

        self.sketch_eps = 0.03
        self.sketch_ratio = 2.0
        self.cache_opt = True
        self.refresh_leaf = True

        self.opt_dense_col = 1.0
        self.monotone_constraints = []

        self.interaction_constraints = ''
        self.split_evaluator = "elastic_net,monotonic"

        self.sparse_threshold = 0.2
        self.enable_feature_grouping = 0
        self.max_conflict_rate = 0
        self.max_search_group = 100

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

    class TreeGrowPolicy(Enum):
        kDepthWise = 0
        kLossGuide = 1

    class SamplingMethod(Enum):
        kUniform = 0
        kGradientBased = 1

    class Direction(Enum):
        learn = 0
        left = 1
        right = 2

    def need_prune(self, loss_chg, depth):
        return loss_chg < self.min_split_loss or (
                self.max_depth != 0 and depth > self.max_depth)

    def max_sketch_size(self):
        ret = self.sketch_ratio / self.sketch_eps
        assert ret > 0
        return ret

    def max_nodes(self):
        assert self.max_depth != 0 or self.max_leaves != 0, "Max leaves and max depth cannot both be unconstrained."
        if self.max_leaves > 0:
            n_nodes = self.max_leaves * 2 - 1
        else:
            assert self.max_depth <= 31
            n_nodes = (1 << (self.max_depth + 1)) - 1
        assert n_nodes != 0
        return n_nodes

    def calc_gain_cost(self, p, sum_grad, sum_hess, test_grad, test_hess):
        w = calc_weight(p, sum_grad, sum_hess)
        ret = test_grad * w + 0.5 * (test_hess + self.reg_lambda) * (w ** 2)
        if self.reg_alpha == 0:
            return -1 * ret
        else:
            return -2 * (ret + self.reg_alpha * np.abs(w))

    def cannot_split(self, sum_hess, depth):
        return sum_hess < self.min_child_weight * 2


class GradStats:
    """
    param_.h
    """

    def __init__(self, sum_grad_i=0.0, sum_hess=0.0):
        if isinstance(sum_grad_i, GradientPair):
            sum_grad = sum_grad_i.get_grad()
            sum_hess = sum_grad_i.get_hess()
        else:
            sum_grad = sum_grad_i
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

    def add(self, grad_i, hess=None):
        if isinstance(grad_i, GradientPair):
            grad = grad_i.get_grad()
            hess = grad_i.get_hess()
        elif isinstance(grad_i, GradStats):
            grad = grad_i.sum_grad
            hess = grad_i.sum_hess
        else:
            grad = grad_i
        self.sum_grad += grad
        self.sum_hess += hess

    def __add__(self, other):
        self.sum_grad += other.sum_grad
        self.sum_hess += other.sum_hess
        return self

    def __sub__(self, other):
        self.sum_grad -= other.sum_grad
        self.sum_hess -= other.sum_hess
        return self

    @staticmethod
    def reduce(a, b):
        return a + b

    def set_substract(self, a, b):
        self.sum_grad = a.sum_grad - b.sum_grad
        self.sum_hess = a.sum_hess - b.sum_hess

    def empty(self):
        return self.sum_hess == 0


class SplitEntryContainer:
    def __init__(self, loss_chg=0, sindex=0, split_value=0.0):
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
        elif self.split_index() <= split_index:
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

    def update(self, e, split_index=0, new_split_value=0,
               default_left=True, left_sum=0, right_sum=0):
        if isinstance(e, SplitEntryContainer):
            new_loss_chg = e.loss_chg
            split_index = e.sindex
            new_split_value = e.split_value
            left_sum = e.left_sum
            right_sum = e.right_sum
        else:
            new_loss_chg = e
            if default_left:
                split_index |= (1 << 31)

        if self.need_replace(new_loss_chg, split_index):
            self.loss_chg = new_loss_chg
            self.sindex = split_index
            self.split_value = new_split_value
            self.left_sum = left_sum
            self.right_sum = right_sum
            return True
        else:
            return False

    @staticmethod
    def reduce(dst, src):
        return dst.update_e(src)


SplitEntry = SplitEntryContainer


class GradientPair:
    def __init__(self, grad_i=0, hess=0):
        if isinstance(grad_i, GradientPair):
            grad = grad_i.get_grad()
            hess = grad_i.get_hess()
        else:
            grad = grad_i
        self.grad_ = grad
        self.hess_ = hess

    def set_grad(self, g):
        self.grad_ = g

    def set_hess(self, h):
        self.hess_ = h

    def get_grad(self):
        return self.grad_

    def get_hess(self):
        return self.hess_

    def add(self, grad, hess):
        self.grad_ += grad
        self.hess_ += hess

    @staticmethod
    def reduce(a, b):
        g = GradientPair(a.grad_, a.hess_)
        g.grad_ += b.grad_
        g.hess_ += b.hess_
        return g

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


def ParseInteractionConstraint(constraint_str, out):
    pass


if __name__ == '__main__':
    nn = GradientPair(2, 3)
    nn1 = GradientPair(12, 13)
    mm = nn / nn1
    nn2 = GradientPair(2, 3)
    nn /= nn2
    print(0)
    nnn = GradStats()
    vv = sys.getsizeof(nnn)
    print(0)
