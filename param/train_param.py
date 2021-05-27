import numpy as np
from enum import Enum
from param.parameters import XGBoostParameter


class TreeGrowPolicy:
    kDepthWise = 0
    kLossGuide = 1


class SamplingMethod:
    kUniform = 0
    kGradientBased = 1


class Direction:
    learn = 0
    left = 1
    right = 2


class TrainParam(XGBoostParameter):
    """
    param_.h
    """

    def __init__(self):
        super().__init__()
        self.learning_rate = 0.3
        self.min_split_loss = 0.0
        self.max_depth = 6
        self.max_leaves = 0
        self.max_bin = 256
        self.grow_policy = TreeGrowPolicy.kDepthWise
        self.min_child_weight = 1.0
        self.reg_lambda = 1.0
        self.reg_alpha = 0.0
        self.default_direction = Direction.learn
        self.max_delta_step = 0.0
        self.subsample = 1.0
        self.sampling_method = SamplingMethod.kUniform
        self.colsample_bynode = 1.0
        self.colsample_bylevel = 1.0
        self.colsample_bytree = 1.0

        self.sketch_eps = 0.03
        self.sketch_ratio = 2.0
        self.cache_opt = True
        self.refresh_leaf = True

        # self.opt_dense_col = 1.0
        self.monotone_constraints = []

        self.interaction_constraints = ''
        self.split_evaluator = "elastic_net,monotonic"

        self.sparse_threshold = 0.2
        self.enable_feature_grouping = 0
        self.max_conflict_rate = 0
        self.max_search_group = 100
        self.lamdba = self.reg_lambda
        self.alpha = self.reg_alpha
        self.gamma = self.min_split_loss
        self.eta = self.learning_rate

    def need_prune(self, loss_chg, depth):
        return loss_chg < self.min_split_loss or (
                self.max_depth != 0 and depth > self.max_depth)

    def max_sketch_size(self):
        ret = self.sketch_ratio / self.sketch_eps
        assert ret > 0
        return ret

    def max_nodes(self):
        msg_error = "Max leaves and max depth cannot both be unconstrained."
        assert self.max_depth != 0 or self.max_leaves != 0, msg_error
        if self.max_leaves > 0:
            n_nodes = self.max_leaves * 2 - 1
        else:
            assert self.max_depth <= 31
            n_nodes = (1 << (self.max_depth + 1)) - 1
        assert n_nodes != 0
        return n_nodes

    # def calc_gain_cost(self, p, sum_grad, sum_hess, test_grad, test_hess):
    #     w = calc_weight(p, sum_grad, sum_hess)
    #     ret = test_grad * w + 0.5 * (test_hess + self.reg_lambda) * (w ** 2)
    #     if self.reg_alpha == 0:
    #         return -1 * ret
    #     else:
    #         return -2 * (ret + self.reg_alpha * np.abs(w))

    def cannot_split(self, sum_hess, depth):
        return sum_hess < self.min_child_weight * 2


class ColMakerTrainParam(XGBoostParameter):
    def __init__(self):
        super().__init__()
        self.opt_dense_col = 1.0

    def need_forward_search(self, default_direction, col_density, indicator):
        return (default_direction == 2) or (
                (default_direction == 0) and (
                 col_density < self.opt_dense_col and not indicator))

    def need_backward_search(self, default_direction):
        return default_direction != 2


class NodeEntry:
    def __init__(self, grad=0, hess=0, root_gain=0.0, weight=0.0, best=None):
        self.sum_grad = grad
        self.sum_hess = hess
        self.root_gain = root_gain
        self.weight = weight
        self.best = best


class SplitEntryContainer:
    def __init__(self, loss_chg=0, sindex=0, split_value=0.0, left_sum=0,
                 right_sum=0):
        self.loss_chg = loss_chg
        self.sindex = sindex
        self.split_value = split_value
        self.left_sum = left_sum
        self.right_sum = right_sum

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






#
# def set_param(self, name, value):
#     if name == 'gamma':
#         self.min_split_loss = value
#     elif name == 'eta':
#         self.learning_rate = value
#     elif name == 'lambda':
#         self.reg_lambda = value
#     elif name == 'learning_rate':
#         self.learning_rate = value
#     elif name == 'min_child_weight':
#         self.min_child_weight = value
#     elif name == 'min_split_loss':
#         self.min_split_loss = value
#     elif name == 'reg_lambda':
#         self.reg_lambda = value
#     elif name == 'reg_alpha':
#         self.reg_alpha = value
#     elif name == 'subsample':
#         self.subsample = value
#     elif name == 'colsample_bylevel':
#         self.colsample_bylevel = value
#     elif name == 'colsample_bytree':
#         self.colsample_bytree = value
#     elif name == 'opt_dense_col':
#         self.opt_dense_col = value
#     elif name == 'size_leaf_vector':
#         self.size_leaf_vector = value
#     elif name == 'max_depth':
#         self.max_depth = value
#     elif name == 'nthread':
#         self.nthread = value
#     elif name == 'default_direction':
#         if value == 'learn':
#             self.default_direction = 0
#         elif value == 'left':
#             self.default_direction = 1
#         elif value == 'right':
#             self.default_direction = 2