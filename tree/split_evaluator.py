import params
from params import *
from utils.util import resize


class TreeEvaluator:
    kRootParentId = (-1 & ((1 << 31) - 1))

    def __init__(self, p, n_features, device):
        self.device_ = device

        if len(p.monotone_constraints) == 0:
            self.monotone_ = [] * n_features
            self.has_constraint_ = False
            self.lower_bounds_ = None
            self.upper_bounds_ = None

        else:
            self.monotone_ = p.monotone_constraints
            resize(self.monotone_, n_features, 0)
            infinity = np.finfo(np.float32())
            self.lower_bounds_ = [infinity.min] * p.max_nodes()
            self.upper_bounds_ = [infinity.max] * p.max_nodes()
            self.has_constraint_ = True

    def get_evaluator(self):
        constraints = self.monotone_
        return SplitEvaluator(constraints, self.lower_bounds_,
                              self.upper_bounds_, self.has_constraint_)

    def add_split(self, nodeid, leftid, rightid, f, left_weight, right_weight):
        if not self.has_constraint_:
            return

        self.lower_bounds_[leftid] = self.lower_bounds_[nodeid]
        self.upper_bounds_[leftid] = self.upper_bounds_[nodeid]

        self.lower_bounds_[rightid] = self.lower_bounds_[nodeid]
        self.upper_bounds_[rightid] = self.upper_bounds_[nodeid]
        c = self.monotone_[f]
        mid = (left_weight + right_weight) / 2

        if c < 0:
            self.lower_bounds_[leftid] = mid
            self.upper_bounds_[rightid] = mid
        elif c > 0:
            self.upper_bounds_[leftid] = mid
            self.lower_bounds_[rightid] = mid


class SplitEvaluator:
    def __init__(self, constraints=None, lower_bounds=None, upper_bounds=None,
                 has_constraint=False):
        self.constraints = constraints if constraints is not None else [None]
        self.has_constraint = has_constraint
        self.lower = lower_bounds
        self.upper = upper_bounds

    def calc_split_gain(self, param, nidx, fidx, left, right):
        constraint = 0
        negative_infinity = np.iinfo(int).min
        wleft = self.calc_weight(nidx, param, left)
        wright = self.calc_weight(nidx, param, right)

        gain_l = self.calc_gain_given_weight(param, left, wleft)
        gain_r = self.calc_gain_given_weight(param, left, wright)
        gain = gain_l + gain_r

        if constraint == 0:
            return gain
        elif constraint > 0:
            return gain if wleft <= wright else negative_infinity
        else:
            return gain if wleft >= wright else negative_infinity

    def calc_weight(self, nodeid, param, stats):
        w = params.calc_weight(param, stats)
        if not self.has_constraint:
            return w
        if nodeid == TreeEvaluator.kRootParentId:
            return w
        elif w < self.lower[nodeid]:
            return self.lower[nodeid]
        elif w > self.upper[nodeid]:
            return self.upper[nodeid]
        else:
            return w

    def calc_gain_given_weight(self, p, stats, w):
        if stats.get_hess() <= 0:
            return 0.0
        if p.max_delta_step == 0.0 and not self.has_constraint:
            first = thresholdL1(stats.sum_grad, p.reg_alpha)
            return first * first / (stats.sum_hess + p.reg_lambda)
        return calc_gain_given_weight(p, stats.sum_grad, stats.sum_hess, w)

    def calc_gain(self, nid, p, stats):
        w = self.calc_weight(nid, p, stats)
        return self.calc_gain_given_weight(p, stats, w)
