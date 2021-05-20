import numpy as np


kDeletedNodeMarker = np.uint32(-1)
kInvalidNodeId = -1
kRoot = 0

kNumerical = 0
kCategorical = 1


class TreeParam:
    def __init__(self):
        self.max_depth = 0
        self.size_leaf_vector = 0
        self.num_roots = 1
        self.num_nodes = 1
        self.num_deleted = 0
        self.num_feature = None

    def set_param(self, name, val):
        if name == 'num_roots':
            self.num_roots = val
        elif name == 'num_feature':
            self.num_feature = val
        elif name == 'size_leaf_vector':
            self.size_leaf_vector = val

    def __eq__(self, b):
        return (self.num_nodes == b.num_nodes) and (
                self.num_deleted == b.num_deleted) and (
                       self.num_feature == b.num_feature) and (
                       self.size_leaf_vector == b.size_leaf_vector)


class Node:
    def __init__(self, cleft=-1, cright=-1, parent=-1,
                 split_ind=0, split_cond=None,
                 default_left=True):
        self.parent_ = parent
        self.cleft_ = cleft
        self.cright_ = cright
        self.info_ = self.Info()
        self.set_parent(self.parent_)
        self.set_split(split_ind, split_cond, default_left)

        self.sindex_ = 0

    def left_child(self):
        return self.cleft_

    def right_child(self):
        return self.cright_

    def default_child(self):
        if self.default_left():
            return self.cleft_
        else:
            return self.cright_

    def cdefault(self):
        if self.default_left():
            return self.cleft_
        else:
            return self.cright_

    def split_index(self):
        return self.sindex_ & ((1 << 31) - 1)

    def default_left(self):
        return (self.sindex_ >> 31) != 0

    def is_leaf(self):
        return self.cleft_ == kInvalidNodeId

    def leaf_value(self):
        return self.info_.leaf_value

    def split_cond(self):
        return self.info_.split_cond

    def parent(self):
        return self.parent_ & ((1 << 31) - 1)

    def is_left_child(self):
        return (self.parent_ & (1 << 31)) != 0

    def is_deleted(self):
        return self.sindex_ == kDeletedNodeMarker

    def is_root(self):
        return self.parent_ == kInvalidNodeId

    def set_left_child(self, nid):
        self.cleft_ = nid

    def set_right_child(self, nid):
        self.cright_ = nid

    def set_split(self, split_ind, split_cond,
                  default_left=False):
        if default_left:
            split_ind |= (1 << 31)
        self.sindex_ = split_ind
        self.info_.split_cond = split_cond

    def set_leaf(self, value, right=-1):
        self.info_.leaf_value = value
        self.cleft_ = kInvalidNodeId
        self.cright_ = right

    def mark_delete(self):
        self.sindex_ = kDeletedNodeMarker

    def reuse(self):
        self.sindex_ = 0

    def set_parent(self, pidx, is_left_child=True):
        if is_left_child:
            pidx |= (1 << 31)
        self.parent_ = pidx

    def __eq__(self, other):
        return (self.parent_ == other.parent_ and
                self.cleft_ == other.cleft_ and
                self.cright_ == other.cright_ and
                self.sindex_ == other.sindex_ and
                self.info_.leaf_value == other.info_.leaf_value)

    @property
    def defaultchild(self):
        if self.default_left():
            return self.cleft_
        else:
            return self.cright_

    class Info:
        def __init__(self, leaf_value=None, split_cond=None):
            self.leaf_value = leaf_value
            self.split_cond = split_cond

    def cleft(self):
        return self.cleft_

    def cright(self):
        return self.cright_


class RTreeNodeStat:
    def __init__(self, loss_chg=None, sum_hess=None, weight=None):
        self.loss_chg = loss_chg
        self.sum_hess = sum_hess
        self.base_weight = weight
        self.leaf_child_cnt = 0

    def print(self, is_leaf):
        if is_leaf:
            print(f'gain = {self.loss_chg} ,cover = {self.sum_hess}')
        else:
            print(f'cover= {self.sum_hess}')

    def __eq__(self, b):
        return self.loss_chg == b.loss_chg and self.sum_hess == b.sum_hess and \
               self.base_weight == b.base_weight and \
               self.leaf_child_cnt == b.leaf_child_cnt


class Segment:
    def __init__(self, beg=0, size=0):
        self.beg = beg
        self.size = size