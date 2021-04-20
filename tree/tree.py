import numpy as np
# tree/model.h
from utils.util import resize


class TreeModel:
    def __init__(self):
        self.param = self.Param()
        self.param.num_nodes = 1
        self.param.num_roots = 1
        self.param.num_deleted = 0
        self.nodes = [self.Node()]

        self.deleted_nodes = []
        self.stats = []
        self.leaf_vector = []

        self.node_stat = None
        # self.split_cond = None

    def alloc_node(self):
        if self.param.num_deleted != 0:
            nd = self.deleted_nodes.pop()
            self.param.num_deleted -= 1
            return nd

        nd = self.param.num_nodes
        self.param.num_nodes += 1
        resize(self.nodes, self.param.num_nodes, self.Node())
        resize(self.stats, self.param.num_nodes, RTreeNodeStat())
        resize(self.leaf_vector, self.param.num_nodes *
               self.param.size_leaf_vector)
        return nd

    def delete_node(self, nid):
        assert nid >= self.param.num_roots, "cannot delete root"
        self.deleted_nodes.append(nid)
        self.nodes[nid].set_parent(-1)
        self.param.num_deleted += 1

    def change_to_leaf(self, rid, value):
        mssg = "cannot delete a non terminal child"
        assert self.nodes[self.nodes[rid].cleft()].is_leaf(), mssg
        assert self.nodes[self.nodes[rid].cright()].is_leaf(), mssg
        self.deleted_nodes(self.nodes[rid].cleft())
        self.deleted_nodes(self.nodes[rid].cright())
        self.nodes[rid].set_leaf(value)

    def collapse_to_leaf(self, rid, value):
        if self.nodes[rid].is_leaf():
            return
        if not self.nodes[self.nodes[rid].cleft()].is_leaf():
            self.collapse_to_leaf(self.nodes[rid].cleft(), 0.0)
        if not self.nodes[self.nodes[rid].cright()].is_leaf():
            self.collapse_to_leaf(self.nodes[rid].cright(), 0.0)
        self.change_to_leaf(rid, value)

    def __getitem__(self, i):
        return self.nodes[i]

    def stat(self, i):
        return self.stats[i]

    def leafvec(self, i):
        if len(self.leaf_vector) == 0:
            return None
        return self.leaf_vector[i * self.param.size_leaf_vector]

    def init_model(self):
        n_node = self.param.num_roots
        self.param.num_nodes = n_node
        resize(self.nodes, n_node, self.Node())
        resize(self.stats, n_node, RTreeNodeStat())
        resize(self.leaf_vector, n_node * self.param.size_leaf_vector)
        for i in range(n_node):
            self.nodes[i].set_leaf(0.0)
            self.nodes[i].set_parent(-1)

    def add_child(self, nid):
        pleft = self.alloc_node()
        pright = self.alloc_node()
        self.nodes[nid].cleft_ = pleft
        self.nodes[nid].cright_ = pright
        self.nodes[self.nodes[nid].cleft()].set_parent(nid, True)
        self.nodes[self.nodes[nid].cright()].set_parent(nid, False)

    def add_right_child(self, nid):
        pright = self.alloc_node()
        self.nodes[nid].right = pright
        self.nodes[self.nodes[nid].right].set_parent(nid, False)

    def get_depth(self, nid, pass_rchild=False):
        depth = 0
        while not self.nodes[nid].is_root():
            if (not pass_rchild) or self.nodes[nid].is_left_child():
                depth += 1
            nid = self.nodes[nid].parent()
        return depth

    def max_depth(self, nid=None):
        if nid is not None:
            if self.nodes[nid].is_leaf():
                return 0
            node = self.nodes[nid]
            return np.max(self.max_depth(node.cleft()) + 1,
                          self.max_depth(node.cright()) + 1)
        else:
            maxd = 0
            for i in self.param.num_roots:
                maxd = np.max(maxd, self.max_depth(i))
            return maxd

    def num_extra_nodes(self):
        """
        number of extra nodes besides the root
        """
        param = self.param
        return param.num_nodes - param.num_roots - param.num_deleted

    class Node:
        def __init__(self):
            self.parent_ = None
            self.cleft_ = self.cright_ = None
            self.sindex_ = None
            self.info_ = {'leaf_value': None, 'split_cond': None}

        def cleft(self):
            return self.cleft_

        def cright(self):
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
            return self.cleft_ == -1

        def leaf_value(self):
            return self.info_['leaf_value']

        def split_cond(self):
            return self.info_['split_cond']

        def parent(self):
            return self.parent_ & ((1 << 31) - 1)

        def is_left_child(self):
            return (self.parent_ & (1 << 31)) != 0

        def is_root(self):
            return self.parent_ == -1

        def set_right_child(self, nid):
            self.cright_ = nid

        def set_split(self, split_ind, split_cond,
                      default_left=False):
            if default_left:
                split_ind |= (1 << 31)
            self.sindex_ = split_ind
            self.info_['split_cond'] = split_cond

        @property
        def defaultchild(self):
            if self.default_left():
                return self.cleft_
            else:
                return self.cright_

        def set_leaf(self, value, right=-1):
            self.info_['leaf_value'] = value
            self.cleft_ = -1
            self.cright_ = right

        def set_parent(self, pidx, is_left_child=True):
            if is_left_child:
                pidx |= (1 << 31)
            self.parent_ = pidx

        def set_left_child(self, nid):
            self.cleft_ = nid

        def mark_delete(self):
            self.sindex_ = np.iinfo(np.uint32)

        def __eq__(self, other):
            return (self.parent_ == other.parent_ and
                    self.cleft_ == other.cleft_ and
                    self.cright_ == other.cright_ and
                    self.sindex_ == other.sindex_ and
                    self.leaf_value == other.leaf_value)

    class Param:
        def __init__(self):
            self.max_depth = 0
            self.size_leaf_vector = 0
            self.num_roots = 1
            self.num_nodes = 0
            self.num_deleted = 0
            self.num_feature = None

        def set_param(self, name, val):
            if name == 'num_roots':
                self.num_roots = val
            elif name == 'num_feature':
                self.num_feature = val
            elif name == 'size_leaf_vector':
                self.size_leaf_vector = val


class RTreeNodeStat:
    def __init__(self):
        self.loss_chg = None
        self.sum_hess = None
        self.base_weight = None
        self.leaf_child_cnt = None

    def print(self, is_leaf):
        if is_leaf:
            print(f'gain = {self.loss_chg} ,cover = {self.sum_hess}')
        else:
            print(f'cover= {self.sum_hess}')


class RegTree(TreeModel):
    def __init__(self):
        super().__init__()

    def get_leaf_index(self, feat, root_id=0):
        pid = root_id
        while not self[pid].is_leaf():
            split_index = self[pid].split_index()
            fvalue = feat.fvalue(split_index)
            is_unknown = feat.is_missing(split_index)
            pid = self.get_next(pid, fvalue, is_unknown)
        return pid

    def get_next(self, pid, fvalue, is_unknown):
        split_value = self[pid].split_cond()
        if is_unknown:
            return self[pid].cdefault()
        else:
            if fvalue < split_value:
                return self[pid].cleft()
            else:
                return self[pid].cright()

    def predict(self, feat, root_id=0):
        pid = self.get_leaf_index(feat, root_id)
        return self[pid].leaf_value()

    class FVec:
        def __init__(self):
            # self.fvalue = None
            # self.flag = None
            self.data = []

        class Entry:
            fvalue = None
            flag = 0

        def init(self, size):
            self.data = []
            for i in range(size):
                self.data.append(self.Entry())
                setattr(self.data[-1], 'flag', -1)

        def fill(self, inst):
            for i in range(inst.length):
                self.data[inst[i].index].fvalue = inst[i].fvalue

        def drop(self, inst):
            for i in range(inst.length):
                self.data[inst[i].index].flag = -1

        def fvalue(self, i):
            return self.data[i].fvalue

        def is_missing(self, i):
            return self.data[i].flag == -1


