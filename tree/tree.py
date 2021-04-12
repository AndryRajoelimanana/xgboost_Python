import numpy as np
# tree/model.h


class TreeModel:
    def __init__(self):
        self.param = self.Param()
        self.nodes = [self.Node()]
        self.deleted_nodes = None

        self.param.num_nodes = 1
        self.param.num_roots = 1
        self.param.num_deleted = 0

        self.node_stat = None
        self.split_cond = None
        self.stats = []
        self.leaf_vector = None

    def alloc_node(self):
        if self.param.num_deleted != 0:
            nd = self.deleted_nodes.pop()
            self.param.num_deleted -= 1
            return nd
        self.param.num_nodes += 1
        nd = self.param.num_nodes
        self.nodes.append(self.Node())
        self.stats.append(RTreeNodeStat())
        self.leaf_vector.extend([0] * nd * self.param.size_leaf_vector)
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
        if self.leaf_vector() == 0:
            return None
        return self.leaf_vector[i * self.param.size_leaf_vector]

    def init_model(self):
        n_node = self.param.num_roots
        self.param.num_nodes = n_node
        self.nodes = [self.Node()]*n_node
        self.stats = [RTreeNodeStat()]*n_node
        self.leaf_vector = [0.0]*n_node * self.param.size_leaf_vector
        for i in range(n_node):
            self.nodes[i].set_leaf(0.0)
            self.nodes[i].set_parent(-1)

    def add_child(self, nid):
        pleft = self.alloc_node()
        pright = self.alloc_node()
        self.nodes[nid].cleft_ = pleft
        self.nodes[nid].cright_ = pright
        self.nodes[pleft].set_parent(nid, True)
        self.nodes[pright].set_parent(nid, False)

    def add_right_child(self, nid):
        pright = self.alloc_node()
        self.nodes[nid].right = pright
        self.nodes[nid].set_parent(nid, False)

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
            self.sindex_ = self.split_cond = None
            self.right = None
            self.leaf_value = None
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
            self.num_roots = self.num_nodes = self.num_deleted = None
            self.num_feature = None

        def set_param(self, name, val):
            setattr(self, name, val)


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
        super(RegTree, self).__init__()

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
            self.fvalue = None
            self.flag = None
            self.data = None

        class Entry:
            fvalue = None
            flag = 0

        def init(self, size):
            e = self.Entry()
            e.flag = -1
            self.data = [e.flag]*size

        def fill(self, inst):
            for i in range(len(inst)):
                self.data[inst[i].index].fvalue = inst[i].fvalue

        def drop(self, inst):
            for i in range(len(inst)):
                self.data[inst[i].index].flag = -1

        def fvalue(self, i):
            return self.data[i].fvalue

        def is_missing(self, i):
            return self.data[i].flag == -1
