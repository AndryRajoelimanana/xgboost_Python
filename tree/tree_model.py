import numpy as np
from tree.tree_model_helper import *
from utils.util import resize
from param.tree_model_param import TreeParam


class TreeModel:
    def __init__(self):
        self.param = TreeParam()
        self.nodes_ = [Node()]

        self.deleted_nodes_ = []
        self.stats_ = [RTreeNodeStat()]

        self.node_stat = None

    def __getitem__(self, i):
        return self.nodes_[i]

    def stat(self, i):
        return self.stats_[i]

    def init_model(self):
        n_node = self.param.num_roots
        self.param.num_nodes = n_node
        for i in range(n_node):
            self.nodes_[i].set_leaf(0.0)
            self.nodes_[i].set_parent(-1)

    def add_child(self, nid):
        pleft = self.alloc_node()
        pright = self.alloc_node()
        self.nodes_[nid].cleft_ = pleft
        self.nodes_[nid].cright_ = pright
        self.nodes_[self.nodes_[nid].cleft_].set_parent(nid, True)
        self.nodes_[self.nodes_[nid].cright_].set_parent(nid, False)

    def add_right_child(self, nid):
        pright = self.alloc_node()
        self.nodes_[nid].cright_ = pright
        self.nodes_[self.nodes_[nid].cright_].set_parent(nid, False)

    def alloc_node(self):
        if self.param.num_deleted != 0:
            nd = self.deleted_nodes_.pop()
            self.nodes_[nd].reuse()
            self.param.num_deleted -= 1
            return nd

        nd = self.param.num_nodes
        self.param.num_nodes += 1
        self.nodes_.append(Node())
        self.stats_.append(RTreeNodeStat())
        return nd

        # resize(self.nodes_, self.param.num_nodes, Node())
        # resize(self.stats_, self.param.num_nodes, RTreeNodeStat())
        # resize(self.split_types_, self.param.num_nodes,
        # FeatureType.kNumerical)
        # resize(self.split_categories_segments_, self.param.num_nodes,
        #       Segment())
        # resize(self.leaf_vector, self.param_.num_nodes *
        #       self.param_.size_leaf_vector)
        # return nd


class RegTree(TreeModel):

    def __init__(self):
        super().__init__()
        self.node_mean_values_ = []
        self.split_categories_ = []

        self.split_types_ = [0]
        self.split_categories_segments_ = [Segment()]
        for i in range(self.param.num_nodes):
            self.nodes_[i].set_leaf(0)
            self.nodes_[i].set_parent(kInvalidNodeId)

    def _delete_node(self, nid):
        assert nid >= 1, "cannot delete root"
        pid = self[nid].parent()
        if nid == self[pid].left_child():
            self[pid].set_left_child(kInvalidNodeId)
        else:
            self[pid].set_right_child(kInvalidNodeId)
        self.deleted_nodes_.append(nid)
        self.nodes_[nid].mark_delete()
        self.param.num_deleted += 1

    def change_to_leaf(self, rid, value):
        mssg = "cannot delete a non terminal child"
        assert self.nodes_[self.nodes_[rid].left_child()].is_leaf(), mssg
        assert self.nodes_[self.nodes_[rid].right_child()].is_leaf(), mssg
        self._delete_node(self.nodes_[rid].left_child())
        self._delete_node(self.nodes_[rid].right_child())
        self.nodes_[rid].set_leaf(value)

    def collapse_to_leaf(self, rid, value):
        if self.nodes_[rid].is_leaf():
            return
        if not self.nodes_[self.nodes_[rid].left_child()].is_leaf():
            self.collapse_to_leaf(self.nodes_[rid].left_child(), 0.0)
        if not self.nodes_[self.nodes_[rid].right_child()].is_leaf():
            self.collapse_to_leaf(self.nodes_[rid].right_child(), 0.0)
        self.change_to_leaf(rid, value)

    def get_nodes(self):
        return self.nodes_

    def get_stats(self):
        return self.stats_

    def stat(self, i):
        return self.stats_[i]

    def walk_tree(self, func):
        nodes = [kRoot]
        while not len(nodes) == 0:
            nidx = nodes.pop()
            if not func(nidx):
                return
            left = self[nidx].left_child()
            right = self[nidx].right_child()
            if left != kInvalidNodeId:
                nodes.append(left)
            if right != kInvalidNodeId:
                nodes.append(right)

    def expandnode(self, nid, split_index, split_value, default_left,
                   base_weight, left_leaf_weight, right_leaf_weight,
                   loss_change, sum_hess, left_sum, right_sum,
                   leaf_right_child=kInvalidNodeId):
        pleft = self._alloc_node()
        pright = self._alloc_node()
        node = self.nodes_[nid]

        assert node.is_leaf()
        self.nodes_[nid].set_left_child(pleft)
        self.nodes_[nid].set_right_child(pright)
        self.nodes_[pleft].set_parent(nid, True)
        self.nodes_[pright].set_parent(nid, False)

        self.nodes_[nid].set_split(split_index, split_value, default_left)

        self.nodes_[pleft].set_leaf(left_leaf_weight, leaf_right_child)
        self.nodes_[pright].set_leaf(right_leaf_weight, leaf_right_child)

        self.stats_[nid] = RTreeNodeStat(loss_change, sum_hess, base_weight)
        self.stats_[pleft] = RTreeNodeStat(0.0, left_sum, left_leaf_weight)
        self.stats_[pright] = RTreeNodeStat(0.0, right_sum, right_leaf_weight)
        self.split_types_[nid] = kNumerical

    def expand_categorical(self, nid, split_index, split_cat, default_left,
                           base_weight, left_leaf_weight, right_leaf_weight,
                           loss_change, sum_hess, left_sum, right_sum):
        self.expandnode(nid, split_index, np.nan, default_left,
                        base_weight, left_leaf_weight, right_leaf_weight,
                        loss_change, sum_hess, left_sum, right_sum)
        orig_size = len(self.split_categories_)
        self.split_categories_ += split_cat
        self.split_types_[nid] = kCategorical
        self.split_categories_segments_[nid].beg = orig_size
        self.split_categories_segments_[nid].size = len(split_cat)

    def fill_node_mean_values(self):
        num_nodes = self.param.num_nodes
        if len(self.node_mean_values_) == num_nodes:
            return

        resize(self.node_mean_values_, num_nodes)
        self.fill_node_mean_value(0)

    def fill_node_mean_value(self, nid):
        node = self[nid]
        if node.is_leaf():
            result = node.leaf_value()
        else:
            result = self.fill_node_mean_value(node.left_child()) * self.stat(
                node.left_child()).sum_hess
            result += self.fill_node_mean_value(node.right_child()) * self.stat(
                node.right_child()).sum_hess
            result /= self.stat(nid).sum_hess
        self.node_mean_values_[nid] = result
        return result

    def get_depth(self, nid):
        depth = 0
        while not self.nodes_[nid].is_root():
            depth += 1
            nid = self.nodes_[nid].parent()
        return depth

    def max_depth(self, nid=0):
        if self.nodes_[nid].is_leaf():
            return 0
        node = self.nodes_[nid]
        return np.maximum(self.max_depth(node.cleft()) + 1,
                          self.max_depth(node.cright()) + 1)

    def num_extra_nodes(self):
        """
        number of extra nodes_ besides the root
        """
        return self.param.num_nodes - 1 - self.param.num_deleted

    def get_num_leaves(self):
        self._nleaves = 0

        def spl(nidx):
            if self[nidx].is_leaf():
                self._nleaves += 1
            return True

        self.walk_tree(spl)
        return self._nleaves

    def get_num_split_nodes(self):
        self._nsplits = 0

        def spl(nidx):
            if not self[nidx].is_leaf():
                self._nsplits += 1
            return True

        self.walk_tree(spl)
        return self._nsplits

    def get_leaf_index(self, feat, has_missing=False):
        nid = 0
        while not self[nid].is_leaf():
            split_index = self[nid].split_index()
            # fvalue = feat.get_fvalue(split_index)
            fvalue = feat[:, split_index]
            is_unknown = has_missing and feat.is_missing(split_index)
            nid = self.get_next(nid, fvalue, has_missing, is_unknown)
        return nid

    def get_next(self, pid, fvalue, is_unknown, has_missing=False):
        # is_unknown = has_missing and is_unknown
        if has_missing:
            if is_unknown:
                return self[pid].defaultchild()
            else:
                if fvalue < self[pid].split_cond():
                    return self[pid].left_child()
                else:
                    return self[pid].right_child()
        else:
            if fvalue < self[pid].split_cond():
                self[pid].left_child()
            else:
                self[pid].right_child()

    def dump_model(self, fmap, with_stats, formats):
        pass

    def node_split_type(self, nidx):
        return self.split_types_[nidx]

    def get_split_types(self):
        return self.split_types_

    def get_split_categories(self):
        return self.split_categories_

    def get_split_categories_ptr(self):
        return self.split_categories_segments_

    def _load_categorical_split(self, inj):
        pass

    def _save_categorical_split(self, inj):
        pass

    def _alloc_node(self):
        if self.param.num_deleted != 0:
            nd = self.deleted_nodes_.pop()
            self.nodes_[nd].reuse()
            self.param.num_deleted -= 1
            return nd

        nd = self.param.num_nodes
        self.param.num_nodes += 1
        resize(self.nodes_, self.param.num_nodes, Node())
        resize(self.stats_, self.param.num_nodes, RTreeNodeStat())
        resize(self.split_types_, self.param.num_nodes, kNumerical)
        resize(self.split_categories_segments_, self.param.num_nodes,
               Segment())
        # resize(self.leaf_vector, self.param_.num_nodes *
        #       self.param_.size_leaf_vector)
        return nd

    def predict(self, feat, root_id=0):
        pid = self.get_leaf_index(feat)
        return self[pid].leaf_value()

    def __getitem__(self, nid):
        return self.nodes_[nid]

    def equal(self, other):
        self._ret = True

        def spl(nidx):
            if not self.nodes_[nidx] == other.nodes_[nidx]:
                self._ret = False
                return False
            return True

        self.walk_tree(spl)
        return self._ret

    def __eq__(self, b):
        return self.nodes_ == b.nodes_ and self.stats_ == b.stats_ and \
               self.deleted_nodes_ == b.deleted_nodes_ and self.param == b.param_
