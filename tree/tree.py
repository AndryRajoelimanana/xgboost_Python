import numpy as np
# tree/model.h
from utils.util import resize
# from  import FeatureType
from data_i.data import FeatureType


class TreeModel:
    def __init__(self):
        self.param = self.Param()
        self.param.num_nodes = 1
        self.param.num_roots = 1
        self.param.num_deleted = 0
        self.nodes_ = [self.Node()]

        self.deleted_nodes_ = []
        self.stats_ = []
        self.leaf_vector = []

        self.node_stat = None
        # self.split_cond = None

    def __getitem__(self, i):
        return self.nodes_[i]

    def stat(self, i):
        return self.stats_[i]

    def leafvec(self, i):
        if len(self.leaf_vector) == 0:
            return None
        return self.leaf_vector[i * self.param.size_leaf_vector]

    def init_model(self):
        n_node = self.param.num_roots
        self.param.num_nodes = n_node
        resize(self.nodes_, n_node, self.Node())
        resize(self.stats_, n_node, RTreeNodeStat())
        resize(self.leaf_vector, n_node * self.param.size_leaf_vector)
        for i in range(n_node):
            self.nodes_[i].set_leaf(0.0)
            self.nodes_[i].set_parent(-1)

    def add_child(self, nid):
        pleft = self.alloc_node()
        pright = self.alloc_node()
        self.nodes_[nid].cleft_ = pleft
        self.nodes_[nid].cright_ = pright
        self.nodes_[self.nodes_[nid].cleft()].set_parent(nid, True)
        self.nodes_[self.nodes_[nid].cright()].set_parent(nid, False)

    def add_right_child(self, nid):
        pright = self.alloc_node()
        self.nodes_[nid].right = pright
        self.nodes_[self.nodes_[nid].right].set_parent(nid, False)

    def get_depth(self, nid, pass_rchild=False):
        depth = 0
        while not self.nodes_[nid].is_root():
            if (not pass_rchild) or self.nodes_[nid].is_left_child():
                depth += 1
            nid = self.nodes_[nid].parent()
        return depth

    def max_depth(self, nid=None):
        if nid is not None:
            if self.nodes_[nid].is_leaf():
                return 0
            node = self.nodes_[nid]
            return np.max(self.max_depth(node.cleft()) + 1,
                          self.max_depth(node.cright()) + 1)
        else:
            maxd = 0
            for i in self.param.num_roots:
                maxd = np.max(maxd, self.max_depth(i))
            return maxd

    def num_extra_nodes(self):
        """
        number of extra nodes_ besides the root
        """
        return self.param.num_nodes - 1 - self.param.num_deleted


class Segment:
    def __init__(self, beg=0, size=0):
        self.beg = beg
        self.size = size


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


class RegTree(TreeModel):
    kInvalidNodeId = -1
    kDeletedNodeMarker = np.iinfo(np.uint32).max
    kRoot = 0

    def __init__(self):
        super().__init__()
        self.param = self.Param()
        self.param.num_nodes = 1
        self.param.num_roots = 1
        self.param.num_deleted = 0
        self.nodes_ = []
        resize(self.nodes_, self.param.num_nodes, self.Node())
        self.node_mean_values_ = []
        self.split_categories_ = []

        self.stats_ = []
        resize(self.stats_, self.param.num_nodes, RTreeNodeStat())
        self.split_types_ = []
        resize(self.split_types_, self.param.num_nodes, FeatureType.kNumerical)
        self.split_categories_segments_ = []
        resize(self.split_categories_segments_, self.param.num_nodes,
               Segment())
        for i in range(self.param.num_nodes):
            self.nodes_[i].set_leaf(0)
            self.nodes_[i].set_parent(RegTree.kInvalidNodeId)

    def alloc_node(self):
        if self.param.num_deleted != 0:
            nd = self.deleted_nodes_.pop()
            self.nodes_[nd].reuse()
            self.param.num_deleted -= 1
            return nd

        nd = self.param.num_nodes
        self.param.num_nodes += 1
        resize(self.nodes_, self.param.num_nodes, self.Node())
        resize(self.stats_, self.param.num_nodes, RTreeNodeStat())
        resize(self.split_types_, self.param.num_nodes, FeatureType.kNumerical)
        resize(self.split_categories_segments_, self.param.num_nodes,
               Segment())
        # resize(self.leaf_vector, self.param.num_nodes *
        #       self.param.size_leaf_vector)
        return nd

    def delete_node(self, nid):
        assert nid >= 1, "cannot delete root"
        pid = self[nid].parent()
        if nid == self[pid].left_child():
            self[pid].set_left_child(RegTree.kInvalidNodeId)
        else:
            self[pid].set_right_child(RegTree.kInvalidNodeId)
        self.deleted_nodes_.append(nid)
        self.nodes_[nid].mark_delete()
        self.param.num_deleted += 1

    class Node:
        def __init__(self, cleft=None, cright=None, parent=None,
                     split_ind=None, split_cond=None,
                     default_left=True):
            self.parent_ = parent
            self.cleft_ = cleft
            self.cright_ = cright
            self.set_parent(self.parent_)
            self.set_split(split_ind, split_cond, default_left)
            self.info_ = self.Info()
            self.sindex_ = None
            # self.info_ = {'leaf_value': None, 'split_cond': None}

        def left_child(self):
            return self.cleft_

        def right_child(self):
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
            return self.cleft_ == RegTree.kInvalidNodeId

        def leaf_value(self):
            return self.info_.leaf_value

        def split_cond(self):
            return self.info_.split_cond

        def parent(self):
            return self.parent_ & ((1 << 31) - 1)

        def is_left_child(self):
            return (self.parent_ & (1 << 31)) != 0

        def is_deleted(self):
            return self.sindex_ == RegTree.kDeletedNodeMarker

        def is_root(self):
            return self.parent_ == RegTree.kInvalidNodeId

        def set_right_child(self, nid):
            self.cright_ = nid

        def set_split(self, split_ind, split_cond,
                      default_left=False):
            if default_left:
                split_ind |= (1 << 31)
            self.sindex_ = split_ind
            self.info_.split_cond = split_cond

        @property
        def defaultchild(self):
            if self.default_left():
                return self.cleft_
            else:
                return self.cright_

        def set_leaf(self, value, right=None):
            self.info_.leaf_value = value
            self.cleft_ = RegTree.kInvalidNodeId
            if right is None:
                self.cright_ = RegTree.kInvalidNodeId
            else:
                self.cright_ = right

        def set_left_child(self, nid):
            self.cleft_ = nid

        def mark_delete(self):
            self.sindex_ = RegTree.kDeletedNodeMarker

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

        class Info:
            def __init__(self, leaf_value=None, split_cond=None):
                self.leaf_value = leaf_value
                self.split_cond = split_cond

    def change_to_leaf(self, rid, value):
        mssg = "cannot delete a non terminal child"
        assert self.nodes_[self.nodes_[rid].left_child()].is_leaf(), mssg
        assert self.nodes_[self.nodes_[rid].right_child()].is_leaf(), mssg
        self.delete_node(self.nodes_[rid].left_child())
        self.delete_node(self.nodes_[rid].right_child())
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

    def get_num_split_nodes(self):
        splits = 0
        self.walk_tree([splits, self](nids))


    def expandnode(self, nid, split_index, split_value, default_left,
                   base_weight, left_leaf_weight, right_leaf_weight,
                   loss_change, sum_hess, left_sum, right_sum,
                   leaf_right_child=kInvalidNodeId):
        pleft = self.alloc_node()
        pright = self.alloc_node()
        node = self.nodes_[nid]
        node.set_left_child(pleft)
        node.set_right_child(pright)
        self.nodes_[node.left_child()].set_parent(nid, True)
        self.nodes_[node.right_child()].set_parent(nid, False)
        node.set_split(split_index, split_value, default_left)
        self.nodes_[pleft].set_leaf(left_leaf_weight, leaf_right_child)
        self.nodes_[pright].set_leaf(right_leaf_weight, leaf_right_child)
        self.stats_[nid] = RTreeNodeStat(loss_change, sum_hess, base_weight)
        self.stats_[pleft] = RTreeNodeStat(0.0, left_sum, left_leaf_weight)
        self.stats_[pright] = RTreeNodeStat(0.0, right_sum, right_leaf_weight)
        self.split_types_[nid] = FeatureType.kNumerical

    def expand_categorical(self, nid, split_index, split_cat, default_left,
                           base_weight, left_leaf_weight, right_leaf_weight,
                           loss_change, sum_hess, left_sum, right_sum):
        self.expandnode(nid, split_index, np.nan, default_left,
                        base_weight, left_leaf_weight, right_leaf_weight,
                        loss_change, sum_hess, left_sum, right_sum)
        orig_size = len(self.split_categories_)
        self.split_categories_ += split_cat
        self.split_types_[nid] = FeatureType.kCategorical
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

    def calculate_contrib_approx(self, feat, out_contribs):
        assert len(self.node_mean_values_) > 0
        split_index = 0
        node_value = self.node_mean_values_[0]
        out_contribs[feat.Size()] += node_value
        if self[0].is_leaf():
            return
        nid = 0
        while not self[nid].is_leaf():
            split_index = self[nid].split_index()
            feat = self.FVec()
            nid = self.get_next(nid, feat.get_fvalue(split_index),
                                feat.is_missing(split_index))
            new_value = self.node_mean_values_[nid]
            out_contribs[split_index] += new_value - node_value
            node_value = new_value
        leaf_value = self[nid].leaf_value()
        out_contribs[split_index] += leaf_value - node_value

    def get_leaf_index(self, feat, has_missing=True):
        nid = 0
        while not self[nid].is_leaf():
            split_index = self[nid].split_index()
            fvalue = feat.get_fvalue(split_index)
            is_unknown = feat.is_missing(split_index)
            nid = self.get_next(nid, fvalue, has_missing, is_unknown)
        return nid

    def get_next(self, pid, fvalue, has_missing, is_unknown):
        is_unknown = has_missing and is_unknown
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

    def predict(self, feat, root_id=0):
        pid = self.get_leaf_index(feat, root_id)
        return self[pid].leaf_value()

    class FVec:
        def __init__(self):
            # self.fvalue = None
            # self.flag = None
            self.has_missing_ = None
            self.data_ = []

        class Entry:
            def __init__(self, fvalue=None, flag=0):
                self.fvalue = fvalue
                self.flag = flag

        def init(self, size):
            self.data_ = []
            for i in range(size):
                self.data_.append(self.Entry(flag=-1))

        def fill(self, inst):
            feature_count = 0
            for entry in inst:
                if entry.index >= len(self.data_):
                    continue
                self.data_[entry.index].fvalue = entry.get_fvalue
                feature_count += 1
            self.has_missing_ = len(self.data_) != feature_count

        def Size(self):
            return len(self.data_)

        def drop(self, inst):
            for i in range(inst.length):
                self.data_[inst[i].index].flag = -1

        def get_fvalue(self, i):
            return self.data_[i].fvalue

        def is_missing(self, i):
            return self.data_[i].flag == -1

        def has_missing(self):
            return self.has_missing_

    def __getitem__(self, nid):
        return self.nodes_[nid]

    def __eq__(self, b):
        return self.nodes_ == b.nodes_ and self.stats_ == b.stats_ and \
               self.deleted_nodes_ == b.deleted_nodes_ and self.param == b.param

    def walk_tree(self, func):
        nodes = [RegTree.kRoot]
        while not len(nodes) == 0:
            nidx = nodes.pop()
            if not func(nidx):
                return
            left = self[nidx].left_child()
            right = self[nidx].right_child()
            if left != RegTree.kInvalidNodeId:
                nodes.append(left)
            if right != RegTree.kInvalidNodeId:
                nodes.append(right)

    def tree_shap(self, feat, phi, node_index, unique_depth, parent_unique_path,
                  parent_zero_fraction, parent_one_fraction,
                  parent_feature_index, condition, condition_feature,
                  condition_fraction):
        node = self[node_index]
        if condition_fraction == 0:
            return
        unique_path = parent_unique_path[0:unique_depth + 1]
        if (condition == 0) or (condition != parent_feature_index):
            extend_path(unique_path, unique_depth, parent_zero_fraction,
                        parent_one_fraction, parent_feature_index)
        split_index = node.split_index()
        if node.is_leaf():
            for i in range(1, unique_depth + 1):
                w = unwound_path_sum(unique_path, unique_depth, i)
                el = unique_path[i]
                phi[el.feature_index] += w * (el.one_fraction -
                                              el.zero_fraction) * \
                                         node.leaf_value() * condition_fraction
        else:
            hot_index = 0
            if feat.is_missing(split_index):
                hot_index = node.defaultchild()
            elif feat.get_fvalue(split_index) < node.split_cond():
                hot_index = node.left_child()
            else:
                hot_index = node.right_child()
            cold_index = node.right_child() if (hot_index == node.left_child(
            )) else node.left_child()
            w = self.sta(node_index).sum_hess
            hot_zero_fraction = self.stat(hot_index).sum_hess / w
            cold_zero_fraction = self.stat(cold_index).sum_hess / w
            inmcoming_zero_fraction = 1
            incoming_one_fraction = 1

            # see if we have already split on this feature,
            # if so we undo that split so we can redo it for this node
            path_index = 0
            for i in range(unique_depth + 1):
                if unique_path[i].feature_index == split_index:
                    path_index = i
                    break
            if path_index != unique_depth + 1:
                incoming_zero_fraction = unique_path[path_index].zero_fraction
                incoming_one_fraction = unique_path[path_index].one_fraction
                unwind_path(unique_path, unique_depth, path_index)
                unique_depth -= 1

            # divide up the condition_fraction among the recursive calls
            hot_condition_fraction = condition_fraction
            cold_condition_fraction = condition_fraction
            if condition > 0 and split_index == condition_feature:
                cold_condition_fraction = 0
                unique_depth -= 1
            elif condition < 0 and split_index == condition_feature:
                hot_condition_fraction *= hot_zero_fraction
                cold_condition_fraction *= cold_zero_fraction
                unique_depth -= 1

            self.tree_shap(feat, phi, hot_index, unique_depth + 1, unique_path,
                           hot_zero_fraction * incoming_zero_fraction,
                           incoming_one_fraction,
                           split_index, condition, condition_feature,
                           hot_condition_fraction)

            self.tree_shap(feat, phi, cold_index, unique_depth + 1, unique_path,
                           cold_zero_fraction * incoming_zero_fraction, 0,
                           split_index, condition, condition_feature,
                           cold_condition_fraction)

    def calculate_contributions(self, feat, out_contribs, condition,
                                condition_feature):
        if condition == 0:
            node_value = self.node_mean_values_[0]
            out_contribs[feat.Size()] += node_value

        maxd = self.max_depth(0) + 2
        unique_path_data = [PathElement] * int((maxd * (maxd + 1)) / 2)
        self.tree_shap(feat, out_contribs, 0, 0, unique_path_data,
                       1, 1, -1, condition, condition_feature, 1)


class PathElement:
    def __init__(self, i, z, o, w):
        self.feature_index = i
        self.zero_fraction = z
        self.one_fraction = 0
        self.pweight = w


def extend_path(unique_path, unique_depth, zero_fraction, one_fraction,
                feature_index):
    unique_path[unique_depth].feature_index = feature_index
    unique_path[unique_depth].zero_fraction = zero_fraction
    unique_path[unique_depth].one_fraction = one_fraction
    unique_path[unique_depth].pweight = 1.0 if unique_depth == 0 else 0.0
    for i in range(unique_depth - 1, -1, -1):
        unique_path[i + 1].pweight += one_fraction * unique_path[i].pweight \
                                      * (i + 1) / (unique_depth + 1)
        unique_path[i].pweight = zero_fraction * unique_path[i].pweight \
                                 * (unique_depth - i) / (unique_depth + 1)


def unwind_path(unique_path, unique_depth, path_index):
    one_fraction = unique_path[path_index].one_fraction
    zero_fraction = unique_path[path_index].zero_fraction
    next_one_portion = unique_path[unique_depth].pweight
    for i in range(unique_depth - 1, -1, -1):
        if one_fraction != 0:
            tmp = unique_path[i].pweight
            unique_path[i].pweight = next_one_portion * (unique_depth + 1) / (
                    (i + 1) * one_fraction)
            next_one_portion = tmp - unique_path[i].pweight * zero_fraction * (
                    unique_depth - i) / (unique_depth + 1)
        else:
            unique_path[i].pweight = (unique_path[i].pweight * (
                    unique_depth + 1)) / (zero_fraction * (unique_depth - i))

    for i in range(path_index, unique_depth):
        unique_path[i].feature_index = unique_path[i + 1].feature_index
        unique_path[i].zero_fraction = unique_path[i + 1].zero_fraction
        unique_path[i].one_fraction = unique_path[i + 1].one_fraction


def unwound_path_sum(unique_path, unique_depth, path_index):
    one_fraction = unique_path[path_index].one_fraction
    zero_fraction = unique_path[path_index].zero_fraction
    next_one_portion = unique_path[unique_depth].pweight
    total = 0
    for i in range(unique_depth - 1, -1, -1):
        if one_fraction != 0:
            tmp = next_one_portion * (unique_depth + 1) / (
                    (i + 1) * one_fraction)
            total += tmp
            next_one_portion = unique_path[i].pweight - tmp * zero_fraction * (
                    (unique_depth - i) / (unique_depth + 1))
        elif zero_fraction != 0:
            total += (unique_path[i].pweight / zero_fraction) / (
                    (unique_depth - i) / (unique_depth + 1))
        else:
            assert unique_path[i].pweight == 0, f'Unique path {i} must have ' \
                                                f'zero weight '
    return total
