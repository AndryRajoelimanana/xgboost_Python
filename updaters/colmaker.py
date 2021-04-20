import numpy as np
from params import TrainParam, GradStats, SplitEntry
from utils.util import resize, sample_binary
import pandas as pd
from utils.simple_matrix import DMatrix
from sklearn import datasets


rt_eps = 1e-5
rt_2eps = 2e-5


class ColMaker:
    """ xgboost.tree  updater_colmaker-inl.hpp"""
    def __init__(self):
        self.param = TrainParam()

    def set_param(self, name, value):
        self.param.set_param(name, value)

    def update(self, gpair, p_fmat, info, trees):
        GradStats.check_info(info)
        lr = self.param.learning_rate
        self.param.learning_rate = lr / len(trees)
        for i in range(len(trees)):
            builder = ColMaker.Builder(self.param)
            builder.update(gpair, p_fmat, info, trees[i])

    class ThreadEntry:
        def __init__(self, param):
            self.stats = GradStats(param)
            self.last_fvalue = None
            self.best = SplitEntry()

    class NodeEntry:
        def __init__(self, param):
            self.stats = GradStats(param)
            self.root_gain = 0.0
            self.weight = 0.0
            self.best = SplitEntry()

    class Builder:
        def __init__(self, param):
            # to remove
            self.param = TrainParam()
            self.nthread = 1
            self.position = []
            self.stemp = []
            self.snode = []
            self.qexpand_ = []
            self.feat_index = None

        def update(self, gpair, p_fmat, info, p_tree):
            self.init_data(gpair, p_fmat, info.root_index, p_tree)
            self.init_new_node(gpair, p_fmat, info, p_tree)
            for depth in range(self.param.max_depth):
                self.find_split(self.qexpand_, gpair, p_fmat, info, p_tree)
                self.reset_position(self.qexpand_, p_fmat, p_tree)
                self.update_queue_expand(p_tree)
                self.init_new_node(gpair, p_fmat, info, p_tree)
                if len(self.qexpand_) == 0:
                    break
            for i in range(len(self.qexpand_)):
                nid = self.qexpand_[i]
                p_tree[nid].set_leaf(self.snode[nid].weight *
                                     self.param.learning_rate)
            for nid in range(p_tree.param.num_nodes):
                p_tree.stat(nid).loss_chg = self.snode[nid].best.loss_chg
                p_tree.stat(nid).base_weight = self.snode[nid].weight
                p_tree.stat(nid).sum_hess = self.snode[nid].stats.sum_hess
                self.snode[nid].stats.set_leaf_vec(self.param,
                                                   p_tree.leafvec(nid))

        def init_data(self, gpair, fmat, root_index, tree):
            assert tree.param.num_nodes == tree.param.num_roots, "Colmaker"
            rowset = fmat.buffered_rowset()
            self.position = self.setup_position(gpair, root_index, rowset)
            self.feat_index = self.init_findex(fmat)
            stemp, snode = self.setup_stat_temp(self.nthread)
            self.stemp = stemp
            self.snode = snode
            self.qexpand_ = list(np.arange(tree.param.num_roots))
            print('te')

        def init_new_node(self, gpair, fmat, info, tree):
            for i in range(len(self.stemp)):
                resize(self.stemp[i], tree.param.num_nodes,
                       ColMaker.ThreadEntry(self.param))
            resize(self.snode, tree.param.num_nodes,
                   ColMaker.NodeEntry(self.param))
            rowset = fmat.buffered_rowset()
            ndata = len(rowset)
            for i in range(ndata):
                ridx = rowset[i]
                tid = 0
                if self.position[ridx] < 0:
                    continue
                self.stemp[tid][self.position[ridx]].stats.add_stats(gpair, info,
                                                                     ridx)
            for j in range(len(self.qexpand_)):
                nid = self.qexpand_[j]
                stats = GradStats(self.param)
                for tid in range(len(self.stemp)):
                    stats.add_pair(self.stemp[tid][nid].stats)
                self.snode[nid].stats = stats
                self.snode[nid].root_gain = stats.calc_gain(self.param)
                self.snode[nid].weight = stats.calc_weight(self.param)

        def update_queue_expand(self, tree):
            newnodes = []
            qexpand = self.qexpand_
            for i in range(len(qexpand)):
                nid = qexpand[i]
                if not tree[nid].is_leaf():
                    newnodes.append(tree[nid].cleft())
                    newnodes.append(tree[nid].cright())
            self.qexpand_ = newnodes

        def enumerate_split(self, data, d_step, fid, gpair, info, temp):
            qexpand = self.qexpand_
            for j in range(len(qexpand)):
                temp[qexpand[j]].stats.clear()
            c = GradStats(self.param)
            if d_step == 1:
                to_loop = np.arange(len(data))
            elif d_step == -1:
                to_loop = np.arange(len(data)-1, -1, -1)
            for it_i in to_loop:
                it = data[it_i]
                ridx = it.index
                nid = self.position[ridx]
                if nid < 0:
                    continue
                fvalue = it.fvalue
                e = temp[nid]
                if e.stats.empty():
                    e.stats.add_stats(gpair, info, ridx)
                    e.last_fvalue = fvalue
                else:
                    if np.abs(fvalue - e.last_fvalue) > rt_2eps and \
                            e.stats.sum_hess >= self.param.min_child_weight:
                        c.set_substract(self.snode[nid].stats, e.stats)
                        if c.sum_hess >= self.param.min_child_weight:
                            loss_chg = e.stats.calc_gain(self.param) + \
                                       c.calc_gain(self.param) - \
                                       self.snode[nid].root_gain
                            e.best.update(loss_chg, fid,
                                          (fvalue + e.last_fvalue)*0.5,
                                          d_step == -1)
                    e.stats.add_stats(gpair, info, ridx)
                    e.last_fvalue = fvalue
            for i in range(len(qexpand)):
                nid = qexpand[i]
                e = temp[nid]
                c.set_substract(self.snode[nid].stats, e.stats)
                if (e.stats.sum_hess >= self.param.min_child_weight) and \
                        (c.sum_hess >= self.param.min_child_weight):
                    loss_chg = e.stats.calc_gain(self.param) + \
                               c.calc_gain(self.param) - \
                               self.snode[nid].root_gain
                    delta = rt_eps if d_step == 1 else -rt_eps
                    e.best.update(loss_chg, fid, e.last_fvalue + delta,
                                  d_step == -1)

        def find_split(self, qexpand, gpair, p_fmat, info, p_tree):
            feat_set = self.feat_index
            if self.param.colsample_bylevel != 1:
                np.random.shuffle(feat_set)
                n = self.param.colsample_bylevel * len(feat_set)
                assert n > 0, 'colsample_bylevel is too small'
                resize(feat_set, n)
            iter_i = p_fmat.col_iterator(feat_set)
            while iter_i.next():
                batch = iter_i.value()
                nsize = batch.size
                # batch_size = np.maximum(nsize/(32*self.nthread), 1)
                for i in range(nsize):
                    fid = batch.col_index[i]
                    tid = 0
                    c = batch[i]
                    if self.param.need_forward_search(
                            p_fmat.get_col_density(fid)):
                        self.enumerate_split(c.data[0:c.length], 1, fid, gpair,
                                             info, self.stemp[tid])
                    if self.param.need_backward_search(
                            p_fmat.get_col_density(fid)):
                        self.enumerate_split(c.data[0:c.length], -1, fid, gpair,
                                             info, self.stemp[tid])

            for i in range(len(qexpand)):
                nid = qexpand[i]
                e = self.snode[nid]
                for tid in range(self.nthread):
                    e.best.update_e(self.stemp[tid][nid].best)
                if e.best.loss_chg > rt_eps:
                    p_tree.add_child(nid)
                    p_tree[nid].set_split(e.best.split_index(),
                                          e.best.split_value,
                                          e.best.default_left())
                else:
                    p_tree[nid].set_leaf(e.weight * self.param.learning_rate)

        def reset_position(self, qexpand, p_fmat, tree):
            rowset = p_fmat.buffered_rowset()
            ndata = len(rowset)
            for i in range(ndata):
                ridx = rowset[i]
                nid = self.position[ridx]
                if nid >= 0:
                    if tree[nid].is_leaf():
                        self.position[ridx] = -1
                    else:
                        new_pos = tree[nid].cleft() if tree[nid].cleft() \
                            else tree[nid].cright()
                        self.position[ridx] = new_pos
            fsplits = []
            for i in range(len(qexpand)):
                nid = qexpand[i]
                if not tree[nid].is_leaf():
                    fsplits.append(tree[nid].split_index())
            fsplits = np.unique(fsplits).tolist()
            iter_i = p_fmat.col_iterator(fsplits)
            while iter_i.next():
                batch = iter_i.value()
                for i in range(batch.size):
                    col = batch[i]
                    fid = batch.col_index[i]
                    ndata = col.length
                    for j in range(ndata):
                        ridx = col[j].index
                        fvalue = col[j].fvalue
                        nid = self.position[ridx]
                        if nid == -1: continue
                        nid = tree[nid].parent()
                        if tree[nid].split_index() == fid:
                            if fvalue < tree[nid].split_cond():
                                self.position[ridx] = tree[nid].cleft()
                            else:
                                self.position[ridx] = tree[nid].cright()

        def setup_position(self, gpair, root_index, rowset):
            position = [0]*len(gpair)
            if len(root_index) == 0:
                for i in range(len(rowset)):
                    position[rowset[i]] = 0
            else:
                for i in range(len(rowset)):
                    ridx = rowset[i]
                    if gpair[ridx].hess < 0:
                        position[ridx] = -1
                    else:
                        position[ridx] = root_index[ridx]
            if self.param.subsample < 1.0:
                for i in range(len(rowset)):
                    ridx = rowset[i]
                    if gpair[ridx].hess < 0:
                        continue
                    if not sample_binary(self.param.subsample):
                        position[ridx] = -1
            return position

        def init_findex(self, fmat):
            feat_index = []
            ncol = fmat.num_col()
            for i in range(ncol):
                if fmat.get_col_size(i) != 0:
                    feat_index.append(i)
            n = int(self.param.colsample_bytree * len(feat_index))
            np.random.shuffle(feat_index)
            return list(feat_index[:n])

        @staticmethod
        def setup_stat_temp(nthread):
            stemp = [[] for _ in range(nthread)]
            snode = []
            return stemp, snode

        def setup_stat_tree(self, tree):
            n_node = tree.param.num_nodes
            stemp = [ColMaker.ThreadEntry(self.param)] * n_node
            snode = [ColMaker.NodeEntry(self.param)] * n_node
            return stemp, snode


if __name__ == "__main__":
    # data = pd.read_csv('~/projects/Learning/OCR_text/opencv-text-detection'
    #                   '/Python_script/data/covid.csv')
    diabetes = datasets.load_diabetes()
    X, y = diabetes.data, diabetes.target
    dmat = DMatrix(X, label=y)
    dmat.handle.fmat().init_col_access()
    print(0)


# class Iupdater:
#     def __init__(self):
#         pass
#
#
# class Entry:
#     def __init__(self, index, fvalue):
#         self.index = index
#         self.fvalue = fvalue
#
#     def cmp_value(self, other):
#         return self.fvalue < other.fvalue
#
#
# class Inst:
#     def __init__(self, entries, length):
#         self.data = entries
#         self.length = length
#
#     def __getitem__(self, item):
#         return self.data[item]
#
#



