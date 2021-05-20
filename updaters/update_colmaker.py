import numpy as np
from params import TrainParam, GradStats, SplitEntry, GradientPair
from utils.util import resize, sample_binary
from data_i.data import DMatrix
from sklearn import datasets
from tree.split_evaluator import TreeEvaluator
from utils.random import ColumnSampler
from objective.loss_function import LinearSquareLoss
from multiprocessing import Process
import time
from data_i.data import CSRAdapter, SortedCSCPage
from scipy.sparse import csr_matrix, csc_matrix


rt_eps = 1e-5
rt_2eps = 2e-5
kRtEps = 1e-6


class ColMakerTrainParam(TrainParam):
    def __init__(self):
        super().__init__()
        self.opt_dense_col = 1.0

    def need_forward_search(self, default_direction, col_density, indicator):
        return (default_direction == 2) or (
                (default_direction == 0) and (
                col_density < self.opt_dense_col and not indicator))

    def need_backward_search(self, default_direction):
        return default_direction != 2


class ColMaker:
    """ xgboost.tree  updater_colmaker-inl.hpp"""

    def __init__(self, data):
        self.param_ = ColMakerTrainParam()
        self.colmaker_train_param_ = ColMakerTrainParam()
        # self.data_ = data
        # self.column_densities_ = self.lazy_get_column_density()
        self.column_densities_ = [1.0]*12
        self.interaction_constraints_ = ''

    def set_param(self, name, value):
        self.param_.set_param(name, value)

    def lazy_get_column_density(self):
        return self.data_.getnnz(axis=0) / self.data_.shape[0]

    def update(self, gpair, p_fmat, trees):
        lr = self.param_.learning_rate
        self.param_.learning_rate = lr / len(trees)
        for tree in trees:
            builder = ColMaker.Builder(self.param_, self.colmaker_train_param_,
                                       self.interaction_constraints_,
                                       self.column_densities_)
            builder.update(gpair, p_fmat, tree)

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

    def __repr__(self):
        return 'grow_colmaker'

    class Builder:
        def __init__(self, param, colmaker_train_param,
                     interaction_constraints,
                     column_densities):
            # to remove
            self.param_ = TrainParam()
            self.colmaker_train_param_ = colmaker_train_param
            self.interaction_constraints_ = interaction_constraints
            self.tree_evaluator_ = TreeEvaluator(param,
                                                 len(column_densities), -1)
            self.column_densities_ = column_densities

            self.nthread = 1
            self.position_ = []
            self.stemp_ = []
            self.snode_ = []
            self.qexpand_ = []
            self.feat_index = None

            self.column_sampler_ = ColumnSampler()

        def update(self, gpair, p_fmat, p_tree):
            self.init_data(gpair, p_fmat)
            self.init_new_node(self.qexpand_, gpair, p_fmat, p_tree)
            for depth in range(self.param_.max_depth):
                self.find_split(depth, self.qexpand_, gpair, p_fmat, p_tree)
            #     self.find_split(depth, self.qexpand_, gpair, p_fmat, p_tree)
                self.reset_position(self.qexpand_, p_fmat, p_tree)
            #     self.update_queue_expand(p_tree, self.qexpand_)
            #     self.init_new_node(self.qexpand_, gpair, p_fmat, p_tree)
            #     if len(self.qexpand_) == 0:
            #         break
            # for i in range(len(self.qexpand_)):
            #     nid = self.qexpand_[i]
            #     p_tree[nid].set_leaf(self.snode_[nid].weight *
            #                          self.param_.learning_rate)
            # for nid in range(p_tree.param.num_nodes):
            #     p_tree.stat(nid).loss_chg = self.snode_[nid].best.loss_chg
            #     p_tree.stat(nid).base_weight = self.snode_[nid].weight
            #     p_tree.stat(nid).sum_hess = self.snode_[nid].stats_.sum_hess
            #     self.snode_[nid].stats_.set_leaf_vec(self.param_,
            #                                          p_tree.leafvec(nid))

        def init_data(self, gpair, fmat):

            self.setup_position(gpair, fmat)
            self.column_sampler_.init(fmat.info().num_col_,
                                      fmat.info().feature_weigths,
                                      self.param_.colsample_bynode,
                                      self.param_.colsample_bylevel,
                                      self.param_.colsample_bytree)

            self.stemp_ = [[None] for _ in range(self.nthread)]
            self.snode_ = []
            self.qexpand_ = [0]

        def init_new_node(self, qexpand, gpair, fmat, tree):
            for i in range(len(self.stemp_)):
                self.stemp_[i] = [ThreadEntry() for _ in
                                  range(tree.param.num_nodes)]
            self.snode_ = [NodeEntry() for _ in range(tree.param.num_nodes)]
            info = fmat.info()
            ndata = info.num_row_
            for ridx in range(ndata):
                tid = 0
                if self.position_[ridx] < 0:
                    continue
                self.stemp_[tid][self.position_[ridx]].stats.add(gpair[ridx])

            for nid in qexpand:
                stats = GradStats()
                for s in self.stemp_:
                    stats.add(s[nid].stats)
                self.snode_[nid].stats = stats

            evaluator = self.tree_evaluator_.get_evaluator()
            for nid in qexpand:
                parentid = tree[nid].parent()
                self.snode_[nid].weight = evaluator.calc_weight(parentid,
                                                                self.param_,
                                                                self.snode_[
                                                                    nid].stats)
                self.snode_[nid].root_gain = evaluator.calc_gain(parentid,
                                                                 self.param_,
                                                                 self.snode_[
                                                                     nid].stats)

        def update_queue_expand(self, tree, qexpand):
            p_newnodes = []
            for nid in qexpand:
                if not tree[nid].is_leaf():
                    p_newnodes.append(tree[nid].left_child())
                    p_newnodes.append(tree[nid].right_child())
            return p_newnodes

        def update_enumeration(self, nid, gstats, fvalue, d_step, fid, c, temp,
                               evaluator):
            e = temp[nid]
            if e.stats.empty():
                e.stats.add(gstats)
                e.last_fvalue = fvalue
            else:
                if fvalue != e.last_fvalue and e.stats.sum_hess >= \
                        self.param_.min_child_weight:
                    c.set_substract(self.snode_[nid].stats, e.stats)
                    if c.sum_hess >= self.param_.min_child_weight:
                        loss_chg = 0
                        if d_step == -1:
                            loss_chg = evaluator.calc_split_gain(self.param_,
                                                                 nid, fid, c,
                                                                 e.stats) - \
                                       self.snode_[nid].root_gain
                            proposed_split = (fvalue + e.last_fvalue) * 0.5
                            if proposed_split == fvalue:
                                e.best.update(loss_chg, fid, e.last_fvalue,
                                              d_step == -1, c, e.stats)
                            else:
                                e.best.update(loss_chg, fid, proposed_split,
                                              d_step == -1, c, e.stats)

                        else:
                            loss_chg = evaluator.calc_split_gain(self.param_,
                                                                 nid, fid,
                                                                 e.stats, c) - \
                                       self.snode_[nid].root_gain
                            proposed_split = (fvalue + e.last_fvalue) * 0.5
                            if proposed_split == fvalue:
                                e.best.update(loss_chg, fid, e.last_fvalue,
                                              d_step == -1, e.stats, c)
                            else:
                                e.best.update(loss_chg, fid, proposed_split,
                                              d_step == -1, e.stats, c)

                e.stats.add(gstats)
                e.last_fvalue = fvalue

        def enumerate_split(self, data, d_step, fid, gpair, tid, evaluator):

            qexpand = self.qexpand_
            temp = self.stemp_[tid]
            for nid in qexpand:
                temp[nid].stats = GradStats()
            c = GradStats()
            kBuffer = len(data)
            buf_position = [0]*kBuffer
            buf_gpair = [GradientPair() for _ in range(kBuffer)]
            for i in range(len(data)):
                it = data[i]
                buf_position[i] = self.position_[it.index]
                buf_gpair[i] = gpair[it.index]
            for i in range(len(data)):
                it = data[i]
                nid = buf_position[i]
                if nid < 0:
                    continue
                self.update_enumeration(nid, buf_gpair[i],
                                        it.fvalue, d_step,
                                        fid, c, temp, evaluator)

            for nid in qexpand:
                e = temp[nid]
                c.set_substract(self.snode_[nid].stats, e.stats)
                if (e.stats.sum_hess >= self.param_.min_child_weight) and \
                        (c.sum_hess >= self.param_.min_child_weight):
                    gap = np.abs(e.last_fvalue) + kRtEps
                    delta = gap if d_step == 1 else -gap
                    if d_step == -1:
                        new_loss = evaluator.calc_split_gain(self.param_, nid,
                                                             fid, c, e.stats)
                        loss_chg = new_loss - self.snode_[nid].root_gain
                        e.best.update(loss_chg, fid, e.last_fvalue + delta,
                                      d_step == -1, c, e.stats)
                    else:
                        new_loss = evaluator.calc_split_gain(self.param_, nid,
                                                             fid, e.stats, c)
                        loss_chg = new_loss - self.snode_[nid].root_gain
                        e.best.update(loss_chg, fid, e.last_fvalue + delta,
                                      d_step == -1, e.stats, c)
                self.stemp_[tid][nid] = e

        def update_solution(self, batch, feat_set, gpair, p_fmat):
            num_features = len(feat_set)
            page = batch.get_view()
            for i in range(num_features):
                evaluator = self.tree_evaluator_.get_evaluator()
                fid = feat_set[i]
                tid = 0
                c = page[fid]
                len_c = len(c)
                ind = (len_c != 0) and (c[0].fvalue == c[-1].fvalue)
                if self.colmaker_train_param_.need_forward_search(
                        self.param_.default_direction,
                        self.column_densities_[fid], ind):
                    self.enumerate_split(c, +1, fid, gpair, tid,
                                         evaluator)
                if self.colmaker_train_param_.need_backward_search(
                        self.param_.default_direction):
                    c.reverse()
                    self.enumerate_split(c, -1, fid, gpair,
                                         tid, evaluator)

        def find_split(self, depth, qexpand, gpair, p_fmat, p_tree):
            evaluator = self.tree_evaluator_.get_evaluator()
            feat_set = self.column_sampler_.get_feature_set(depth)

            page = p_fmat.sparse_page_.get_transpose(p_fmat.info().num_col_)
            batch = SortedCSCPage(page)
            self.update_solution(batch, feat_set, gpair, p_fmat)
            # for batch in p_fmat.get

#
#         def find_split(self, depth, qexpand, gpair, p_fmat, p_tree):
#             evaluator = self.tree_evaluator_.get_evaluator()
#             feat_set = self.column_sampler_.get_feature_set(depth)
#             p_fmat =
#             for batch in p_fmat.get
#
#
#             if self.param_.colsample_bylevel != 1:
#                 np.random.shuffle(feat_set)
#                 n = self.param_.colsample_bylevel * len(feat_set)
#                 assert n > 0, 'colsample_bylevel is too small'
#                 resize(feat_set, n)
#             iter_i = p_fmat.col_iterator(feat_set)
#             while iter_i.next():
#                 batch = iter_i.value()
#                 nsize = batch.size
#                 # batch_size = np.maximum(nsize/(32*self.nthread), 1)
#                 for i in range(nsize):
#                     fid = batch.col_index[i]
#                     tid = 0
#                     c = batch[i]
#                     if self.param_.need_forward_search(
#                             p_fmat.get_col_density(fid)):
#                         self.enumerate_split(c.data_[0:c.length], 1, fid, gpair,
#                                              info, self.stemp_[tid])
#                     if self.param_.need_backward_search(
#                             p_fmat.get_col_density(fid)):
#                         self.enumerate_split(c.data_[0:c.length], -1, fid,
#                                              gpair,
#                                              info, self.stemp_[tid])
#
#             for i in range(len(qexpand)):
#                 nid = qexpand[i]
#                 e = self.snode_[nid]
#                 for tid in range(self.nthread):
#                     e.best.update_e(self.stemp_[tid][nid].best)
#                 if e.best.loss_chg > rt_eps:
#                     p_tree.add_child(nid)
#                     p_tree[nid].set_split(e.best.split_index(),
#                                           e.best.split_value,
#                                           e.best.default_left())
#                 else:
#                     p_tree[nid].set_leaf(e.weight * self.param_.learning_rate)
#
        def reset_position(self, qexpand, p_fmat, tree):
            rowset = p_fmat.buffered_rowset()
            ndata = len(rowset)
            for i in range(ndata):
                ridx = rowset[i]
                nid = self.position_[ridx]
                if nid >= 0:
                    if tree[nid].is_leaf():
                        self.position_[ridx] = -1
                    else:
                        new_pos = tree[nid].left_child() if tree[
                            nid].left_child() \
                            else tree[nid].right_child()
                        self.position_[ridx] = new_pos
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
                        fvalue = col[j].get_fvalue
                        nid = self.position_[ridx]
                        if nid == -1: continue
                        nid = tree[nid].parent()
                        if tree[nid].split_index() == fid:
                            if fvalue < tree[nid].split_cond():
                                self.position_[ridx] = tree[nid].left_child()
                            else:
                                self.position_[ridx] = tree[nid].right_child()



        def setup_position(self, gpair, fmat):
            self.position_ = [0] * len(gpair)
            assert fmat.info().num_row_ == len(self.position_)
            for ridx in range(len(self.position_)):
                if gpair[ridx].get_hess() < 0.0:
                    self.position_[ridx] = ~self.position_[ridx]
            if self.param_.subsample < 1.0:
                assert self.param_.sampling_method == 0
                for ridx in range(len(self.position_)):
                    if gpair[ridx].hess < 0:
                        continue
                    if not sample_binary(self.param_.subsample):
                        self.position_[ridx] = ~self.position_[ridx]

        def init_findex(self, fmat):
            feat_index = []
            ncol = fmat.num_col()
            for i in range(ncol):
                if fmat.get_col_size(i) != 0:
                    feat_index.append(i)
            n = int(self.param_.colsample_bytree * len(feat_index))
            np.random.shuffle(feat_index)
            return list(feat_index[:n])

        @staticmethod
        def setup_stat_temp(nthread):
            stemp = [[] for _ in range(nthread)]
            snode = []
            return stemp, snode

        def setup_stat_tree(self, tree):
            n_node = tree.param_.num_nodes
            stemp = [ColMaker.ThreadEntry(self.param_)] * n_node
            snode = [ColMaker.NodeEntry(self.param_)] * n_node
            return stemp, snode


class ThreadEntry:
    def __init__(self, last_fvalue=0):
        self.stats = GradStats()
        self.last_fvalue = last_fvalue
        self.best = SplitEntry()


class NodeEntry:
    def __init__(self, root_gain=0.0, weight=0.0):
        self.stats = GradStats()
        self.root_gain = root_gain
        self.weight = weight
        self.best = None


if __name__ == "__main__":
    # data_ = pd.read_csv('~/projects/Learning/OCR_text/opencv-text-detection'
    #                   '/Python_script/data_/covid.csv')
    # diabetes = datasets.load_diabetes()
    # X, y = diabetes.data, diabetes.target
    # dmat = DMatrix(X, label=y)
    # dmat.handle.fmat().init_col_access()
    # print(0)
    import xgboost as xgb

    boston = datasets.load_boston()
    data = boston['data']
    X = data[:, :-1]
    y = data[:, -1]

    x_csr = csr_matrix(X)
    row_ptr = x_csr.indptr
    feature_idx = x_csr.indices
    values = x_csr.data
    num_rows, num_features = x_csr.shape
    num_elements = x_csr.nnz

    adapt = CSRAdapter(row_ptr, feature_idx, values, num_rows, num_elements,
                       num_features)

    dm = DMatrix(x_csr).create()

    # dm = DMatrix(x_csr)
    # dm = DMatrix(X, label=y)

    gg = LinearSquareLoss()
    base_score = 0.5
    base_s = np.full(y.shape, base_score)
    grad = gg.gradient(base_s, y)
    hess = gg.hessian(base_s, y)
    gpair = []
    for i in range(len(grad)):
        gpair.append(GradientPair(grad[i], hess[i]))

    col = ColMaker(dm)

    from tree.tree import RegTree
    trees = [RegTree()]

    col.update(gpair, dm, trees)
    print('done')



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
#         self.data_ = entries
#         self.length = length
#
#     def __getitem__(self, item):
#         return self.data_[item]
#
#
