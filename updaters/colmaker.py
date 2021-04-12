import numpy as np
from params import TrainParam, GradStats
from utils.util import resize, sample_binary


class ColMaker:
    def __init__(self):
        self.param = TrainParam()
        self.nthread = 1

    def set_param(self, name, value):
        self.param.set_param(name, value)

    def update(self, gpair, p_fmat, info, trees):
        GradStats.check_info(info)
        lr = self.param.learning_rate
        self.param.learning_rate = lr / len(trees)
        for i in range(len(trees)):
            builder = Builder(self.param)


class Builder:
    def __init__(self, param):
        self.param = TrainParam()
        self.qexpand_ = self.snode = self.stemp = self.position = None
        self.feat_index = None
        self.nthread = 1

    def update(self, gpair, p_fmat, info, p_tree):
        self.init_data(gpair, p_fmat, info.root_index, p_tree)
        self.init_new_node(gpair, p_fmat, info, p_tree)

    def init_data(self, gpair, fmat, root_index, tree):
        assert tree.param.num_nodes == tree.param.num_roots, "Colmaker"
        rowset = fmat.buffered_rowset()
        self.position = self.setup_position(gpair, root_index, rowset)
        self.feat_index = self.init(fmat)
        stemp, snode = self.setup_stat_temp(self.nthread)
        self.stemp = stemp
        self.snode = snode
        self.qexpand_ = list(np.arange(tree.param.num_roots))

    def init_new_node(self, gpair, fmat, info, tree):
        for i in range(len(self.stemp)):
            resize(self.stemp[i], tree.param.num_nodes, ThreadEntry(self.param))
        resize(self.snode, tree.param.num_nodes, NodeEntry(self.param))
        rowset = fmat.buffered_rowset()
        ndata = rowset.size()
        for i in range(ndata):
            ridx = rowset[i]
            tid = 1
            if self.position[ridx] < 0:
                continue
            self.stemp[tid][self.position[ridx]].stats.add_stats(gpair, info,
                                                                 ridx)
        for j in len(self.qexpand_):
            nid = self.qexpand[j]
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
        self.qexpand_ = np.array(newnodes)

    def enumerate_split(self, data, dstep, fid, gpair, info, temp):
        qexpand = self.qexpand_
        for j in range(len(qexpand)):
            temp[qexpand[j]].stats.clear()
        c = GradStats(self.param)
        for it in data:
            ridx = it.index



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
        n = self.param.colsample_bytree * len(feat_index)
        np.random.shuffle(feat_index)
        return list(feat_index[:n])

    @staticmethod
    def setup_stat_temp(self, nthread):
        stemp = [[] for _ in range(nthread)]
        snode = []
        return stemp, snode

    def setup_stat_tree(self, tree):
        n_node = tree.param.num_nodes
        stemp = np.full(n_node, ThreadEntry(self.param), dtype=object)
        snode = np.full(n_node, NodeEntry(self.param), dtype=object)
        return stemp, snode


class Iupdater:
    def __init__(self):
        pass


class Entry:
    def __init__(self, index, fvalue):
        self.index = index
        self.fvalue = fvalue

    def cmp_value(self, other):
        return self.fvalue < other.fvalue


class Inst:
    def __init__(self, entries, length):
        self.data = entries
        self.length = length

    def __getitem__(self, item):
        return self.data[item]


class ThreadEntry:
    def __init__(self, param):
        self.stats = GradStats(param)
        self.last_fvalue = None
        self.best = None


class NodeEntry:
    def __init__(self, param):
        self.stats = GradStats(param)
        self.root_gain = 0.0
        self.weight = 0.0
        self.best = None
