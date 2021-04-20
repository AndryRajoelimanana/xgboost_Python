import numpy as np
from collections import defaultdict
from info_class import MetaInfo


class bst_gpair:
    def __init__(self, grad=None, hess=None):
        self.grad = grad
        self.hess = hess


class Sparse:
    def __init__(self, indata):
        self.indata = indata

    def get_col_density(self, i):
        return 1 - (len(self.indices) - self.indptr[i + 1] +
                    self.indptr[i]) / len(self.indices)

    def have_col_access(self):
        return self.indptr.shape[0] != 0

    def num_col(self):
        return self.indptr.shape[0] - 1

    def get_col_size(self, i):
        return self.indptr[i + 1] - self.indptr[i]

    def init_col_access(self, enabled, pkeep=1):
        if self.have_col_access():
            return
        self.init_col_data(pkeep, enabled)


class SparseCSR(Sparse):
    def __init__(self, indata):
        super().__init__(indata)
        self.indptr, self.indices, self.data = self.to_sparse()

    def to_sparse(self):
        indptr = [0]
        indices = []
        data_list = []
        for i, d in enumerate(self.indata):
            for j, dd in enumerate(d):
                if dd != 0:
                    indices.append(j)
                    data_list.append(dd)
            indptr.append(len(indices))
        return indptr, indices, data_list

    def __getitem__(self, item):
        return self.Inst(self.data[self.indptr[item]:self.indptr[item + 1]],
                    self.indptr[item + 1] - self.indptr[item])


class SparseCSC(Sparse):
    def __init__(self, indata):
        super().__init__(indata)
        self.indptr, self.indices, self.data = self.to_sparse()

    def to_sparse(self):
        indptr = [0] * (self.indata.shape[1] + 1)
        indices = []
        data_list = []
        for i, d in enumerate(self.indata):
            d = self.indata[i, :]
            count = 1
            for j, dd in enumerate(d):
                if dd != 0:
                    indices.append(i)
                    data_list.append(dd)
                    count += 1
                    indptr[j + 1] += 1
        indptr = np.cumsum(indptr)
        return indptr, indices, data_list

    def buffered_rowset(self):
        return self.buffered_rowset_

    def __getitem__(self, item):
        return Inst(self.data[self.indptr[item]:self.indptr[item + 1]],
                    self.indptr[item + 1] - self.indptr[item])


class NodeEntry:
    def __init__(self, param):
        self.stats = param
        self.root_gain = 0
        self.weight = 0


class ThreadEntry:
    def __init__(self, param):
        self.stats = param


