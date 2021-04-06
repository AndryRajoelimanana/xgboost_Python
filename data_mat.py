import numpy as np


class Sparse:
    def __init__(self, indata):
        self.indata = indata

    def get_col_density(self, i):
        return 1 - (len(self.indices) - self.indptr[i + 1] + self.indptr[
            i]) / len(
            self.indices)

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
        return Inst(self.data[self.indptr[item]:self.indptr[item + 1]],
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


class Entry:
    def __init__(self, index, fvalue):
        """ Entry of sparse vector"""
        self.index = index
        self.fvalue = fvalue

    def __cmp__(self, other):
        return self.fvalue < other.fvalue


class NodeEntry:
    def __init__(self, param):
        self.stats = param
        self.root_gain = 0
        self.weight = 0


class ThreadEntry:
    def __init__(self, param):
        self.stats = param


class Inst:
    def __init__(self, data, length):
        self.data = data
        self.length = length

    def __getitem__(self, item):
        return self.data[item]


class Fmatrix:
    def __init__(self, rptr=[], findex=[], dummy_aclist=[],
                 use_aclist=False):
        self.rptr = rptr
        self.findex = findex
        self.aclist = dummy_aclist
        self.use_aclist = use_aclist

    def init_bugets(self, nrows=0):
        if not self.use_aclist:
            self.rptr = np.zeros(nrows + 1)
        else:
            assert nrows + 1 == self.rptr.shape[0]
            self.cleanup()

    def cleanup(self):
        for ac in self.aclist:
            self.rptr[ac] = 0
            self.rptr[ac + 1] = 0
        self.aclist = []

    def add_budget(self, row_id, nelem=1):
        if self.rptr.shape[0] < row_id + 2:
            self.rptr.resize(row_id+2)
        if self.use_aclist:
            if self.rptr[row_id+1] == 0:
                self.aclist = np.append(self.aclist, row_id)
        self.rptr[row_id+1] += nelem

    def init_storage(self):
        start = 0
        if not self.use_aclist:
            for i in range(self.rptr.shape[0]):
                rlen = self.rptr[i]
                self.rptr[i] = start
                start += rlen
        else:
            aclist = np.sort(self.aclist)
            for i in range(aclist.shape[0]):
                ridx = aclist[i]
                rlen = self.rptr[ridx+1]
                self.rptr[ridx+1] = start
                if i == 0 or (ridx != aclist[i-1] + 1):
                    self.rptr[ridx] = start
                start += rlen

    def push_elem(self, row_id, col_id):
        rp = self.rptr[row_id + 1]
        self.findex[rp+1] = col_id


def for_numpy(data, missing=0):
    n_row, n_col = data.shape
    mat = Fmatrix()




if __name__ == "__main__":
    # y0 = np.array([1, 0, 1, 0])
    # ypred0 = np.array([0.3, 0.5, 0.2, 0.3])
    # loss = SquareErrorLoss()
    # yy = np.random.uniform(size=(5, 3))
    # print(yy, loss.softmax(yy))
    # print(loss.get_gradient(y0, ypred0))
    nn = np.array([[1, 0, 3, 4],
                   [0, 1, 3, 0],
                   [1, 0, 0, 0],
                   [0, 0, 1, 5]])
    mmm = SparseCSC(nn.T)
    print(mmm[1])
