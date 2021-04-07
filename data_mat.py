import numpy as np
from collections import defaultdict
from info_class import MetaInfo


class CacheEntry:
    def __init__(self, mat, buffer_offset, num_row):
        self.mat_ = mat
        self.buffer_offset = buffer_offset
        self.num_row = num_row


class bst_gpair:
    def __init__(self, grad=None, hess=None):
        self.grad = grad
        self.hess = hess


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


class SparseCSRMBuilder:
    def __init__(self, rptr=[], findex=[], dummy_aclist=[],
                 use_aclist=False):
        self.rptr = rptr
        self.findex = findex
        self.aclist = dummy_aclist
        self.use_aclist = use_aclist

    def init_budget(self, nrows=0):
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


class RowBatch:
    def __init__(self, base_rowid=None, ind_ptr=None, data_ptr=None, size=None):
        self.base_rowid = base_rowid
        self.ind_ptr = ind_ptr
        self.data_ptr = data_ptr
        self.size = size

    def __getitem__(self, item):
        return Inst(self.data_ptr[self.ind_ptr[item]], self.ind_ptr[item+1] -
                    self.ind_ptr[item])


class ColBatch:
    def __init__(self, col_index=None, col_data=None):
        self.col_index = col_index
        self.col_data = col_data

    def __getitem__(self, item):
        return self.col_data[item]


class FMatrixS:
    def __init__(self, iter):
        self.iter = iter
        self.iter_ = RowBatch()
        self.col_iter_ = OneBatchIter()
        self.buffered_rowset_ = None

    def get_col_density(self, i):
        return 1 - (self.buffered_rowset_.size - self.col_ptr_[i + 1] +
                    self.col_ptr_[i]) / len(self.buffered_rowset_.size)

    def have_col_access(self):
        return self.col_ptr_.shape[0] != 0

    def num_col(self):
        return self.col_ptr_.shape[0] - 1

    def buffered_rowset(self):
        return self.buffered_rowset_

    def get_col_size(self, i):
        return self.col_ptr_[i + 1] - self.col_ptr_[i]

    def init_col_access(self, enabled, pkeep=1):
        if self.have_col_access():
            return
        self.init_col_data(pkeep)

    def init_col_data(self):
        self.buffered_rowset_ = None
        builder = SparseCSRMBuilder()
        builder.init_budget(0)


class OneBatchIter:
    def __init__(self, at_first=True):
        self.at_first_ = at_first
        self.batch_ = ColBatch()

    def before_first(self):
        self.at_first_ = True

    def next(self):
        if self.at_first_:
            return False
        self.at_first_ = False
        return True

    def value(self):
        return self.batch_

    def set_batch(self, ptr, data):
        self.batch_.size = self.col_index_.size
        self.col_data_ = np.empty(self.col_index_.size, dtype=object)
        for i in range(self.col_data_.size):
            ridx = self.col_index_[i]
            self.col_data_[i] = Inst(data[0:ptr[ridx]], ptr[ridx+1]-ptr[ridx])
        self.batch_ = ColBatch(self.col_index_, self.col_data_)



class OneBatchIter_row:
    def __init__(self, parent):
        self.at_first_ = True
        self.parent_ = parent
        self.batch_ = None

    def before_first(self):
        self.at_first_ = True

    def next(self):
        if self.at_first_:
            return False
        self.at_first_ = False
        self.batch_ = RowBatch(0, self.parent_.row_ptr, self.parent_.row_data,
                               size=self.parent_.row_ptr_.size - 1)
        return True

    def value(self):
        return self.batch_


class DMatrixSimple:
    def __init__(self):
        self.row_ptr_ = np.zeros(1)
        self.row_data_ = np.empty(0, dtype=object)
        self.info = MetaInfo()
        self.fmat_ = FMatrixS(OneBatchIter_row(self))
        self.size = 0

    def fmat(self):
        return self.fmat_

    def clear(self):
        self.row_ptr_ = np.zeros(1)
        self.row_data_ = np.empty(0, dtype=object)
        self.info.clear()

    def add_row(self, feats):
        for i in range(feats.size):
            np.append(self.row_data_, feats[i])
            num_col = np.max(self.info.info.num_col, feats[i].index + 1)
            setattr(self.info.info, 'num_col', num_col)
        np.append(self.row_ptr_, self.row_ptr_[-1] + feats.size())
        setattr(self.info.info, 'num_row', self.info.info.num_row + 1)
        return self.row_ptr_.size - 2


class DMatrix:
    def __init__(self, data, label=None, missing=0, weight=None):
        nr, nc = data.shape
        self.data = data.flatten()
        self.label = label
        self.weight = weight
        self.handle = from_np(self.data, nr, nc, missing=missing)
        if label is not None:
            self.set_label(label)
        if weight is not None:
            self.set_weight(weight)

    def set_param(self, name, val):
        setattr(self, name, val)
        return self

    def get_param(self, name):
        return getattr(self, name)

    def set_label(self, label):
        self.set_param('label', label)

    def set_base_margin(self, margin):
        self.set_param('base_margin', margin)

    def set_weight(self, weight):
        self.set_param('weight', weight)

    def num_row(self):
        return self.handle.info.num_row()


def from_np(data, nrow, ncol, missing=0.0):
    mat = DMatrixSimple()
    mat.info.info.num_row = nrow
    mat.info.info.num_col = ncol
    data_0 = 0
    for i in range(nrow):
        nelem = 0
        for j in range(ncol):
            if data[data_0+j] != missing:
                mat.row_data_ = np.append(mat.row_data_, Entry(j, data[data_0+j]))
                nelem += 1
        mat.row_ptr_ = np.append(mat.row_ptr_, mat.row_ptr_[-1] + nelem)
        data_0 += ncol
    return mat




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
    nnn = nn.flatten()
    nr, nc = nn.shape
    mattt = DMatrix(nn)
    print(mattt.row_ptr_)
