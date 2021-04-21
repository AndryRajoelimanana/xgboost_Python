from data.matrix_csr import SparseCSRMBuilder
import numpy as np
from data.data_io import SparseBatch, RowBatch, ColBatch
from utils.iterator import IIterator
from utils.util import sample_binary
from data.dmatrix import DataMatrix
from utils.util import resize


class IFMatrix:
    def row_iterator(self):
        pass

    def col_iterator(self, fset=None):
        pass

    def init_col_access(self, subsample):
        pass

    def have_col_access(self):
        pass

    def num_col(self):
        pass

    def get_col_size(self, cidx):
        pass

    def get_col_density(self, cidx):
        pass

    def buffered_rowset(self):
        pass


class FMatrixS(IFMatrix):
    """
    simple_fmatrix-inl.hpp
    """

    def __init__(self, iters):
        self.iter_ = iters
        self.col_iter_ = self.OneBatchIter()
        self.col_ptr_ = []
        self.col_data_ = []
        self.buffered_rowset_ = []

    def have_col_access(self):
        return len(self.col_ptr_) != 0

    def num_col(self):
        return len(self.col_ptr_) - 1

    def buffered_rowset(self):
        return self.buffered_rowset_

    def get_col_size(self, i):
        return self.col_ptr_[i + 1] - self.col_ptr_[i]

    def get_col_density(self, i):
        bsize = len(self.buffered_rowset_)
        return 1.0 - (bsize - self.col_ptr_[i + 1] + self.col_ptr_[i]) / bsize

    def init_col_access(self, pkeep=1):
        if self.have_col_access():
            return
        self.init_col_data(pkeep)

    def row_iterator(self):
        self.iter_.before_first()
        return self.iter_

    def col_iterator(self, fset=None):
        if fset is None:
            ncol = self.num_col()
            print(ncol)
            resize(self.col_iter_.col_index_, ncol)
            for i in range(ncol):
                self.col_iter_.col_index_[i] = i
            self.col_iter_.set_batch(self.col_ptr_, self.col_data_)
        else:
            self.col_iter_.col_index_ = fset
            self.col_iter_.set_batch(self.col_ptr_, self.col_data_)
        return self.col_iter_

    def init_col_data(self, pkeep):
        self.buffered_rowset_.clear()
        builder = SparseCSRMBuilder(self.col_ptr_, self.col_data_)
        builder.init_budget(0)
        self.iter_.before_first()
        while self.iter_.next():
            batch = self.iter_.value()
            for i in range(batch.size):
                if pkeep == 0 or sample_binary(pkeep):
                    self.buffered_rowset_.append(batch.base_rowid + i)
                    inst = batch[i]
                    for j in range(inst.length):
                        builder.add_budget(inst[j].index)
        builder.init_storage()
        self.iter_.before_first()
        ktop = 0
        while self.iter_.next():
            batch = self.iter_.value()
            for i in range(batch.size):
                if (ktop < len(self.buffered_rowset_) and
                        self.buffered_rowset_[ktop] == batch.base_rowid + i):
                    ktop += 1
                    inst = batch[i]
                    for j in range(inst.length):
                        new_entry = SparseBatch.Entry(batch.base_rowid + i,
                                                      inst[j].fvalue)
                        builder.push_elem(inst[j].index, new_entry)
        ncol = self.num_col()
        for i in range(ncol):
            unsorted = self.col_data_[self.col_ptr_[i]:self.col_ptr_[i + 1]]
            unsorted.sort(key=lambda x: x.fvalue)

    class OneBatchIter(IIterator):
        def __init__(self):
            self.at_first_ = True
            self.batch_ = ColBatch()
            self.col_index_ = []
            self.col_data_ = []

        def before_first(self):
            self.at_first_ = True

        def next(self):
            if not self.at_first_:
                return False
            self.at_first_ = False
            return True

        def value(self):
            return self.batch_

        def set_batch(self, ptr, data):
            n_col_idx = len(self.col_index_)
            self.batch_.size = n_col_idx
            resize(self.col_data_, n_col_idx, SparseBatch.Inst(None, 0))
            for i in range(len(self.col_data_)):
                ridx = self.col_index_[i]
                x1 = ptr[ridx]
                x2 = ptr[ridx + 1]
                self.col_data_[i] = SparseBatch.Inst(data[x1:x2], x2-x1)
            self.batch_.col_index = self.col_index_
            self.batch_.col_data = self.col_data_
            self.before_first()


class DMatrixSimple(DataMatrix):
    """ xgboost.io  io/simple_dmatrix-inl.hpp"""
    kmagic = int('0xffffab01', 16)

    def __init__(self):
        super().__init__(DMatrixSimple.kmagic)
        self.row_ptr_ = [0]
        self.row_data_ = []
        self.fmat_ = FMatrixS(DMatrixSimple.OneBatchIter(self))
        self.clear()
        self.size = None

    def fmat(self):
        return self.fmat_

    def clear(self):
        self.row_ptr_.clear()
        self.row_ptr_.append(0)
        self.row_data_.clear()
        self.info.clear()

    def add_row(self, feats):
        for i in range(len(feats)):
            self.row_data_.append(feats[i])
            num_col = np.max(self.info.info.num_col, feats[i].index + 1)
            setattr(self.info.info, 'num_col', num_col)
        self.row_ptr_.append(self.row_ptr_[-1] + len(feats))
        setattr(self.info.info, 'num_row', self.info.info.num_row + 1)
        return len(self.row_ptr_) - 2

    class OneBatchIter(IIterator):
        def __init__(self, parent):
            self.at_first_ = True
            self.parent_ = parent
            self.batch_ = RowBatch()

        def before_first(self):
            self.at_first_ = True

        def next(self):
            if not self.at_first_:
                return False
            self.at_first_ = False
            self.batch_.ind_ptr = self.parent_.row_ptr_
            self.batch_.data_ptr = self.parent_.row_data_
            self.batch_.size = len(self.parent_.row_ptr_) - 1
            return True

        def value(self):
            return self.batch_


class DMatrix:
    """
    xgboost ..Dmatrix
    """

    def __init__(self, data, label=None, missing=0, weight=None):
        if data is None:
            self.handle = None
            return
        self.data = data.flatten()
        self.label = label
        self.weight = weight
        self.base_margin = None
        self.handle = DMatrix.create_from_mat(data, missing)
        if label is not None:
            self.set_label(label)
        if weight is not None:
            self.set_weight(weight)

    @staticmethod
    def create_from_mat(data, missing):
        nrow, ncol = data.shape
        data = data.flatten()
        mat = DMatrixSimple()
        mat.info.info.num_row = nrow
        mat.info.info.num_col = ncol
        col = 0
        for i in range(nrow):
            nelem = 0
            for j in range(ncol):
                if data[col + j] != missing:
                    new_entry = RowBatch.Entry(j, data[col + j])
                    mat.row_data_.append(new_entry)
                    nelem += 1
            mat.row_ptr_.append(mat.row_ptr_[-1] + nelem)
            col += ncol
        return mat

    # def set_param(self, name, val):
    #    setattr(self, name, val)
    #    return self

    def get_param(self, name):
        return getattr(self, name)

    def get_label(self):
        return self.label

    def set_label(self, label):
        self.handle.info.labels = label

    def get_base_margin(self):
        return self.base_margin

    def set_base_margin(self, margin):
        self.base_margin = margin

    def get_weight(self):
        return self.weight

    def set_weight(self, weight):
        self.weight = weight

    def num_row(self):
        return self.handle.info.num_row()

    def slice(self):
        res = DMatrix(None)


def slice_matrix(handle, lst, length):
    src = handle
    ret = DMatrixSimple()
    ret.clear()
    ret.info.info.num_row = length
    ret.info.info.num_col = src.info.num_col()

    iter = ret.fmat().row_iterator()
    iter.before_first()
    batch = iter.value()
    for i in range(length):
        ridx = lst[i]
        inst = batch[ridx]
        resize(ret.row_data_, len(ret.row_data_) + inst.length)


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
    # iter_i = mattt.handle.fmat().col_iterator()
    # print(mattt.handle.row_ptr_)
    # bb = DMatrixSimple()
    mattt.handle.fmat().init_col_access()
    iter_i = mattt.handle.fmat().col_iterator([1, 3])
    # iter.before_first()
    # batch = iter.value()
    print(0)
