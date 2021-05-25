import numpy as np
from utils.util import resize
from enum import Enum
# from data_i.simple_dmatrix import SimpleDMatrix


# src/data_/data_.c

# namespace: xgboost

class DMat:
    def __init__(self, data, label=None, weight=None):
        xcsr = csr_matrix(data)




class FeatureType(Enum):
    kNumerical = 0
    kCategorical = 1


class DataType(Enum):
    kFloat32 = 1
    kDouble = 2
    kUInt32 = 3
    kUInt64 = 4
    kStr = 5


class BoosterInfo:
    def __init__(self, num_row=0, num_col=0):
        self.num_row = num_row
        self.num_col = num_col
        self.root_index = []
        self.fold_index = []

    def get_root(self, i):
        if len(self.root_index) == 0:
            return 0
        else:
            return self.root_index[i]


class MetaInfo:
    kNumField = 11

    def __init__(self, num_row=0, num_col=0, num_nonzero=0):
        self.info = BoosterInfo()
        self.num_row_ = num_row
        self.num_col_ = num_col
        self.num_nonzero_ = num_nonzero
        self.labels_ = self.group_ptr_ = self.weights_ = self.base_margin_ = []
        self.labels_lower_bound_ = self.labels_upper_bound_ = []
        self.feature_type_names = self.feature_names = self.feature_types = []
        self.feature_weigths = self.label_order_cache_ = []
        self.clear()

    def num_row(self):
        return self.info.num_row

    def num_col(self):
        return self.info.num_col

    def clear(self):
        self.num_row_ = self.num_col_ = self.num_nonzero_ = 0
        self.labels_.clear()
        self.group_ptr_.clear()
        self.weights_.clear()
        self.base_margin_.clear()

    def get_weight(self, i):
        if len(self.weights_) != 0:
            return self.weights_[i]
        else:
            return 1

    def label_abs_sort(self):
        if self.label_order_cache_ == len(self.labels_):
            return self.label_order_cache_
        self.label_order_cache_ = [i[0] for i in sorted(enumerate(
            self.labels_), key=lambda x: abs(x[1]))]
        return self.label_order_cache_

    def get_param(self, field):
        return getattr(self, field)

    def set_param(self, field, val):
        setattr(self, field, val)
        return self

    def set_info(self, key, val):
        if key == 'label':
            self.labels_ = val
        elif key == 'weight':
            self.weights_ = val
            assert np.all([w >= 0 for w in
                           self.weights_]), "Weights must be positive values."
        elif key == 'base_margin':
            self.base_margin_ = val
        elif key == 'group':
            self.group_ptr_ = np.cumsum(val)
        elif key == 'qid':
            is_sorted = lambda a: np.all(a[:-1] <= a[1:])
            assert is_sorted(val), "`qid` must be sorted in non-decreasing " \
                                   "order along with data. "
            self.group_ptr_ = list(np.unique(val))
            if self.group_ptr_[-1] != len(val):
                self.group_ptr_.append(len(val))
        elif key == 'label_lower_bound':
            self.labels_lower_bound_ = val
        elif key == 'label_upper_bound':
            self.labels_upper_bound_ = val
        elif key == 'feature_weights':
            assert np.all(val >= 0)
            self.feature_weigths = val
        else:
            raise NameError('Unknown key for MetaInfo: ')

    def get_info(self, key):
        if key == "label":
            return self.labels_
        elif key == "weight":
            return self.weights_
        elif key == "base_margin":
            return self.base_margin_
        elif key == "label_lower_bound":
            return self.labels_lower_bound_
        elif key == "label_upper_bound":
            return self.labels_upper_bound_
        elif key == "feature_weights":
            return self.feature_weigths
        elif key == "group_ptr":
            return self.group_ptr_
        else:
            raise NameError("Unknown float field name: {key}")

    def slice(self, ridxs):
        out = MetaInfo(len(ridxs), self.num_col_)
        out.labels_ = gather(self.labels_, ridxs)
        out.labels_upper_bound_ = gather(self.labels_upper_bound_, ridxs)
        out.labels_lower_bound_ = gather(self.labels_lower_bound_, ridxs)
        if len(self.weights_) + 1 == len(self.group_ptr_):
            h_weights = out.weights_
        else:
            out.weights_ = gather(self.weights_, ridxs)
        if len(self.base_margin_) != self.num_row_:
            assert len(self.base_margin_) % self.num_row_ == 0, "Incorrect " \
                                                                "size of base" \
                                                                " margin " \
                                                                "vector. "
            stride = len(self.base_margin_) / self.num_row_
            out.base_margin_ = gather(self.base_margin_, ridxs, stride)
        else:
            out.base_margin_ = gather(self.base_margin_, ridxs)

        out.feature_weigths = self.feature_weigths
        out.feature_names = self.feature_names
        out.feature_types = self.feature_types
        out.feature_type_names = self.feature_type_names
        return out


def gather(ins, ridxs, stride=1):
    if len(ins) == 0:
        return []
    size = len(ridxs)
    out = [] * size * stride
    for i in range(size):
        ridx = ridxs[i]
        for j in range(stride):
            out[i * stride + j] = ins[ridx * stride + j]
    return out


class Entry:
    def __init__(self, index, fvalue):
        """ Entry of sparse vector"""
        self.index = index
        self.fvalue = fvalue

    @staticmethod
    def cmp_value(a, b):
        return a.fvalue < b.fvalue

    def __eq__(self, other):
        return (self.index == other.index) and (self.fvalue == other.fvalue)


class BatchParam:
    def __init__(self, device=0, max_bin=0, gpu_page_size=0):
        self.gpu_id = device
        self.max_bin = max_bin
        self.gpu_page_size = gpu_page_size

    def __ne__(self, other):
        return (self.gpu_id != other.gpu_id) or (
                self.max_bin != other.max_bin) or (
                       self.gpu_page_size != other.gpu_page_size)


class HostSparsePageView:
    def __init__(self, offset=None, data=None):
        self.offset = offset if offset is not None else [0]
        self.data = data if data is not None else []

    def __getitem__(self, item):
        return self.data[self.offset[item]:self.offset[item + 1]]

    def Size(self):
        return 0 if len(self.offset) == 0 else len(self.offset) - 1


class SparsePage:
    def __init__(self, offset=None, data=None):
        self.offset = offset if offset else [0]
        self.data = data if data else []
        self.base_rowid = 0

    def get_view(self):
        return HostSparsePageView(self.offset, self.data)

    def size(self):
        return 0 if len(self.offset) == 0 else len(self.offset) - 1

    def clear(self):
        self.base_rowid = 0
        self.offset = [0]
        self.data = []

    def set_base_row_id(self, row_id):
        self.base_rowid = row_id

    def get_transpose(self, num_columns):
        page = self.get_view()
        builder = ParallelGroupBuilder([],[])
        builder.init_budget(num_columns, 1)
        for i in range(self.size()):
            tid = 0
            inst = page[i]
            for entry in inst:
                builder.add_budget(entry.index, tid)
        builder.init_storage()
        for i in range(self.size()):
            tid = 0
            inst = page[i]
            for entry in inst:
                builder.push(entry.index, Entry(self.base_rowid+i,
                                                entry.fvalue), tid)
        return SparsePage(builder.rptr_, builder.data_)

    def sort_row(self):
        ncol = self.size()
        for i in range(ncol):
            ndata = self.data[self.offset[i]:self.offset[i + 1]]
            sdata = sorted(ndata, key=lambda x: x.fvalue)
            self.data[self.offset[i]:self.offset[i + 1]] = sdata

    def push(self, batch, missing=None, nthread=None):
        max_columns_local = 0
        if isinstance(batch, SparsePage):
            self.data += batch.data
            top = self.offset[-1]
            self.offset += [i + top for i in batch.offset]
        elif isinstance(batch, CSRAdapterBatch):
            for i in range(batch.num_rows_):
                line = batch.getline(i)
                for j in range(line.size()):
                    element = line.get_element(j)
                    max_columns_local = max(max_columns_local, element.column_idx)
                    self.data.append(Entry(element.column_idx, element.value))
            self.offset = batch.row_ptr_
        return max_columns_local

    def pushCSC(self, batch):
        if batch.data.empty():
            return
        if len(self.data) == 0:
            self.data = batch.data
            self.offset = batch.offset
            return
        offset = [0] * len(batch.offset)
        data = [Entry() for _ in range(len(self.data) + len(batch.data))]
        n_features = len(batch.offset) - 1
        beg = 0
        ptr = 1
        for i in range(n_features):
            off_i = self.offset[i]
            off_i1 = self.offset[i + 1]
            length = off_i1 - off_i
            data[beg:beg + length] = self.data[off_i:off_i1]
            beg += length
            off_i = batch.offset[i]
            off_i1 = batch.offset[i + 1]
            length = off_i1 - off_i
            data[beg:beg + length] = batch.data[off_i:off_i1]
            beg += length
            assert len(offset) > 1
            offset[ptr] = beg
            ptr += 1
        self.data = data
        self.offset = offset


class SortedCSCPage(SparsePage):

    def __init__(self, page=None):
        if page is not None:
            offs = page.offset
            page.sort_row()
            dat = page.data
        else:
            offs = dat = None
        super(SortedCSCPage, self).__init__(offs, dat)


class ParallelGroupBuilder:
    def __init__(self, p_rptr, p_data, base_row_offset = 0):
        self.rptr_ = p_rptr
        self.data_ = p_data
        self.base_row_offset_ = base_row_offset

    def init_budget(self, max_key, nthread):
        self.thread_rptr_ = [[] for _ in range(nthread)]
        for i in range(nthread):
            self.thread_rptr_[i] = [0]*max(max_key -
                                           min(max_key,
                                               self.base_row_offset_), 0)

    def add_budget(self, key, threadid, nelem=1):
        trptr = self.thread_rptr_[threadid]
        offset_key = key - self.base_row_offset_
        if len(trptr) < offset_key + 1:
            resize(self.thread_rptr_[threadid], offset_key+1, 0)
        self.thread_rptr_[threadid][offset_key] += nelem

    def init_storage(self):
        rptr_fill_value = 0 if len(self.rptr_)==0 else self.rptr_[-1]
        for tid in range(len(self.thread_rptr_)):
            if len(self.rptr_) <= len(self.thread_rptr_[tid]) + \
                    self.base_row_offset_:
                resize(self.rptr_, len(self.thread_rptr_[tid]) +
                       self.base_row_offset_ + 1, rptr_fill_value)
        count = 0
        for i in range(self.base_row_offset_, len(self.rptr_)-1):
            for tid in range(len(self.thread_rptr_)):
                trptr = self.thread_rptr_[tid]
                if i < len(trptr) + self.base_row_offset_:
                    thread_count = trptr[i-self.base_row_offset_]
                    self.thread_rptr_[tid][i-self.base_row_offset_] = count +\
                                                                      self.rptr_[-1]
                    count += thread_count
            self.rptr_[i+1] += count
        resize(self.data_, self.rptr_[-1])

    def push(self, key, value, threadid):
        offset_key = key - self.base_row_offset_
        rp = self.thread_rptr_[threadid][offset_key]
        self.data_[rp] = value
        self.thread_rptr_[threadid][offset_key] += 1


class CSCPage(SparsePage):
    def __init__(self, page):
        super().__init__()
        self.page = page


class BatchIteratorImpl:
    pass


class BatchIterator:
    def __init__(self, impl):
        self.impl_ = impl

    def __ne__(self, other):
        return not self.impl_.at_end()

    def at_end(self):
        return self.impl_.at_end()


class BatchSet:
    def __init__(self, begin_iter):
        self.begin_iter_ = begin_iter

    def begin(self):
        return self.begin_iter_

    def end(self):
        return None


class COOTuple:
    def __init__(self, row_idx=0, column_idx=0, value=0):
        self.row_idx = row_idx
        self.column_idx = column_idx
        self.value = value


class CSRAdapterBatch:
    def __init__(self, row_ptr, feature_idx, values, num_rows):
        self.row_ptr_ = row_ptr
        self.feature_idx_ = feature_idx
        self.num_rows_ = num_rows
        self.values_ = values

    def size(self):
        return self.num_rows_

    def getline(self, idx):
        beg = self.row_ptr_[idx]
        end = self.row_ptr_[idx + 1]
        return CSRAdapterBatch.Line(idx, end - beg, self.feature_idx_[beg:end],
                                    self.values_[beg:end])

    class Line:
        def __init__(self, row_idx, size, feature_idx, values):
            self.row_idx_ = row_idx
            self.values_ = values
            self.feature_idx_ = feature_idx
            self.size_ = size

        def size(self):
            return self.size_

        def get_element(self, idx):
            return COOTuple(self.row_idx_, self.feature_idx_[idx], self.values_[
                idx])


class CSRAdapter:
    def __init__(self, row_ptr, feature_idx, values, num_rows, num_elements,
                 num_features):
        self.batch_ = CSRAdapterBatch(row_ptr, feature_idx, values, num_rows)
        self.num_rows_ = num_rows
        self.num_columns_ = num_features

    def value(self):
        return self.batch_

    def num_rows(self):
        return self.num_rows_

    def num_columns(self):
        return self.num_columns_


class CSRArrayAdapterBatch:
    def __init__(self, indptr, indices, values):
        self.indptr_ = indptr
        self.indices_ = indices
        self.values_ = values

    class Line:
        def __init__(self, indices, values, ridx):
            self.ridx_ = ridx
            self.indices_ = indices
            self.values_ = values

        def get_element(self, idx):
            return COOTuple(self.ridx_, self.indices_[idx], self.values_[idx])

        def size(self):
            return


class CSCAdapter:
    def __init__(self, data):
        self.batch_ = data
        r, c = data.shape
        self.indptr_ = data.indptr
        self.indices_ = data.indices
        self.values_ = data.data

        self.num_rows_ = r
        self.num_cols_ = c

    def value(self):
        return self.batch_.data

    def num_rows(self):
        return self.num_rows_

    def num_columns(self):
        return self.num_cols_


class DMatrix:
    def __init__(self, csr_mat, label=None,
                 weight=None,
                 base_margin=None,
                 missing=None,
                 silent=False,
                 feature_names=None,
                 feature_types=None,
                 nthread=None,
                 group=None,
                 qid=None,
                 label_lower_bound=None,
                 label_upper_bound=None,
                 feature_weights=None,
                 enable_categorical=False):
        self.data = csr_mat
        self.missing = missing if missing is not None else np.nan
        self.nthread = nthread if nthread is not None else -1
        self.handle = self.create()
        self.silent = silent
        self.info_ = MetaInfo()
        self.params = {'label': label, 'weight': weight,
                       'base_margin': base_margin,
                       'group': group, 'qid': qid,
                       'label_lower_bound': label_lower_bound,
                       'label_upper_bound': label_upper_bound,
                       'feature_names': feature_names,
                       'feature_types': feature_types,
                       'feature_weights': feature_weights}

    def info(self):
        return self.info_

    def set_info(self):
        for k, v in self.params.items():
            if v:
                self.handle.info().set_info(k, v)

    def GetThreadLocal(self):
        pass

    def get_batches_sp(self, param={}):
        return self.get_row_batches()

    def get_batches_csc(self, param={}):
        return self.get_column_batches()

    def get_row_batches(self):
        pass

    def get_column_batches(self):
        pass

    def page_exists(self):
        return

    def SingleColBlock(self):
        pass

    def is_dense(self):
        return self.info().num_nonzero_ == self.info().num_row_ * self.info().num_col_

    def create(self):
        x_csr = self.data
        row_ptr = x_csr.indptr
        feature_idx = x_csr.indices
        values = x_csr.data
        num_rows, num_features = x_csr.shape
        num_elements = x_csr.nnz

        adapter = CSRAdapter(row_ptr, feature_idx, values, num_rows,
                            num_elements, num_features)
        return SimpleDMatrix(adapter, self.missing, self.nthread)

    def load_binary(self, buffer):
        pass


class GetBuffer:
    def __init__(self, buf):
        self.buf = buf.tobytes()
        self.datatype = {1: np.float32, 2: np.double, 3: np.uint32,
                         4: np.uint64, 5: str}

    def get_b(self, beg, dtypes):
        pass
        # return self.buf[beg:beg + size].decode('utf-8'), beg + size

    def get_int(self, beg, size=4, gg='little'):
        return int.from_bytes(self.buf[beg:beg + size], gg), beg + size

    def get_scalar(self, start_idx, dtypes):
        s_dtype = np.zeros(1, dtype=dtypes).itemsize
        return np.frombuffer(self.buf[start_idx:start_idx + s_dtype],
                             dtype=dtypes)[0]

    def get_vector(self, start_idx, dtypes, nrow):
        s_dtype = np.zeros(1, dtype=dtypes).itemsize
        if nrow == 0:
            return np.zeros(0, dtype=dtypes)
        data = np.frombuffer(self.buf[start_idx:start_idx + s_dtype * nrow],
                             dtype=dtypes)
        return data

    def get_value(self, name):
        start = self.buf.find(name)
        end_name = start + len(name)
        names = self.buf[start:end_name]
        dtypes_int = self.buf[end_name]
        dtypes = self.datatype[dtypes_int]
        is_scalar = self.buf[end_name + 1] == 1
        if is_scalar:
            v = self.get_scalar(end_name + 2, dtypes)
        else:
            nrow = int.from_bytes(self.buf[end_name + 2:end_name + 10],
                                  'little')
            v = self.get_vector(end_name + 18, dtypes, nrow)
        return names, v

    def to_dict(self):
        key_list = [b'num_row', b'num_col', b'labels']
        dict_dm = {}
        for k in key_list:
            nn, v = self.get_value(k)
            dict_dm[k] = v
        return dict_dm


kAdapterUnknownSize = np.iinfo(int).max


class SimpleDMatrix(DMatrix):
    kMagic = 0xffffab01
    kPageSize = 32 << 12

    def __init__(self, adapter, missing=0, nthread=1):
        # super().__init__(None)
        self.adapter = adapter
        self.sparse_page_ = SparsePage()
        self.info_ = MetaInfo()

        nthread_original = 1
        qids = []
        default_max = np.iinfo(np.uint()).max
        last_group_id = default_max
        group_size = 0

        offset_vec = self.sparse_page_.offset
        data_vec = self.sparse_page_.data
        inferred_num_columns = 0
        total_batch_size = 0

        batch = adapter.value()
        batch_max_columns = self.sparse_page_.push(batch)
        inferred_num_columns = max(batch_max_columns, inferred_num_columns)
        total_batch_size += batch.size()

        if adapter.num_columns() == kAdapterUnknownSize:
            self.info_.num_col_ = inferred_num_columns
        else:
            self.info_.num_col_ = adapter.num_columns()

        if adapter.num_rows() == kAdapterUnknownSize:
            pass
        else:
            if len(self.sparse_page_.offset) == 0:
                self.sparse_page_.offset = [0]
            while len(self.sparse_page_.offset) - 1 < adapter.num_rows():
                self.sparse_page_.offset.append(self.sparse_page_.offset[-1])
            self.info_.num_row_ = adapter.num_rows()

        self.info_.num_nonzero_ = len(self.sparse_page_.data)

    def info(self):
        return self.info_

    def single_col_block(self):
        return True

    def sparse_page_exists(self):
        return True

    def slice(self, ridxs):
        out = SimpleDMatrix()
        out_page = out.sparse_page_
        for page in self.get_batches_sp():
            batch = page.GetView()
            h_data = out_page.data
            h_offset = out_page.offset
            rptr = 0
            for ridx in ridxs:
                inst = batch[ridx]
                rptr += len(inst)
                h_data += inst
                h_offset += rptr
            out.info_ = self.info().slice(ridxs)
            out.info_.num_nonzero_ = h_offset[-1]
        return out


if __name__ == "__main__":
    offset = [0, 3, 5, 6, 8]
    index = [0, 0, 0, 1, 1, 2, 3, 3]
    data = [3, 1, -4, 1, 3, 1, 1, 5]
    pp = []
    for i in range(8):
        pp.append(Entry(index[i], data[i]))
    ncol = len(offset)
    for i in range(ncol - 1):
        ndata = pp[offset[i]:offset[i + 1]]
        pp[offset[i]:offset[i + 1]] = sorted(ndata, key=lambda x: x.fvalue)
    for i in range(8):
        print(pp[i].fvalue)

    from sklearn.datasets import load_boston
    from scipy.sparse import csr_matrix

    boston = load_boston()
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

    nnn = DMatrix(x_csr).create()
    page = nnn.sparse_page_.get_transpose(nnn.info().num_col_)
    sorted_csc = SortedCSCPage(page)
    print(0)

