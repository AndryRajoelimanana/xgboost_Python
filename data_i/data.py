import numpy as np
from utils.util import resize
from enum import Enum


# src/data_/data_.c

# namespace: xgboost


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

    def __init__(self, num_row=0, num_col=0):
        self.info = BoosterInfo()
        self.num_row_ = num_row
        self.num_col_ = num_col
        self.num_nonzero_ = 0
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
        self.labels_.clear()
        self.group_ptr_.clear()
        self.weights_.clear()
        self.base_margin_.clear()
        self.info = BoosterInfo(0, 0)

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
        return a.get_fvalue < b.get_fvalue

    def __eq__(self, other):
        return (self.index == other.index) and (self.fvalue == other.get_fvalue)


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
    def __init__(self):
        self.offset = []
        self.data = []

    def __getitem__(self, item):
        return self.data[self.offset[item]:self.offset[item + 1]]

    def Size(self):
        return 0 if len(self.offset) == 0 else len(self.offset) - 1


class SparsePage:
    def __init__(self):
        self.offset = []
        self.data = []
        self.base_rowid = 0

    def get_view(self):
        return self.offset, self.data

    def Size(self):
        return 0 if len(self.offset) == 0 else len(self.offset) - 1

    def clear(self):
        self.base_rowid = 0
        self.offset = [0]
        self.data = []

    def set_base_row_id(self, row_id):
        self.base_rowid = row_id

    def get_transpose(self, num_columns):
        pass

    def sort_row(self):
        ncol = self.Size()
        for i in range(ncol - 1):
            ndata = self.data[self.offset[i]:self.offset[i + 1]]
            sdata = sorted(ndata, key=lambda x: x.fvalue)
            self.data[self.offset[i]:self.offset[i + 1]] = sdata

    def push(self, batch):
        self.data += batch.data
        top = self.offset[-1]
        self.offset += [i + top for i in batch.offset]

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


class CSCPage(SparsePage):
    def __init__(self, page):
        super().__init__()
        self.page = page


class SortedCSCPage(SparsePage):
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


class DMatrix:
    def __init__(self, begin_iter):
        pass

    def info(self):
        pass

    def set_info(self, key, val):
        self.info().set_info(key, val)

    def GetThreadLocal(self):
        pass

    def GetBatches(self, param={}):
        return

    def SingleColBlock(self):
        pass

    def is_dense(self):
        return self.info().num_non_zero_ == self.info().num_row_ * self.info().num_col_

    def create(self, iters, proxy, reset, next, missing, nthread, max_bin):
        pass

    def load_binary(self, buffer):
        pass


class GetBuffer:
    def __init__(self, buf):
        self.buf = buf.tobytes()
        self.datatype = {1: np.float32, 2: np.double, 3: np.uint32,
                         4: np.uint64, 5: str}

    def get_b(self, beg, dtypes):
        return self.buf[beg:beg + size].decode('utf-8'), beg + size

    def get_int(self, beg, size=4, gg='little'):
        return int.from_bytes(self.buf[beg:beg + size], gg), beg + size

    def get_scalar(self, start_idx, dtypes):
        s_dtype = np.zeros(1, dtype=dtypes).itemsize
        return np.frombuffer(self.buf[start_idx:start_idx+s_dtype],
                             dtype=dtypes)[0]

    def get_vector(self, start_idx, dtypes, nrow):
        s_dtype = np.zeros(1, dtype=dtypes).itemsize
        if nrow == 0:
            return np.zeros(0, dtype=dtypes)
        data = np.frombuffer(self.buf[start_idx:start_idx+s_dtype*nrow],
                             dtype=dtypes)
        return data

    def get_value(self, name):
        start = self.buf.find(name)
        end_name = start + len(name)
        names = self.buf[start:end_name]
        dtypes_int = self.buf[end_name]
        dtypes = self.datatype[dtypes_int]
        is_scalar = self.buf[end_name+1] == 1
        if is_scalar:
            v = self.get_scalar(end_name+2, dtypes)
        else:
            nrow = int.from_bytes(self.buf[end_name + 2:end_name + 10],
                                  'little')
            v = self.get_vector(end_name+18,  dtypes, nrow)
        return names, v

    def to_dict(self):
        key_list = [b'num_row', b'num_col', b'labels']
        dict_dm = {}
        for k in key_list:
            nn, v = self.get_value(k)
            dict_dm[k] = v
        return dict_dm


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
