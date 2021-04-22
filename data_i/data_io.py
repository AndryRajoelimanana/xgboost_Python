# from utils.data_mat import D
# begin data_.h


class SparseBatch:
    def __init__(self):
        self.size = None

    class Entry:
        def __init__(self, index, fvalue):
            """ Entry of sparse vector"""
            self.index = index
            self.fvalue = fvalue

        def cmp_value(self, other):
            return self.fvalue < other.get_fvalue

    class Inst:
        def __init__(self, data, length):
            self.data = data
            self.length = length

        def __getitem__(self, item):
            return self.data[item]


class RowBatch(SparseBatch):
    def __init__(self):
        super().__init__()
        self.base_rowid = 0
        self.ind_ptr = []
        self.data_ptr = []
        self.size = 0

    def __getitem__(self, i):
        x1 = self.ind_ptr[i]
        x2 = self.ind_ptr[i+1]
        return self.Inst(self.data_ptr[x1:x2],  x2-x1)


class ColBatch(SparseBatch):
    def __init__(self, col_index=[], col_data=[]):
        super().__init__()
        self.col_index = col_index
        self.col_data = col_data

    def __getitem__(self, item):
        return self.col_data[item]

