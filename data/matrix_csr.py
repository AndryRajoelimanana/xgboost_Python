from utils.util import resize


class SparseCSRMBuilder:
    """
    utils/matrix_csr.h
    """
    def __init__(self, rptr=[], findex=[], dummy_aclist=[]):
        self.rptr = rptr
        self.findex = findex
        self.aclist = dummy_aclist
        self.use_aclist = False

    def init_budget(self, nrows=0):
        if not self.use_aclist:
            self.rptr.clear()
            resize(self.rptr, nrows + 1, 0)
        else:
            assert (nrows + 1) == len(self.rptr)
            self.cleanup()

    def cleanup(self):
        for ac in self.aclist:
            self.rptr[ac] = 0
            self.rptr[ac + 1] = 0
        self.aclist.clear()

    def add_budget(self, row_id, nelem=1):
        if len(self.rptr) < row_id + 2:
            resize(self.rptr, row_id + 2, 0)
        if self.use_aclist:
            if self.rptr[row_id+1] == 0:
                self.aclist.append(row_id)
        self.rptr[row_id+1] += nelem

    def init_storage(self):
        start = 0
        if not self.use_aclist:
            for i in range(1, len(self.rptr)):
                rlen = self.rptr[i]
                self.rptr[i] = start
                start += rlen
        else:
            aclist = self.aclist.sort()
            self.aclist.sort()
            for i in range(len(aclist)):
                ridx = aclist[i]
                rlen = self.rptr[ridx+1]
                self.rptr[ridx+1] = start
                if i == 0 or (ridx != aclist[i-1] + 1):
                    self.rptr[ridx] = start
                start += rlen
        resize(self.findex, start)

    def push_elem(self, row_id, col_id):
        rp = self.rptr[row_id + 1]
        # print(rp+1)
        self.findex[rp] = col_id
        self.rptr[row_id + 1] += 1

