from data_i.data import DMatrix


class SimpleDMatrix(DMatrix):
    kMagic = 0xffffab01;

    def __init__(self, adapter, missing, nthread):
        self.info_ = MetaInfo()
        self.column_page_ = CSCPage()
        self.sorted_column_page_ = SortedCSCPage()
        self.sparse_page_ = SparsePage()
        self.ellpack_page_ = EllpackPage()
        self.batch_param_ = BatchParam()

    def SparsePageExists(self):
        return True

    def info(self):
        return self.info_

    def slice(self, ridxs):
        out = SimpleDMatrix()
        out_page = out.sparse_page_
        for page in self.get_batches():
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

    def get_row_batches(self):
        begin_iter = BatchIterator(SimpleBatchIteratorImpl(self.sparse_page_))
        return BatchSet(begin_iter)

    def get_column_batches(self):
        if not self.column_page_:
            self.column_page_ = CSCPage(self.sparse_page_.get_transpose(self.info_.num_col_))
        begin_iter = BatchIterator(
            SimpleBatchIteratorImpl(self.column_page_.get()))
        return BatchSet(begin_iter)

    def get_sorted_column_batches(self):
        if not self.sorted_column_page_:
            self.sorted_column_page_ = SortedCSCPage(self.sparse_page_.get_transpose(self.info_.num_col_))
            self.sorted_column_page_.sort_rows()
        begin_iter = BatchIterator(
            SimpleBatchIteratorImpl(self.column_page_.get()))
        return BatchSet(begin_iter)



