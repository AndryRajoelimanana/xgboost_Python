from data_i.data import DMatrix, MetaInfo, CSCAdapter, SparsePage, CSRAdapter
import numpy as np


kAdapterUnknownSize = np.iinfo(int).max


class SimpleDMatrix(DMatrix):
    kMagic = 0xffffab01
    kPageSize = 32 << 12

    def __init__(self, adapter, missing=0, nthread=1):
        super().__init__()
        # adapter = CSRAdapter()
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

    # def get_row_batches(self):
    #     pass
    #
    # def get_column_batches(self):
    #     if not self.column_page_:
    #         self.column_page_ = CSCPage(self.sparse_page_.get_transpose(self.info_.num_col_))
    #     begin_iter = BatchIterator(
    #         SimpleBatchIteratorImpl(self.column_page_.get()))
    #     return BatchSet(begin_iter)
    #
    # def get_sorted_column_batches(self):
    #     if not self.sorted_column_page_:
    #         self.sorted_column_page_ = SortedCSCPage(self.sparse_page_.get_transpose(self.info_.num_col_))
    #         self.sorted_column_page_.sort_rows()
    #     begin_iter = BatchIterator(
    #         SimpleBatchIteratorImpl(self.column_page_.get()))
    #     return BatchSet(begin_iter)



