import numpy as np


class GBTree:
    def __init__(self):
        self.mparam = ModelParam()
        self.tparam = TrainParam()
        self.pred_buffer = np.empty(0)
        self.pred_counter = np.empty(0)
        self.updaters = np.empty(0, dtype=object)

    def set_param(self, name, val):
        if name[:4] == 'bst:':
            cfg.append((name[4:], val))
            for i in range(self.updaters.size):
                self.updaters[i].set_param(name[4:], val)
        self.tparam.set_param(name, val)
        if len(self.trees) == 0:
            self.mparam.set_param(name, val)

    def init_model(self):
        self.pred_buffer = np.empty(0)
        self.pred_counter = np.empty(0)
        self.pred_buffer.resize(self.mparam.pred_buffer_size())
        self.pred_counter.resize(self.mparam.pred_buffer_size())

    def do_boost(self, p_fmat, info, gpair):
        if self.mparam.num_output_group == 1:
            self.boost_new_trees(gpair, p_fmat, info, 0)
        else:
            ngroup = self.mparam.num_output_group
            nsize = len(gpair)/ngroup
            tmp = np.empty(nsize, dtype=object)
            for gid in range(ngroup):
                for i in range(nsize):
                    tmp[i] = gpair[i * ngroup + gid]
                self.boost_new_trees(tmp, p_fmat, info, gid)

    # def predict(self, p_fmat, buffer_offset, info, ntree_limit=0):
    def init_updater(self):
        if self.tparam.updater_initialized !=0:
            return
        self.updaters = []
        tval = self.tparam.updater_seq
        pstr = tval.split(',')
        for pstr_i in pstr:
            self.updaters.append()


class TrainParam:
    def __init__(self):
        self.nthread = 0
        self.updater_seq = "grow_colmaker,prune"
        self.num_parallel_tree = 1
        self.updater_initialized = 0

    def set_param(self, name, val):
        setattr(self, name, val)


class ModelParam:
    def __init__(self):
        self.num_trees = 0
        self.num_roots = self.num_feature = 0
        self.num_pbuffer = 0
        self.num_output_group = 1
        self.size_leaf_vector = 0
        self.reserved= np.zeros(31)

    def set_param(self, name, val):
        setattr(self, name, val)

    def pred_buffer_size(self):
        return self.num_output_group * self.num_pbuffer * \
               (self.size_leaf_vector + 1)

    def buffer_offset(self, buffer_index, bst_group):
        if buffer_index < 0:
            return -1
        return (buffer_index + self.num_pbuffer * bst_group) * \
               (self.size_leaf_vector + 1);


