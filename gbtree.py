import numpy as np
from utils.util import resize
from updaters.colmaker import ColMaker
from updaters.pruner import TreePruner
from updaters.refresher import TreeRefresher
from tree.tree import RegTree
from info_class import BoosterInfo


class GBTree:
    """
    xgboost.gbm : GBTREE gbm/gbtree-inl.hpp
    """

    def __init__(self):
        self.cfg = []
        self.mparam = self.ModelParam()
        self.tparam = self.TrainParam()
        self.pred_buffer = np.empty(0)
        self.pred_counter = np.empty(0)
        self.updaters = []
        self.trees = []
        self.tree_info = []
        self.thread_temp = []

    def set_param(self, name, val):
        if name[:4] == 'bst:':
            self.cfg.append((name[4:], val))
            for i in range(len(self.updaters)):
                self.updaters[i].set_param(name[4:], val)
        self.tparam.set_param(name, val)
        if len(self.trees) == 0:
            self.mparam.set_param(name, val)

    def init_model(self):
        self.pred_buffer = []
        self.pred_counter = []
        resize(self.pred_buffer, self.mparam.pred_buffer_size())
        resize(self.pred_counter, self.mparam.pred_buffer_size())

    def do_boost(self, p_fmat, info, gpair):
        #
        if self.mparam.num_output_group == 1:
            self.boost_new_trees(gpair, p_fmat, info, 0)
        else:
            # ngroup = number of classes in label
            ngroup = self.mparam.num_output_group
            nsize = len(gpair) / ngroup
            tmp = [None] * nsize
            for gid in range(ngroup):
                for i in range(nsize):
                    tmp[i] = gpair[i * ngroup + gid]
                self.boost_new_trees(tmp, p_fmat, info, gid)

    def predict(self, p_fmat, buffer_offset, info, ntree_limit=0):
        nthread = 1
        info = BoosterInfo()
        resize(self.thread_temp, nthread, RegTree.FVec())
        for i in range(nthread):
            self.thread_temp[i].init(self.mparam.num_feature)
        stride = info.num_row * self.mparam.num_output_group


    def clear(self):
        self.trees.clear()
        self.pred_buffer.clear()
        self.pred_counter.clear()

    def init_updater(self):
        if self.tparam.updater_initialized != 0:
            return
        self.updaters = []
        tval = self.tparam.updater_seq
        pstr = tval.split(',')
        for pstr_i in pstr:
            if pstr_i == 'prune':
                self.updaters.append(TreePruner())
            elif pstr_i == 'refresh':
                self.updaters.append(TreeRefresher())
            elif pstr_i == 'grow_colmaker':
                self.updaters.append(ColMaker())
            else:
                raise ValueError('updater should be ', 'prune', 'refresh',
                                 'grow_colmaker')
            for name, val in self.cfg:
                self.updaters[-1].set_param(name, val)
        self.tparam.updater_initialized = 1

    def boost_new_trees(self, gpair, p_fmat, info, bst_group):
        self.init_updater()
        # create tree
        new_trees = []
        for i in self.tparam.num_parallel_tree:
            new_trees.append(RegTree())
            for name, val in self.cfg:
                new_trees[-1].param.set_param(name, val)
            new_trees[-1].init_model()
        # update tree
        for i in range(len(self.updaters)):
            self.updaters[i].update(gpair, p_fmat, info, new_trees)

        # push back model
        for tree in new_trees:
            self.trees.append(tree)
            self.tree_info.append(bst_group)
        self.mparam.num_trees += self.tparam.num_parallel_tree

    def pred(self, inst, buffer_index, bst_group, root_index, p_feats,
             stride, ntree_limit):
        """ make a prediction for a single instance """
        itop = 0
        psum = 0
        p_feats = RegTree.FVec()
        vec_psum = [0] * self.mparam.size_leaf_vector
        bid = self.mparam.buffer_offset(buffer_index, bst_group)
        if ntree_limit == 0:
            treeleft = np.iinfo(np.uint32).max
        else:
            treeleft = ntree_limit
        # load buffered results if any
        if bid >= 0 and ntree_limit == 0:
            itop = self.pred_counter[bid]
            psum = self.pred_buffer[bid]
            for i in range(self.mparam.size_leaf_vector):
                vec_psum[i] = self.pred_buffer[bid + i + 1]
        if itop != len(self.trees):
            p_feats.fill(inst)
            for i in range(itop, len(self.trees)):
                if self.tree_info[i] == bst_group:
                    tid = self.trees[i].get_leaf_index(p_feats, root_index)
                    psum += self.trees[i][tid].leaf_value()
                    for j in range(self.mparam.size_leaf_vector):
                        vec_psum[j] += self.trees[i].leafvec(tid)[j]
                    treeleft -= 1
                    if treeleft == 0:
                        break
            p_feats.drop(inst)
        # updated the buffered results
        if bid >= 0 and ntree_limit == 0:
            self.pred_counter[bid] = len(self.trees)
            self.pred_buffer[bid] = psum
            for i in range(self.mparam.size_leaf_vector):
                self.pred_buffer[bid + i + 1] = vec_psum[i]

        out_pred[0] = psum
        for i in range(self.mparam.size_leaf_vector):


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
            self.reserved = np.zeros(31)

        def set_param(self, name, val):
            setattr(self, name, val)

        def pred_buffer_size(self):
            """ size of needed preduction buffer """
            return self.num_output_group * self.num_pbuffer * \
                   (self.size_leaf_vector + 1)

        def buffer_offset(self, buffer_index, bst_group):
            """ get the buffer offset given a buffer index and group id """
            if buffer_index < 0:
                return -1
            return (buffer_index + self.num_pbuffer * bst_group) * \
                   (self.size_leaf_vector + 1)
