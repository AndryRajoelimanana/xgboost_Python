from gbm.gbms import GradientBooster
# from tree.gbtree import GBTreeTrainParam, TreeProcessType
from gbm.gbtree_model import GBTreeModel, GBTreeModelParam
from enum import Enum
from utils.util import resize
from tree.tree import RegTree
from predictor.predictors import CPUPredictor


class TreeMethod:
    kAuto = 0
    kApprox = 1
    kExact = 2
    kHist = 3
    kGPUHist = 5


class TreeProcessType:
    kDefault = 0
    kUpdate = 1


class PredictorType:
    kAuto = 0
    kCPUPredictor = 1
    kGPUPredictor = 2
    kOneAPIPredictor = 3


class GBTreeTrainParam:
    def __init__(self, process_type=0, predictor=0, tree_method=0):
        self.nthread = 0
        self.updater_seq = None
        self.num_parallel_tree = 1
        self.updater_initialized = 0
        self.process_type = process_type
        self.predictor = predictor
        self.tree_method = tree_method
        self.initialised_ = False

    def set_param(self, name, val):
        if name == 'updater' and val not in self.updater_seq:
            self.updater_seq = [val]
            self.updater_initialized = 0
        elif name == 'num_parallel_tree':
            self.num_parallel_tree = val

    def get_initialised(self):
        return self.initialised_


class DartTrainParam:
    def __init__(self, sample_type=0, normalize_type=0, rate_drop=0,
                 one_drop=False, skip_drop=0.0, learning_rate=0.3):
        self.sample_type = sample_type
        self.normalize_type = normalize_type
        self.rate_drop = rate_drop
        self.one_drop = one_drop
        self.skip_drop = skip_drop
        self.learning_rate = learning_rate
        self.eta = self.learning_rate


def layer_to_tree(model, tparam, layer_begin, layer_end):
    groups = model.learner_model_param.num_output_group
    tree_begin = layer_begin * groups * tparam.num_parallel_tree
    tree_end = layer_end * groups * tparam.num_parallel_tree
    if tree_end == 0:
        tree_end = len(model.trees)
    if len(model.trees) != 0:
        assert tree_begin <= tree_end
    return tree_begin, tree_end


def slice_trees(layer_begin, layer_end, step, model, tparam, layer_trees, fn):
    tree_begin, tree_end = layer_to_tree(model, tparam, layer_begin, layer_end)
    if tree_end > len(model.trees):
        return True
    layer_end = len(model.trees)/layer_trees if layer_end == 0 else layer_end
    n_layers = (layer_end - layer_begin) / step
    in_it = tree_begin
    out_it = 0
    for l in range(n_layers):
        for i in range(layer_trees):
            assert in_it < tree_end
            fn(in_it, out_it)
            out_it += 1
            in_it += 1
        in_it += (step - 1) * layer_trees
    return False


class GBTree(GradientBooster):
    """
    xgboost.gbm : GBTREE gbm/gbtree-inl.hpp
    """

    def __init__(self, booster_config):
        super().__init__()
        self.model_ = GBTreeModel(booster_config)
        self.cfg_ = {}
        self.tparam_ = GBTreeTrainParam()
        self.dparam_ = DartTrainParam()
        self.pred_buffer = []
        self.pred_counter = []
        self.updaters_ = []
        self.trees = []
        self.tree_info = []
        self.thread_temp = []
        self.cpu_predictor_ = CPUPredictor()
        self.specified_updater_ = False
        self.configured_ = False

    def configure(self, cfg):
        self.cfg_ = cfg
        updater_seq = self.tparam_.updater_seq
        for k, v in cfg.items():
            if hasattr(self.tparam_, k):
                setattr(self.tparam_, k, v)
        self.model_.configure(cfg)
        if self.tparam_.process_type == TreeProcessType.kUpdate:
            self.model_.init_trees_to_update()
        if self.cpu_predictor_ is None:
            self.cpu_predictor_ = CPUPredictor()
        self.configured_ = True
        self.specified_updater_ = 'updater' in cfg.keys()
        self.configure_updaters()
        if updater_seq != self.tparam_.updater_seq:
            self.updaters_ = []
            self.init_updater()
        else:
            for i in range(len(self.updaters_)):
                self.updaters_[i].configure(cfg)
        self.configured_ = True

    def get_train_param(self):
        return self.tparam_

    def layer_trees(self):
        """ Numbser trees per layer"""
        n_group = self.model_.learner_model_param.num_output_group
        n_parallel_tree = self.tparam_.num_parallel_tree
        n_trees = n_group * n_parallel_tree
        return n_trees

    def boosted_rounds(self):
        assert self.tparam_.num_parallel_tree != 0
        assert self.model_.learner_model_param.num_output_group != 0
        return len(self.model_.trees)/self.layer_trees()

    def slice(self, layer_begin, layer_end, step, out, out_of_bound):
        layer_trees = self.layer_trees()
        if layer_end == 0:
            layer_end = self.boosted_rounds()
        n_layers = (layer_end - layer_begin) / step
        out_trees = out.model_.trees
        resize(out_trees, layer_trees * n_layers, GBTreeModel())
        out_trees_info = out.model_.tree_info
        resize(out_trees_info, layer_trees * n_layers, 0)
        out.model_.param.num_trees = len(out.model_.trees)

    def get_predictor(self):
        return self.cpu_predictor_

    def do_boost(self, p_fmat, gpair):
        """ Boost one iteration gpair of size [n_sample, 2, n_group]"""
        new_trees = []
        ngroup = self.model_.learner_model_param.num_output_group
        self.configure_with_known_data(self.cfg_, p_fmat)
        assert ngroup != 0
        if ngroup == 1:
            new_tree = self.boost_new_trees(gpair[:, 0], gpair[:, 1],
                                            p_fmat)
            new_trees += [new_tree]
        else:
            for gid in range(ngroup):
                new_tree = self.boost_new_trees(gpair[:, 0, gid],
                                                gpair[:, 1, gid],
                                                p_fmat)
                new_trees += [new_tree]
        self.commit_model(new_trees)
        return new_trees

    def boost_new_trees(self, grad, hess, p_fmat):
        new_trees = []
        for i in range(self.tparam_.num_parallel_tree):
            tree = RegTree()
            for k, v in self.cfg_.items():
                setattr(tree.param, k, v)
            for up in self.updaters_:
                up.update(grad, hess, p_fmat, tree)
            new_trees.append(tree)
        return new_trees

    def inplace_predict(self):
        pass

    def commit_model(self, new_trees):
        ngroup = self.model_.learner_model_param.num_output_group
        for gid in range(ngroup):
            self.model_.commit_model(new_trees[gid], gid)

    def predict_batch(self, dmat, bools, layer_begin, layer_end):
        predictor = self.get_predictor()
        tree_begin, tree_end = self.layer_to_tree(layer_begin, layer_end)

        preds = predictor.predict_batch(dmat, self.model_, tree_begin,
                                        tree_end)
        return preds

    def predict_leaf(self, dmat, layer_begin, layer_end):
        predictor = self.get_predictor()
        tree_begin, tree_end = self.layer_to_tree(layer_begin, layer_end)
        preds = predictor.predict_leaf(dmat, self.model_, tree_begin,
                                       tree_end)
        return preds

    def predict_contribution(self, data, layer_begin, layer_end, approximate):
        tree_begin, tree_end = self.layer_to_tree(layer_begin, layer_end)
        mssg_error = "Predict contribution supports only iteration end:"
        assert tree_begin == 0, mssg_error
        return self.get_predictor().predict_contribution(data, self.model_,
                                                         tree_end, None,
                                                         approximate)

    def configure_updaters(self):
        if self.specified_updater_:
            return
        if self.tparam_.tree_method == TreeMethod.kAuto:
            return
        elif self.tparam_.tree_method == TreeMethod.kApprox:
            self.tparam_.updater_seq = "grow_histmaker,prune"
        elif self.tparam_.tree_method == TreeMethod.kExact:
            self.tparam_.updater_seq = "grow_colmaker"
        elif self.tparam_.tree_method == TreeMethod.kHist:
            self.tparam_.updater_seq = "grow_quantile_histmaker"
        elif self.tparam_.tree_method == TreeMethod.kGPUHist:
            self.tparam_.updater_seq = "grow_gpu_hist"
        else:
            print(self.tparam_.tree_method)
            raise Exception("Unknown Tree Method")

    def configure_with_known_data(self, cfg, fmat):
        assert self.configured_
        updater_seq = self.tparam_.updater_seq
        assert self.tparam_.initialised_
        for k, v in cfg:
            if hasattr(self.tparam_, k):
                setattr(self.tparam_, k, v)
        self.configure_updaters()
        if updater_seq != self.tparam_.updater_seq:
            self.updaters_ = []
            self.init_updater(cfg)

    def init_updater(self):
        tval = self.tparam_.updater_seq
        ups = tval.split(',')
        if len(self.updaters_) != 0:
            assert len(ups) == len(self.updaters_)
            for up in self.updaters_:
                contains = up in ups
                if not contains:
                    raise Exception('Internal Error: mismatched '
                                    'updater sequence')
            return

    def layer_to_tree(self, layer_begin, layer_end):
        model = self.model_
        tparam = self.tparam_
        return layer_to_tree(model, tparam, layer_begin, layer_end)


class Dart(GBTree):
    def __init__(self, booster_config):
        super().__init__(booster_config)
        self.idx_drop_ = []
        self.weight_drop_ = []

    def configure(self, cfg):
        pass

    def commit_model(self, new_trees, dmat, pred):
        ngroup = self.model_.learner_model_param.num_output_group
        num_new_trees = 0
        for gid in range(ngroup):
            num_new_trees += len(new_trees[gid])
            self.model_.commit_model(new_trees[gid], gid)
        num_drop = self.normalize_tree(num_new_trees)
        print(f'drop {num_drop} trees, weight = {self.weight_drop_[-1]}')

    def normalize_tree(self, size_new_trees):
        """ used for droupout tree, if no dropout weight_drop = 1.0"""
        lr = self.dparam_.learning_rate / size_new_trees
        num_drop = len(self.idx_drop_)
        if num_drop == 0:
            for i in range(size_new_trees):
                self.weight_drop_.append(1.0)
        else:
            if self.dparam_.normalize_type == 1:
                factor = 1.0 / (1.0 + lr)
                for i in self.idx_drop_:
                    self.weight_drop_[i] *= factor
                for i in range(size_new_trees):
                    self.weight_drop_.append(factor)
            else:
                factor = 1.0 * num_drop / (num_drop + lr)
                for i in self.idx_drop_:
                    self.weight_drop_[i] *= factor
                for i in range(size_new_trees):
                    self.weight_drop_.append(1.0 / (num_drop + lr))

        self.idx_drop_ = []
        return num_drop


    #
    #
    # def configure(self, cfg):
    #     self.cfg_ = cfg
    #     updater_seq = self.tparam_.updater_seq
    #     for k, v in cfg:
    #         setattr(self.tparam_, k, v)
    #     self.model_.configure(cfg)
    #     if self.tparam_.process_type == TreeProcessType.kUpdate:
    #         self.model_.init_trees_to_update()
    #     self.configured_ = True
    #     self.specified_updater_ = 'updater' in cfg.keys()
    #     self.configure_updaters()
    #     if updater_seq != self.tparam_.updater_seq:
    #         self.updaters_ = []
    #         self.init_updater(cfg)


#
#
#
# class GBTree(GradientBooster):
#     def __init__(self, booster_config):
#         # TODO
#         self.model_ = GBTreeModel()
#         # end to do
#         self.tparam_ = GBTreeTrainParam()
#         self.showed_updater_warning_ = False
#         self.specified_updater_ = False
#         self.configured_ = False
#         self.updaters_ = []
#         self.cpu_predictor_ = None
#         self.cfg_ = None
#
#     def configure(self, cfg):
#         self.cfg_ = cfg
#         updater_seq = self.tparam_.updater_seq
#         for k, v in cfg:
#             setattr(self.tparam_, k, v)
#         self.model_.configure(cfg)
#         if self.tparam_.process_type == TreeProcessType.kUpdate:
#             self.model_.init_trees_to_update()
#
#     def do_boost(self, p_fmat, in_gpair, predt):
#         new_trees = []
#         ngroup = self.model_.learner_model_param.num_output_group
#

#
#     def perform_tree_method_heuristic(self, fmat):
#         if self.specified_updater_:
#             return
#         if self.tparam_.tree_method != TreeMethod.kAuto:
#             return
#
#     def configure_updaters(self):
#         if self.specified_updater_:
#             return
#         if self.tparam_.tree_method == TreeMethod.kAuto:
#             return
#         elif self.tparam_.tree_method == TreeMethod.kApprox:
#             self.tparam_.updater_seq = "grow_histmaker,prune"
#         elif self.tparam_.tree_method == TreeMethod.kExact:
#             self.tparam_.updater_seq = "grow_colmaker,prune"
#         elif self.tparam_.tree_method == TreeMethod.kHist:
#             self.tparam_.updater_seq = "grow_quantile_histmaker"
#         elif self.tparam_.tree_method == TreeMethod.kGPUHist:
#             self.tparam_.updater_seq = "grow_gpu_hist"
#         else:
#             raise Exception("Unknown Tree Method")
#
#     def configure_with_known_data(self, cfg, fmat):
#         assert self.configured_
#         updater_seq = self.tparam_.updater_seq
#         assert self.tparam_.get_initialised()
#         for k, v in cfg:
#             setattr(self.tparam_, k, v)
#         self.perform_tree_method_heuristic(fmat)
#         self.configure_updaters()
#         if updater_seq != self.tparam_.updater_seq:
#             print('Using updaters:', self.tparam_.updater_seq)
#             self.updaters_ = []
#             self.init_updater(cfg)
#
#     def do_boost(self, p_fmat, in_gpair, predt):
#         new_trees = []
#         ngroup = self.model_.learner_model_param.num_output_group
#         self.configure_with_known_data(self.cfg_, p_fmat)
#         assert ngroup != 0
#         if ngroup == 1:
#             ret = self.boost_new_trees(in_gpair, p_fmat, 0)
#             num_new_trees = len(ret)
#             new_trees += ret
#         #            if len(self.updaters_) > 0 and num_new_trees == 1 and self.
#
#         #####
#         if self.mparam.num_output_group == 1:
#             self.boost_new_trees(gpair, p_fmat, info, 0)
#         else:
#             # ngroup = number of classes in label
#             ngroup = self.mparam.num_output_group
#             nsize = len(gpair) / ngroup
#             tmp = []
#             resize(tmp, nsize, bst_gpair())
#             for gid in range(ngroup):
#                 for i in range(nsize):
#                     tmp[i] = gpair[i * ngroup + gid]
#                 self.boost_new_trees(tmp, p_fmat, info, gid)
#
#     ###3
#
#     def use_gpu(self):
#         return self.tparam_.predictor
#
#     def set_param(self, name, val):
#         if name[:4] == 'bst:':
#             self.cfg.append((name[4:], val))
#             for i in range(len(self.updaters_)):
#                 self.updaters_[i].set_param(name[4:], val)
#         if name == 'silent':
#             self.set_param('bst:silent', val)
#         self.tparam_.set_param(name, val)
#         if len(self.trees) == 0:
#             print(name)
#             self.mparam.set_param(name, val)
#
#     def init_model(self):
#         self.pred_buffer = []
#         self.pred_counter = []
#         resize(self.pred_buffer, self.mparam.pred_buffer_size())
#         resize(self.pred_counter, self.mparam.pred_buffer_size())
#         assert self.mparam.num_trees == 0, "GBTree: model already initialized"
#         assert len(self.trees) == 0, "GBTree: model already initialized"
#
#     def do_boost(self, p_fmat, info, gpair):
#
#         if self.mparam.num_output_group == 1:
#             self.boost_new_trees(gpair, p_fmat, info, 0)
#         else:
#             # ngroup = number of classes in label
#             ngroup = self.mparam.num_output_group
#             nsize = len(gpair) / ngroup
#             tmp = []
#             resize(tmp, nsize, bst_gpair())
#             for gid in range(ngroup):
#                 for i in range(nsize):
#                     tmp[i] = gpair[i * ngroup + gid]
#                 self.boost_new_trees(tmp, p_fmat, info, gid)
#
#     def predict(self, p_fmat, buffer_offset, info, out_pred, ntree_limit=0):
#         """ TODO """
#         nthread = 1
#         # info = BoosterInfo()
#         # p_fmat = FMatrixS()
#
#         resize(self.thread_temp, nthread, RegTree.FVec())
#         for i in range(nthread):
#             self.thread_temp[i].init(self.mparam.num_feature)
#
#         num_class = self.mparam.num_output_group
#         stride = info.num_row * num_class
#         out_pred = []
#         resize(out_pred, stride * (self.mparam.size_leaf_vector + 1))
#         iter_i = p_fmat.row_iterator()
#         iter_i.before_first()
#         while iter_i.next():
#             batch = iter_i.value()
#             nsize = batch.size
#             for i in range(nsize):
#                 # tid is from omp_get_thread_num
#                 tid = 0
#                 # feats is a reference to thread_temp[tid]
#                 feats = self.thread_temp[tid]
#                 ridx = batch.base_rowid + i
#                 # assert ridx < info.num_row, "data_ row index exceed bound"
#                 for gid in range(self.mparam.num_output_group):
#                     buff = -1 if buffer_offset < 0 else buffer_offset + ridx
#                     root_idx = info.get_root(ridx)
#                     new_idx = ridx * num_class + gid
#                     out_pred[new_idx] = self.pred(batch[i], buff, gid, root_idx,
#                                                   feats, out_pred[new_idx],
#                                                   stride, ntree_limit)
#         return out_pred
#
#     # inst, buffer_index, bst_group, root_index, p_feats,
#     # stride, ntree_limit
#
#     def clear(self):
#         self.trees.clear()
#         self.pred_buffer.clear()
#         self.pred_counter.clear()
#
#     def init_updater(self, cfg):
#         tval = self.tparam_.updater_seq
#         ups = tval.split(',')
#         if len(self.updaters_) != 0:
#             assert len(ups) == len(self.updaters_)
#             for up in self.updaters_:
#                 contains = up in ups
#                 if not contains:
#                     print(ups)
#                     print(self.updaters_)
#                     raise Exception('Internal Error: mismatched '
#                                     'updater sequence')
#             return
#
#         pstr = tval
#         for pstr_i in pstr:
#             if pstr_i == 'prune':
#                 self.updaters_.append(TreePruner())
#             elif pstr_i == 'refresh':
#                 self.updaters_.append(TreeRefresher())
#             elif pstr_i == 'grow_colmaker':
#                 self.updaters_.append(ColMaker())
#             else:
#                 raise ValueError('updater should be ', 'prune', 'refresh',
#                                  'grow_colmaker')
#             for name, val in self.cfg:
#                 self.updaters_[-1].set_param(name, val)
#         self.tparam_.updater_initialized = 1
#
#     def boost_new_trees(self, gpair, p_fmat, bst_group):
#         new_trees = []
#         ret = []
#         for i in range(self.tparam_.num_parallel_tree):
#             if self.tparam_.process_type == TreeProcessType.kDefault:
#                 assert self.updaters_[0].can_modify_tree()
#                 ptr = RegTree()
#                 for k, v in self.cfg_.items():
#                     setattr(ptr.param, k, v)
#                 new_trees.append(ptr)
#                 ret.append(ptr)
#             elif self.tparam_.process_type == TreeProcessType.kUpdate:
#                 assert len(self.model_.trees) < len(self.model_.trees_to_update)
#                 t = self.model_.trees_to_update[len(self.model_.trees) +
#                                                 bst_group * self.tparam_.num_parallel_tree]
#                 new_trees.append(t)
#                 ret.append(t)
#         assert len(gpair) == p_fmat.info().num_row_
#         for up in self.updaters_:
#             up.update(gpair, p_fmat, new_trees)
#         return ret
#
#     def pred(self, inst, buffer_index, bst_group, root_index, p_feats, preds,
#              stride, ntree_limit):
#         """ make a prediction for a single instance """
#         itop = 0
#         psum = 0
#         #  p_feats = RegTree.FVec()
#         vec_psum = [0] * self.mparam.size_leaf_vector
#         # (buffer_offset+ridx) + num_pbuffer * id_group * (size_leaf_vector +1)
#         bid = self.mparam.buffer_offset(buffer_index, bst_group)
#         if ntree_limit == 0:
#             treeleft = np.iinfo(np.uint32).max
#         else:
#             treeleft = ntree_limit
#         # load buffered results if any
#         if bid >= 0 and ntree_limit == 0:
#             itop = self.pred_counter[bid]
#             psum = self.pred_buffer[bid]
#             for i in range(self.mparam.size_leaf_vector):
#                 vec_psum[i] = self.pred_buffer[bid + i + 1]
#         if itop != len(self.trees):
#             p_feats.fill(inst)
#             for i in range(itop, len(self.trees)):
#                 if self.tree_info[i] == bst_group:
#                     tid = self.trees[i].get_leaf_index(p_feats, root_index)
#                     psum += self.trees[i][tid].leaf_value()
#                     for j in range(self.mparam.size_leaf_vector):
#                         vec_psum[j] += self.trees[i].leafvec(tid)[j]
#                     treeleft -= 1
#                     if treeleft == 0:
#                         break
#             p_feats.drop(inst)
#         # updated the buffered results
#         if bid >= 0 and ntree_limit == 0:
#             self.pred_counter[bid] = len(self.trees)
#             self.pred_buffer[bid] = psum
#             for i in range(self.mparam.size_leaf_vector):
#                 self.pred_buffer[bid + i + 1] = vec_psum[i]
#
#         return psum
#         # preds[0] = psum
#         # for i in range(self.mparam.size_leaf_vector):
#         #    preds[stride * (i+1)] = vec_psum[i]
#         # return preds
#
#         # out_pred[0] = psum
#         # for i in range(self.mparam.size_leaf_vector):
#
