from gbm.gbtree_model import GBTreeModel, GBTreeModelParam
from utils.util import resize
from tree.tree_model import RegTree
from predictor.predictors import create_predictor
from param.gbtree_params import GBTreeTrainParam, DartTrainParam
from updaters.tree_updater import create_treeupdater


class GradientBooster:
    def __init__(self):
        self.generic_param_ = None

    def configure(self, cfg):
        pass

    def slice(self, layer_begin, layer_end, step, out, out_of_bound):
        raise Exception("Slice is not supported by current booster.")

    def allow_lazy_check_point(self):
        return False

    def boosted_rounds(self):
        pass

    def do_boost(self, p_fmat, in_gpair):
        pass

    def predict_batch(self, dmat, training, layer_begin, layer_end):
        pass

    def predict_leaf(self, dmat, layer_begin, layer_end):
        pass


def create_gbm(name, generic_param, learner_model_param):
    if name == 'gbtree':
        bst = GBTree(learner_model_param)
    elif name == 'gblinear':
        bst = GBLinear(learner_model_param)
    else:
        raise ValueError(f"Unknown GradientBooster: {name}")
    bst.generic_param_ = generic_param
    return bst


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


def layer_to_tree(model, tparam, layer_begin, layer_end):
    groups = model.learner_model_param.num_output_group
    tree_begin = layer_begin * groups * tparam.num_parallel_tree
    tree_end = layer_end * groups * tparam.num_parallel_tree
    if tree_end == 0:
        tree_end = len(model.trees)
    if len(model.trees) != 0:
        assert tree_begin <= tree_end
    return int(tree_begin), int(tree_end)


def slice_trees(layer_begin, layer_end, step, model, tparam, layer_trees, fn):
    tree_begin, tree_end = layer_to_tree(model, tparam, layer_begin, layer_end)
    if tree_end > len(model.trees):
        return True
    layer_end = len(model.trees) / layer_trees if layer_end == 0 else layer_end
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
        self.updaters_ = []
        self.cpu_predictor_ = None
        self.specified_updater_ = False
        self.configured_ = False

    def configure(self, cfg):
        self.init_updater(cfg)
        self.cfg_ = cfg
        updater_seq = self.tparam_.updater_seq
        self.tparam_.update_allow_unknown(cfg)

        self.model_.configure(cfg)
        if self.tparam_.process_type == TreeProcessType.kUpdate:
            self.model_.init_trees_to_update()
        if self.cpu_predictor_ is None:
            self.cpu_predictor_ = create_predictor('cpu_predictor',
                                                   self.generic_param_)
        self.cpu_predictor_.configure(cfg)

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

    def allow_lazy_check_point(self):
        return self.model_.learner_model_param.num_output_group == 1

    def layer_trees(self):
        """ Numbser trees per layer"""
        n_group = self.model_.learner_model_param.num_output_group
        n_parallel_tree = self.tparam_.num_parallel_tree
        n_trees = n_group * n_parallel_tree
        return n_trees

    def boosted_rounds(self):
        assert self.tparam_.num_parallel_tree != 0
        assert self.model_.learner_model_param.num_output_group != 0
        return len(self.model_.trees) / self.layer_trees()

    def slice(self, layer_begin, layer_end, step, out, out_of_bound):
        layer_trees = self.layer_trees()
        if layer_end == 0:
            layer_end = self.boosted_rounds()
        n_layers = (layer_end - layer_begin) / step
        out_trees = out.model_.trees
        resize(out_trees, layer_trees * n_layers, GBTreeModel())
        out_trees_info = out.model_.tree_info
        resize(out_trees_info, layer_trees * n_layers, 0)
        out.model_.param_.num_trees = len(out.model_.trees)

    def get_predictor(self):
        assert self.configured_
        return self.cpu_predictor_

    def do_boost(self, p_fmat, gpair):
        """ Boost one iteration gpair of size [n_sample, 2, n_group]"""

        grad = gpair[:, 0, :]
        hess = gpair[:, 1, :]

        new_trees = []
        ngroup = self.model_.learner_model_param.num_output_group
        assert ngroup != 0
        assert ngroup == grad.shape[1]
        self.configure_with_known_data(self.cfg_, p_fmat)

        if ngroup == 1:
            new_tree = self.boost_new_trees(grad[:, 0], hess[:, 0],
                                            p_fmat)
            new_trees.append(new_tree)
        else:
            # assert gpair.shape[0] % ngroup == 0
            for gid in range(ngroup):
                new_tree = self.boost_new_trees(grad[:, gid],
                                                hess[:, gid],
                                                p_fmat)
                new_trees.append(new_tree)
        self.commit_model(new_trees)
        return new_trees

    def boost_new_trees(self, grad, hess, p_fmat):
        new_trees = []
        for i in range(self.tparam_.num_parallel_tree):
            if self.tparam_.process_type == TreeProcessType.kDefault:
                first_updater = self.updaters_[0]
                msg_error = f'Updater: {first_updater} cannot used to create ' \
                            f'new tree'
                assert not first_updater.can_modify_tree(), msg_error
                tree = RegTree()
                tree.param.update_allow_unknown(self.cfg_)
                new_trees.append(tree)
            elif self.tparam_.process_type == TreeProcessType.kUpdate:
                for up in self.updaters_:
                    assert up.can_modify_tree(), f'Updater: {up.name} cannot ' \
                                                 f'be modify'
                    assert len(self.model_.trees) < \
                           len(self.model_.trees_to_update), f'No more tree ' \
                                                             f'left for ' \
                                                             f'updating'
                    t = self.model_.trees_to_update[len(self.model_.trees) +
                                                    self.tparam_.num_parallel_tree]
                    new_trees.append(t)

        # update the trees
        assert grad.shape[0] == p_fmat.shape[0], 'Mismatch row size'
        for up in self.updaters_:
            up.update(grad, hess, p_fmat, new_trees)
        return new_trees

    def inplace_predict(self):
        pass

    def commit_model(self, new_trees):
        ngroup = self.model_.learner_model_param.num_output_group
        for gid in range(ngroup):
            self.model_.commit_model(new_trees[gid], gid)

    def predict_batch(self, dmat, training, layer_begin, layer_end):
        assert self.configured_
        if layer_end == 0:
            layer_end = self.boosted_rounds()
        predictor = self.get_predictor()
        tree_begin, tree_end = self.layer_to_tree(layer_begin, layer_end)
        init_pred = predictor.init_prediction(self.model_, dmat.shape[0])
        preds = predictor.predict_batch(dmat, self.model_,
                                        tree_begin,
                                        tree_end)
        return init_pred + preds

    def predict_instance(self, inst, layer_begin, layer_end):
        assert self.configured_, 'GBTree : not configured'
        tree_begin, tree_end = self.layer_to_tree(layer_begin, layer_end)
        return self.cpu_predictor_.predict_instance(inst, self.model_,
                                                    tree_end)

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
            raise Exception(f'Unknown Tree Method {self.tparam_.tree_method} '
                            f'detected')

    def configure_with_known_data(self, cfg, fmat):
        assert self.configured_
        updater_seq = self.tparam_.updater_seq
        assert self.tparam_.get_initialised()
        self.tparam_.update_allow_unknown(cfg)
        self.configure_updaters()
        if updater_seq != self.tparam_.updater_seq:
            self.updaters_ = []
            self.init_updater(cfg)

    def init_updater(self, cfg):
        tval = self.tparam_.updater_seq
        ups = tval.split(',')
        if len(self.updaters_) != 0:
            assert len(ups) == len(self.updaters_)
            for up in self.updaters_:
                if up.name() not in ups:
                    raise Exception('Internal Error: mismatched '
                                    'updater sequence')
            return
        for pstr in ups:
            up = create_treeupdater(pstr, self.generic_param_)
            up.configure(cfg)
            self.updaters_.append(up)

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

    def commit_model(self, new_trees):
        ngroup = self.model_.learner_model_param.num_output_group
        num_new_trees = 0
        for gid in range(ngroup):
            num_new_trees += len(new_trees[gid])
            self.model_.commit_model(new_trees[gid], gid)

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


class GBLinear(GradientBooster):
    def __init__(self, booster_config):
        super(GBLinear, self).__init__()