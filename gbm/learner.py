from param.model_param import LearnerModelParamLegacy, LearnerModelParam
from param.model_param import LearnerTrainParam
from gbm.gbtree import create_gbm
from objective.loss_function import LinearSquareLoss, create_objective
import numpy as np
from param.generic_param import GenericParameter
from utils.util import check_random_state
import time

kRandSeedMagic = 127


class DataSplitMode:
  kAuto = 0
  kCol = 1
  kRow = 2


class Learner:
    def __init__(self, seed=0):
        self.obj_ = None
        self.gbm_ = None
        self.metrics_ = None
        self.generic_parameters_ = GenericParameter(seed=seed)

    def configure(self):
        pass

    def allow_lazy_check_point(self):
        self.gbm_.allow_lazy_check_point()

    def create(self, cache):
        return LearnerImpl(cache)


class LearnerConfiguration(Learner):
    kEvalMetric = ''

    def __init__(self, cache, obj=None, seed=0):
        super().__init__(seed)
        self.need_configuration_ = True
        self.data = cache
        self.rng_ = check_random_state(seed)

        self.cfg_ = {}
        self.attributes_ = {}
        self.feature_names_ = []
        self.feature_types_ = []

        self.mparam_ = LearnerModelParamLegacy()
        self.learner_model_param_ = LearnerModelParam(self.mparam_, 0.5)
        self.tparam_ = LearnerTrainParam()
        self.metric_names_ = []

        self.gbm_ = None
        self.obj_ = obj if obj is not None else LinearSquareLoss()
        self.p_metric = None

    def configure(self):
        if not self.need_configuration_:
            return
        old_tparam = self.tparam_
        args = self.cfg_
        self.tparam_.update_allow_unknown(args)
        mparam_backup = self.mparam_
        self.mparam_.update_allow_unknown(args)

        initialized = self.generic_parameters_.get_initialised()
        old_seed = self.generic_parameters_.seed

        self.generic_parameters_.update_allow_unknown(args)

        if not initialized or self.generic_parameters_.seed != old_seed:
            self.rng_.seed(self.generic_parameters_.seed)

        self.configure_num_feature()
        args = self.cfg_
        self.configure_objective(old_tparam, args)

        if not self.learner_model_param_.initialized() or \
                self.mparam_.base_score != mparam_backup.base_score:
            base_score = self.obj_.prob_to_margin(self.mparam_.base_score)
            self.learner_model_param_ = LearnerModelParam(self.mparam_,
                                                          base_score)
        self.congifure_gbm(old_tparam, args)
        self.need_configuration_ = False

    def configure_num_feature(self):
        num_feature = self.data.shape[1]
        if num_feature > self.mparam_.num_feature:
            self.mparam_.num_feature = num_feature
        assert self.mparam_.num_feature > 0
        self.cfg_['num_feautre'] = self.mparam_.num_feature
        self.cfg_['num_class'] = self.mparam_.num_class

    def configure_objective(self, old_tparam, args):
        if 'num_class' in self.cfg_:
            if self.cfg_['num_class'] != 0 and self.tparam_.objective != \
                    "multi:softprob":
                self.cfg_['num_output_group'] = self.cfg_['num_class']
                if (self.cfg_['num_class'] > 1) and ("objective" not in
                                                     self.cfg_):
                    self.tparam_.objective = "multi:softmax"

        if 'max_delta_step' not in self.cfg_ and 'objective' in self.cfg_ and\
                self.tparam_.objective == 'count:poisson':
            self.cfg_['max_delta_step'] = 0.7   # kMaxDeltaStepDefaultValue

        if self.obj_ is None or self.tparam_.objective != old_tparam.objective:
            self.obj_ = create_objective(self.tparam_.objective)
            self.obj_.configure(args)

    def set_param(self, key, value):
        self.need_configuration_ = True
        if key == "eval_metric":
            if value not in self.metric_names_:
                self.metric_names_.append(value)
        else:
            self.cfg_[key] = value

    def set_params(self, args):
        for k, v in args:
            self.set_param(k, v)

    def get_num_feature(self):
        return self.learner_model_param_.num_feature

    def configure_num_features(self):
        # num feature
        num_feature = self.data.shape[1]
        self.mparam_.num_feature = num_feature
        self.learner_model_param_.num_feature = num_feature
        self.cfg_["num_feature"] = self.mparam_.num_feature
        self.cfg_["num_class"] = self.mparam_.num_class

    def congifure_gbm(self, old, args):
        if not self.gbm_ or old.booster != self.tparam_.booster:
            self.gbm_ = create_gbm(self.tparam_.booster,
                                   self.generic_parameters_,
                                   self.learner_model_param_)
        self.gbm_.configure(args)


class LearnerIO(LearnerConfiguration):
    def __init__(self, cache, seed=0):
        super().__init__(cache, seed=seed)


class LearnerImpl(LearnerIO):
    def __init__(self, cache, labels=None, weights=None, seed=0):
        super(LearnerImpl, self).__init__(cache, seed=seed)
        self.labels_ = labels
        self.weights_ = weights

    def update_one_iter(self, i_iter, train):
        self.configure()
        if self.generic_parameters_.seed_per_iteration:
            self.rng_.seed(self.generic_parameters_.seed * kRandSeedMagic
                           + i_iter)

        predt = self.predict_raw(train, True, 0, 0)
        n_group = self.mparam_.num_class
        gpair_ = np.zeros((predt.shape[0], 2, n_group))
        for i in range(n_group):
            gpair_[:, :, i] = self.obj_.get_gradient(predt[:, i], self.labels_[
                                                               :, i],
                                                     self.weights_, i_iter)
        self.gbm_.do_boost(train, gpair_)

    def boost_one_iter(self, i_iter, train, in_gpair):
        tic = time.perf_counter()
        print(f'Iteration {i_iter}: Starting.....')
        self.configure()
        self.gbm_.do_boost(train, in_gpair)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        print(f'Iteration {i_iter}: Finished in {elapsed_time:0.4f} seconds\n')

    def predict(self, data, output_margin, layer_begin, layer_end, training,
                pred_leaf, pred_contribs, approx_contribs, pred_interactions):
        self.configure()
        if pred_leaf + pred_contribs + pred_interactions > 1:
            raise Exception("Perform one kind of prediction at a time.")
        if pred_contribs:
            out_preds = self.gbm_.predict_contribution(data, layer_begin,
                                                       layer_end,
                                                       approx_contribs)
        elif pred_interactions:
            out_preds = self.predict_interactionContributions(data, layer_begin,
                                                              layer_end,
                                                              approx_contribs)
        elif pred_leaf:
            out_preds = self.gbm_.predict_leaf(data, layer_begin, layer_end)
        else:
            out_preds = self.predict_raw(data, training, layer_begin, layer_end)
            if not output_margin:
                self.obj_.pred_transform(out_preds)
        return out_preds

    def predict_raw(self, data, training, layer_begin, layer_end):
        predt = self.gbm_.predict_batch(data, training, layer_begin, layer_end)
        return predt

    def boosted_round(self):
        if not self.gbm_:
            return 0
        return self.gbm_.boosted_rounds()

    def groups(self):
        return self.learner_model_param_.num_output_group

    def get_configuration_arguments(self):
        return self.cfg_
