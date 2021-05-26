from param.model_param import LearnerModelParamLegacy, LearnerModelParam
from param.model_param import LearnerTrainParam
from utils.util import resize, GenericParameter
from gbm.gbms import GradientBooster
from gbm.gbtree import GBTree
from objective.loss_function import LinearSquareLoss, ObjFunction
import numpy as np

class DataSplitMode:
  kAuto = 0
  kCol = 1
  kRow = 2


class Learner:
    def __init__(self):
        self.obj_ = None
        self.gbm_ = None
        self.metrics_ = None
        self.generic_parameters_ = GenericParameter()

    def configure(self):
        pass

    def allow_lazy_check_point(self):
        self.gbm_.allow_lazy_check_point()

    def create(self, cache):
        return LearnerImpl(cache)


class LearnerConfiguration(Learner):
    kEvalMetric = ''

    def __init__(self, cache, obj=None):
        super().__init__()
        self.need_configuration_ = True
        self.data = cache

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
        mparam_backup = self.mparam_
        for k, v in self.cfg_.items():
            if hasattr(self.tparam_, k):
                setattr(self.tparam_, k, v)
            if hasattr(self.mparam_, k):
                setattr(self.mparam_, k, v)
        self.configure_num_feature()
        self.configure_objective(old_tparam, self.cfg_)
        if self.learner_model_param_.initialized() or \
                self.mparam_.base_score != mparam_backup.base_score:
            base_score = self.obj_.prob_to_margin(self.mparam_.base_score)
            self.learner_model_param_ = LearnerModelParam(self.mparam_,
                                                          base_score)
        self.congifure_gbm(old_tparam, self.cfg_)
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
            self.obj_ = ObjFunction().create(self.tparam_.objective)
            self.obj_.configure(args)

    def create_obj(self, objective):
        obj_dict = {'multi'}

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
            self.gbm_ = GBTree(self.learner_model_param_)
        self.gbm_.configure(args)


class LearnerIO(LearnerConfiguration):
    def __init__(self, cache):
        super().__init__(cache)


class LearnerImpl(LearnerIO):
    def __init__(self, cache, labels=None, weights=None):
        super(LearnerImpl, self).__init__(cache)
        self.labels_ = labels
        # self.gbm_.model_.learner_model_param.num_output_group = 1
        self.weights_ = weights

    def update_one_iter(self, i_iter, train):
        self.configure()
        predt = self.predict_raw(train, False, 0, 0)
        n_group = self.mparam_.num_class
        if n_group == 1:
            gpair_ = self.obj_.get_gradient(predt, self.labels_, self.weights_,
                                            i_iter)
        else:
            gpair_ = np.zeros((predt.shape[0], 2, n_group))

            for i in range(n_group):
                label = self.labels_ == i
                gpair_[:, :, i] = self.obj_.get_gradient(predt, label,
                                                         self.weights_,
                                                         i_iter)
        self.gbm_.do_boost(train, gpair_)

    def boost_one_iter(self, i_iter, train, in_gpair):
        self.configure()
        # predt = self.predict_raw(train, False, 0, 0)
        self.gbm_.do_boost(train, in_gpair)

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
