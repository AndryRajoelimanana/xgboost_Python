from param.model_param import LearnerModelParamLegacy, LearnerModelParam
from param.model_param import LearnerTrainParam
from utils.util import resize, GenericParameter
from gbm.gbm import GradientBooster
from gbm.gbtree import GBTree
from objective.loss_function import LinearSquareLoss


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

        self.cfg_ = None
        self.attributes_ = None
        self.feature_names_ = None
        self.feature_types_ = None

        self.mparam_ = LearnerModelParamLegacy()
        self.learner_model_param_ = LearnerModelParam(self.mparam_, 0.5)
        self.tparam_ = LearnerTrainParam()
        self.metric_names_ = []

        self.gbm_ = GBTree(self.learner_model_param_)
        self.obj_ = obj if obj is not None else LinearSquareLoss()
        self.p_metric = None

    def configure(self):
        if not self.need_configuration_:
            return
        for k, v in self.cfg_.items():
            setattr(self.tparam_, k, v)

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
        self.weights_ = weights

    def update_one_iter(self, i_iter, train):
        predt = self.predict_raw(train, False, 0, 0)
        gpair_ = self.obj_.get_gradient(predt, self.labels_, i_iter)
        self.gbm_.do_boost(train, gpair_, predt)




    def predict_raw(self, data, training, layer_begin, layer_end):
        predt = self.gbm_.predict_batch(data, training, layer_begin, layer_end)
        return predt


