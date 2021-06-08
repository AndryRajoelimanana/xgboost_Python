from gbm.gb_interface import GradientBooster
from gbm.gblinear_model import GBLinearModel
from param.gblinear_param import GBLinearTrainParam
from updaters.linear_updater import LinearUpdater
import numpy as np


class GBLinear(GradientBooster):
    def __init__(self, booster_config):
        super(GBLinear, self).__init__()
        self.num_boosted_rounds = 0
        self.learner_model_param_ = booster_config
        self.model_ = GBLinearModel(booster_config)
        self.previous_model_ = GBLinearModel(booster_config)
        self.sum_instance_weight_ = 0
        self.sum_weight_complete_ = False
        self.is_converged_ = False
        self.param_ = GBLinearTrainParam()
        self.updater_ = None

    def configure(self, cfg):
        if self.model_.weight.shape[0] == 0:
            self.model_.configure(cfg)
        self.param_.update_allow_unknown(cfg)
        self.updater_ = LinearUpdater.create(self.param_.updater,
                                             self.generic_param_)
        self.updater_.configure(cfg)

    def boosted_rounds(self):
        return self.model_.num_boosted_rounds

    def do_boost(self, p_fmat, in_gpair):
        self.model_.lazy_init_model()
        self.lazy_sum_weights(p_fmat)
        if not self.check_convergence():
            self.updater_.update(in_gpair, p_fmat, self.model_,
                                 self.sum_instance_weight_)
        self.model_.num_boosted_rounds += 1

    def lazy_sum_weights(self, weights):
        if not self.sum_weight_complete_:
            # todos we have to get weight (currently weights=1 for all
            # instance)
            self.sum_instance_weight_ = weights.shape[0]
            self.sum_weight_complete_ = True

    def check_convergence(self):
        if self.param_.tolerance == 0:
            return False
        if self.is_converged_:
            return True
        if self.previous_model_.weight.size != self.model_.weight.size:
            self.previous_model_ = self.model_
            return False
        largest_dw = np.max(np.abs(self.model_.weight -
                                   self.previous_model_.weight))
        self.previous_model_ = self.model_
        self.is_converged_ = largest_dw <= self.param_.tolerance
        return self.is_converged_

    def predict_batch(self, dmat, training, layer_begin, layer_end):
        assert layer_begin == 0 and layer_end == 0, f'Linear booster does not' \
                                                    f' support prediction ' \
                                                    f'range.'
        return self.predict_batch_internal(dmat)

    def predict_leaf(self, dmat, layer_begin, layer_end):
        return self.predict_batch(dmat, True, layer_begin, layer_end)

    def predict_instance(self, dmat):
        return self.predict_batch(dmat, True, 0, 0)

    def predict_contribution(self, dmat):
        # todos
        return self.predict_batch(dmat, True, 0, 0)

    def predict_interaction_contributions(self, dmat):
        # todos
        return self.predict_batch(dmat, True, 0, 0)

    def predict_batch_internal(self, dmat):
        self.model_.lazy_init_model()
        base_margin = self.model_.learner_model_param.base_score
        ngroup = self.model_.learner_model_param.num_output_group
        # if base_margin.shape[0] != 0:
        #    assert base_margin.shape[0] == dmat.shape[0]
        pred = np.zeros((dmat.shape[0], ngroup))
        for gid in range(ngroup):
            bias = self.model_.weight[-1:, gid]
            weights = self.model_.weight[:-1, gid]
            pred[:, gid] = bias + base_margin
            pred[:, gid] += np.sum(weights * dmat, axis=1)
        return pred
