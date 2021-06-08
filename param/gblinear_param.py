from param.parameters import XGBoostParameter, dmlcParameter
import numpy as np


class FeatureSelectorEnum:
    kCyclic = 0
    kShuffle = 1
    kThrifty = 2
    kGreedy = 3
    kRandom = 4


class GBLinearTrainParam(XGBoostParameter):
    def __init__(self):
        super(GBLinearTrainParam, self).__init__()
        self.updater = 'shotgun'
        self.tolerance = 0.0
        self.max_row_prebatch = np.iinfo(np.uint32).max


class GBLinearModelParam(dmlcParameter):
    def __init__(self):
        # This is deprecated
        super(GBLinearModelParam, self).__init__()
        pass


class LinearTrainParam(XGBoostParameter):
    def __init__(self):
        super(LinearTrainParam, self).__init__()
        self.learning_rate = 0.5
        self.reg_lambda = 0.0
        self.reg_alpha = 0.0
        self.feature_selector = 'cyclic'
        self.reg_lambda_denorm = self.reg_alpha_denorm = None

    def denormalize_penalties(self, sum_instance_weight):
        self.reg_lambda_denorm = self.reg_lambda * sum_instance_weight
        self.reg_alpha_denorm = self.reg_alpha * sum_instance_weight


class CoordinateParam(XGBoostParameter):
    def __init__(self):
        super(CoordinateParam, self).__init__()
        self.top_k = 0
