from param.parameters import XGBoostParameter
from objective.loss_function import ObjFunction
import numpy as np


class RegLossParam(XGBoostParameter):
    def __init__(self):
        super().__init__()
        self.scale_pos_weight = 1.0


class RegLossObj(ObjFunction):
    def __init__(self):
        super(RegLossObj, self).__init__()
        self.param_ = RegLossParam()
        self.additional_input_ = np.zeros(3)

    def configure(self, args):
        self.param_.update_allow_unknown(args)

    def set_param(self, name, value):
        if name == "scale_pos_weight":
            self.param_.scale_pos_weight = value

    def check_label(self, x):
        pass

    def get_gradient(self, preds, labels, weights, it):
        n = preds.shape[0]
        nstep = len(labels)
        assert n % nstep == 0, 'labels_ are not correctly provided'
        gpair = np.zeros((n, 2))
        ndata = len(preds)

        self.additional_input_[0] = 1
        is_null_weight = len(weights) == 0
        if not is_null_weight:
            assert len(weights) == ndata, "Number of weights should be equal " \
                                          "to number of data points."

        self.additional_input_[1] = self.param_.scale_pos_weight
        self.additional_input_[2] = is_null_weight
        p = self.pred_transform(preds)
        weights[labels == 1] *= self.param_.scale_pos_weight
        gpair[:, 0] = self.gradient(p, labels) * weights
        gpair[:, 1] = self.hessian(p, labels) * weights
        return gpair

    def gradient(self, ypred, y):
        return 0

    def hessian(self, ypred, y):
        return 0

    def pred_transform(self, x):
        return x


class LinearSquareLoss(RegLossObj):

    def check_label(self, x):
        return np.full(x.shape, True, dtype=bool)

    def pred_transform(self, x):
        return x

    def gradient(self, predt, label):
        return predt - label

    def hessian(self, predt, label):
        return np.ones_like(predt)

    def prob_to_margin(self, base_score):
        return base_score

    def default_eval_metric(self):
        return "rmsle"

    def name(self):
        return "reg:squaredlerror"