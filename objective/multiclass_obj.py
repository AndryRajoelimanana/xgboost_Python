from objective.objective_interface import ObjFunction
from utils.util import softmax
import numpy as np
from param.parameters import XGBoostParameter


class SoftMultiClassParam(XGBoostParameter):
    def __init__(self, num_class=1):
        super(SoftMultiClassParam, self).__init__()
        self.num_class = num_class


class SoftmaxMultiClassObj(ObjFunction):
    def __init__(self, output_prob):
        super(SoftmaxMultiClassObj, self).__init__()
        self.output_prob_ = output_prob
        self.param_ = SoftMultiClassParam()

    def configure(self, args):
        self.param_.update_allow_unknown(args)

    def set_param(self, name, value):
        if name == 'nclass':
            self.param_.num_class = value

    def get_gradient(self, preds, labels, weights, it):
        # labels_, preds, w
        if len(labels) == 0:
            return 0

        ndata = preds.shape[0]
        gpair = np.zeros((ndata, 2))
        probas = softmax(preds)
        eps = 1e-16
        if weights is None:
            sample_weight = np.ones(labels.shape[0])
        else:
            sample_weight = weights
        hess = np.maximum((2.0 * probas * (1.0 - probas)) *
                          sample_weight, eps)
        grad = (probas - labels) * sample_weight
        gpair[:, 0] = grad
        gpair[:, 1] = hess
        return gpair

    def pred_transform(self, io_preds, is_eval=True):
        if is_eval:
            return SoftmaxMultiClassObj.__transform(io_preds, 0)
        else:
            return SoftmaxMultiClassObj.__transform(io_preds, self.output_prob_)

    @staticmethod
    def __transform(io_preds, prob):
        if prob == 0:
            return np.max(io_preds, axis=1)
        else:
            return softmax(io_preds)
