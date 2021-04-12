import numpy as np
from utils.data_mat import bst_gpair
from utils.util import sigmoid, softmax, one_hot_encoding


class IObjFunction:
    """
    IObjFunction xgboost.learner : learner/objective.h
    interface
    """
    def set_param(self, name, value):
        pass

    def get_gradient(self, preds, info, iters):
        pass

    def default_eval_metric(self):
        pass

    def pred_transform(self, io_preds):
        pass

    def eval_transform(self, io_pred):
        self.pred_transform(io_pred)

    def prob_to_margin(self, base_score):
        return base_score


class RegLossObj(IObjFunction):
    def __init__(self, is_classifier):
        self.is_classifier = is_classifier
        self.scale_pos_weight = 1.0

    def set_param(self, name, value):
        if name == "scale_pos_weight":
            self.scale_pos_weight = value

    def get_gradient(self, labels, preds, w):
        nstep = labels.shape[0]
        ndata = preds.shape[0]
        assert nstep == ndata, 'labels are not correctly provided'
        return bst_gpair(self.gradient(labels, preds) * w,
                         self.hessian(labels, preds) * w)

    def gradient(self, y, ypred):
        pass

    def hessian(self, y, ypred):
        pass

    def pred_transform(self, x):
        return x


class LinearSquare(RegLossObj):
    def __init__(self, is_classifier=False):
        super().__init__(is_classifier)

    def pred_transform(self, x):
        return x

    def gradient(self, y, ypred):
        return y - ypred

    def hessian(self, y, ypred):
        return np.ones_like(y)

    def prob_to_margin(self, base_score):
        assert np.min(base_score) >= 0 and np.max(base_score) <= 1, \
            'base score should be in (0,1)'
        base_score = -np.log(1/(base_score - 1))
        return base_score


class LogisticNeglik(RegLossObj):
    def __init__(self, is_classifier=False):
        super().__init__(is_classifier)

    def pred_transform(self, x):
        return sigmoid(x)

    def gradient(self, y, ypred):
        preds = sigmoid(ypred)
        return y - preds

    def hessian(self, y, ypred):
        preds = sigmoid(ypred)
        return preds * (1 - preds)

    def prob_to_margin(self, base_score):
        assert np.min(base_score) >= 0 and np.max(base_score) <= 1, \
            'base score should be in (0,1)'
        base_score = -np.log(1/(base_score - 1))
        return base_score


class SoftmaxMultiClassObj(IObjFunction):
    def __init__(self, output_prob):
        self.output_prob = output_prob
        self.nclass = 0

    def set_param(self, name, value):
        if name == 'nclass':
            self.nclass = value

    def get_gradient(self, y, y_pred, w):
        # labels, preds, w
        assert self.nclass > 0, 'must set n_class'
        probas = softmax(y_pred)
        eps = 1e-16
        if w is None:
            sample_weight = np.ones(y.shape[0])
        hess = np.maximum((2.0 * probas * (1.0 - probas)) *
                          sample_weight[:, np.newaxis], eps)
        grad = (probas - one_hot_encoding(y)) * sample_weight[:, np.newaxis]
        return bst_gpair(grad, hess)

    def pred_transform(self, io_preds, is_eval=True):
        if is_eval:
            return SoftmaxMultiClassObj.__transform(io_preds, 0)
        else:
            return SoftmaxMultiClassObj.__transform(io_preds, self.output_prob)

    @staticmethod
    def __transform(self, io_preds, prob):
        if prob == 0:
            return np.max(io_preds, axis=1)
        else:
            return softmax(io_preds)


class LogLoss(RegLossObj):
    def __init__(self):
        super().__init__('LogLoss')

    def gradient(self, y, ypred):
        return y - ypred

    def hessian(self, y, ypred):
        return ypred * (1 - ypred)

    def pred_transform(self, x):
        return self.sigmoid(x)


class MultiLogLoss(RegLossObj):
    def __init__(self):
        super().__init__('LogLoss')

    def gradient(self, y, ypred, k=0):
        p = self.softmax(ypred)

    def hessian(self, y, ypred):
        return ypred * (1 - ypred)

    def pred_transform(self, x):
        return self.sigmoid(x)


class SquareErrorLoss(RegLossObj):
    def __init__(self, is_classifier=False):
        super().__init__(is_classifier)

    def pred_transform(self, x):
        return x

    def gradient(self, y, ypred):
        return y - ypred

    def hessian(self, y, ypred):
        return np.ones_like(y)

    def prob_to_margin(self, base_score):
        assert np.min(base_score) >= 0 and np.max(base_score) <= 1, \
            'base score should be in (0,1)'
        base_score = -np.log(1/(base_score - 1))
        return base_score