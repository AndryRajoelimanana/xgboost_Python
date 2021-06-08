from utils.util import softmax, sigmoid, one_hot_encoding
import numpy as np
from objective.objective_interface import ObjFunction


class RegLossObj(ObjFunction):
    def __init__(self, is_classifier=True):
        super(RegLossObj, self).__init__()
        self.is_classifier = is_classifier
        self.scale_pos_weight = 1.0

    def set_param(self, name, value):
        if name == "scale_pos_weight":
            self.scale_pos_weight = value

    def check_label(self, x):
        pass

    def get_gradient(self, preds, labels, weights, it):
        n = preds.shape[0]
        nstep = len(labels)
        assert n % nstep == 0, 'labels_ are not correctly provided'
        gpair = np.zeros((n, 2))
        p = self.pred_transform(preds)
        weights[labels == 1] *= self.scale_pos_weight
        try:
            gpair[:, 0] = self.gradient(labels, p) * weights
            gpair[:, 1] = self.hessian(labels, p) * weights
        except:
            print(0)
        return gpair

    def gradient(self, y, ypred):
        return 0

    def hessian(self, y, ypred):
        return 0

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
        assert np.min(base_score) > 0 and np.max(base_score) < 1, \
            'base score should be in (0,1)'
        base_score = -np.log(1./base_score - 1)
        return base_score


class SoftmaxMultiClassObj(ObjFunction):
    def __init__(self, output_prob_):
        super(SoftmaxMultiClassObj, self).__init__()
        self.output_prob_ = output_prob_
        self.nclass = 0

    def set_param(self, name, value):
        if name == 'nclass':
            self.nclass = value

    def get_gradient(self, y, y_pred, w, i):
        # labels_, preds, w
        assert self.nclass > 0, 'must set n_class'
        probas = softmax(y_pred)
        eps = 1e-16
        if w is None:
            sample_weight = np.ones(y.shape[0])
        else:
            sample_weight = w
        hess = np.maximum((2.0 * probas * (1.0 - probas)) *
                          sample_weight[:, np.newaxis], eps)
        grad = (probas - one_hot_encoding(y)) * sample_weight[:, np.newaxis]
        return grad, hess

    def pred_transform(self, io_preds, is_eval=True):
        if is_eval:
            return SoftmaxMultiClassObj.__transform(io_preds, 0)
        else:
            return SoftmaxMultiClassObj.__transform(io_preds,
            self.output_prob_)

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


class LinearSquareLoss(RegLossObj):

    def check_label(self, x):
        return np.full(x.shape, True, dtype=bool)

    def pred_transform(self, x):
        return x

    def gradient(self, predt, label):
        print('tato')
        return predt - label

    def hessian(self, predt, label):
        return np.ones_like(predt)

    def prob_to_margin(self, base_score):
        return base_score

    def default_eval_metric(self):
        return "rmsle"

    def name(self):
        return "reg:squaredlerror"


class SquareLogError(RegLossObj):

    def check_label(self, x):
        return x > -1

    def pred_transform(self, x):
        return x

    def gradient(self, predt, label):
        ypredi = predt.clip(min=-1 + 1e-6)
        res = (np.log1p(ypredi) - np.log1p(label))/(ypredi + 1)
        return res

    def hessian(self, predt, label):
        ypredi = predt.clip(min=-1 + 1e-6)
        res = (-np.log1p(ypredi) + np.log1p(label) + 1
        ) / np.power(ypredi+1, 2)
        res = res.clip(min=1e-6)
        return res

    def prob_to_margin(self, base_score):
        return base_score

    def default_eval_metric(self):
        return "rmsle"

    def name(self):
        return "reg:squaredlogerror"


class LogisticRegression(RegLossObj):
    def pred_transform(self, x):
        return sigmoid(x)

    def check_label(self, x):
        return (x >= 0) & (x <= 1)

    def gradient(self, predt, label):
        return predt - label

    def hessian(self, predt, label):
        predt_i = predt * (1.0 - predt)
        return predt_i.clip(min=1e-16)

    def prob_to_margin(self, base_score):
        msg = "base_score must be in (0,1) for logistic loss, got: "
        assert np.all((base_score > 0.0) & (base_score < 1.0)), msg
        return -np.log(1.0/base_score - 1.0)

    def default_eval_metric(self):
        return "rmse"

    def name(self):
        return "reg:logistic"


class LogisticClassification(LogisticRegression):
    def default_eval_metric(self):
        return "logloss"

    def name(self):
        return "binary:logistic"


class LogisticRaw(LogisticRegression):
    def pred_transform(self, x):
        return x

    def gradient(self, predt, label):
        return sigmoid(predt) - label

    def hessian(self, predt, label):
        predt = sigmoid(predt)
        predt_i = predt * (1.0 - predt)
        return predt_i.clip(min=1e-16)

    def prob_to_margin(self, base_score):
        return base_score

    def default_eval_metric(self):
        return "logloss"

    def name(self):
        return "binary:logitraw"


if __name__ == "__main__":
    pass