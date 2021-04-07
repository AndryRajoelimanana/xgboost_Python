import numpy as np


class RegLossObj:
    def __init__(self, loss_type):
        self.loss_type = loss_type
        self.scale_pos_weight = 1.0

    def get_gradient(self, labels, preds, w):
        nstep = labels.shape[0]
        ndata = preds.shape[0]
        assert nstep == ndata, 'labels are not correctly provided'
        return self.gradient(labels, preds)*w, self.hessian(labels, preds)*w

    def gradient(self, y, ypred):
        pass

    def hessian(self, y, ypred):
        pass

    def pred_transform(self, x):
        pass

    @staticmethod
    def sigmoid(x):
        return 1.0/(1.0 + np.exp(-x))

    def softmax(self, score):
        score = np.asarray(score, dtype=float)
        score_max = np.max(score)
        score = np.exp(score - score_max)
        score /= np.sum(score, axis=1)[:, np.newaxis]
        return score


class SquareErrorLoss(RegLossObj):
    def __init__(self):
        super().__init__('square_error')

    def gradient(self, y, ypred):
        return y - ypred

    def hessian(self, y, ypred):
        return np.ones_like(y)

    def pred_transform(self, x):
        return x


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


class SoftmaxMultiClassObj:
    def __init__(self):
        self.n_class = 0

    def get_gradient(self, labels, preds):
        n_class = preds.shape[1]
        nstep = labels.shape[0]
        ndata = preds.shape[0]
        assert nstep == ndata, 'labels are not correctly provided'
        for k in range(n_class):
            p = preds[:, k]

