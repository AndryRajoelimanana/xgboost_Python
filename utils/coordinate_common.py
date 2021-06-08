import numpy as np
from utils.random_sampler import shuffle_std
from utils.util import check_random_state


def coordinate_delta(sum_grad, sum_hess, w, reg_alpha, reg_lambda):
    if sum_hess < 1e-5:
        return 0.0
    sum_grad_l2 = sum_grad + reg_lambda * w
    sum_hess_l2 = sum_hess + reg_lambda
    tmp = w - sum_grad_l2 / sum_hess_l2
    if tmp >= 0:
        return max(-(sum_grad_l2 + reg_alpha) / sum_hess_l2, -w)
    else:
        return min(-(sum_grad_l2 - reg_alpha) / sum_hess_l2, -w)


def coordinate_delta_bias(sum_grad, sum_hess):
    """ Calculate update to bias"""
    return -sum_grad / sum_hess


def get_gradient(fdata, grad, hess):
    """

    :param fdata: the feature matrix for specific feature
    :param grad Gradient
    :param hess Hessian
    :return: gradient and diagonal Hessian for a given feature
    """
    ndata = fdata.shape[0]
    sum_grad = (grad[hess >= 0.0] * fdata).sum()
    sum_hess = (hess[hess >= 0.0] * fdata * fdata).sum()
    gpair = np.zeros((ndata, 2))
    gpair[:, 0] = sum_grad
    gpair[:, 1] = sum_hess
    return gpair


def get_bias_gradient(gpair):
    pos_hessian = gpair[:, 1] >= 0
    sum_grad = gpair[pos_hessian, 0].sum()
    sum_hess = gpair[pos_hessian, 1].sum()
    return sum_grad, sum_hess


def update_residual(fdata, grad, hess, dw):
    if dw == 0.0:
        return
    ndata = fdata.shape[0]
    new_grad = grad + hess * fdata * dw
    gpair = np.zeros((ndata, 2))
    gpair[:, 0] = new_grad
    gpair[:, 1] = hess
    return gpair


def update_bias_residual(fdata, grad, hess, dbias):
    if dbias == 0.0:
        return
    ndata = fdata.shape[0]
    new_grad = grad + hess * dbias
    gpair = np.zeros((ndata, 2))
    gpair[:, 0] = new_grad
    gpair[:, 1] = hess
    return gpair


def create_feature_selector(choice):
    if choice == 'cyclic':
        return CyclicFeatureSelector()
    elif choice == 'shuffle':
        return ShuffleFeatureSelector()
    elif choice == 'random':
        return RandomFeatureSelector()


class FeatureSelector:

    def setup(self, model, gpair, data, reg_alpha, reg_lambda, param,
              random_state=0):
        pass

    def next_feature(self, iteration, model, group_idx, gpair, fdata,
                     reg_alpha, reg_lambda):
        pass


class CyclicFeatureSelector(FeatureSelector):
    def __init__(self):
        pass

    def next_feature(self, iteration, model, group_idx, gpair, fdata,
                     reg_alpha=0, reg_lambda=0):
        return iteration % model.learner_model_param.num_feature


class ShuffleFeatureSelector(FeatureSelector):
    def __init__(self):
        self.feat_index_ = []

    def setup(self, model, gpair, data, reg_alpha=0,
              reg_lambda=0, param=0, random_state=0):
        rng = check_random_state(random_state)
        if len(self.feat_index_):
            self.feat_index_ = list(range(
                model.learner_model_param.num_feature))
        self.feat_index_ = shuffle_std(self.feat_index_, rng)

    def next_feature(self, iteration, model, group_idx, gpair, fdata,
                     reg_alpha, reg_lambda):
        num_feature = model.learner_model_param.num_feature
        return self.feat_index_[iteration % num_feature]


class RandomFeatureSelector(FeatureSelector):
    def __init__(self):
        pass

    def next_feature(self, iteration, model, group_idx, gpair, fdata,
                     reg_alpha=0, reg_lambda=0, random_state=0):
        rng = check_random_state(random_state)
        return rng.randint(2**32) % model.learner_model_param.num_feature
