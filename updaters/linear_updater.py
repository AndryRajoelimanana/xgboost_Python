from param.generic_param import GenericParameter
from param.gblinear_param import LinearTrainParam
from utils.coordinate_common import *


class LinearUpdater:
    def __init__(self):
        self.learner_param_ = GenericParameter()

    @staticmethod
    def create(name, lparam, random_state=0):
        if name == 'shotgun':
            updater = ShotgunUpdater(random_state)
        elif name == 'coord_descent':
            updater = CoordinateUpdater(random_state)
        else:
            raise ValueError(f"Unknown GradientBooster: {name}")
        updater.learner_param_ = lparam
        return updater


class ShotgunUpdater(LinearUpdater):
    def __init__(self, random_state=0):
        super(ShotgunUpdater, self).__init__()
        self.param_ = LinearTrainParam()
        self.selector_ = None
        self.rng = check_random_state(random_state)

    def configure(self, args):
        self.param_.update_allow_unknown(args)
        feat_selector = self.param_.feature_selector
        if feat_selector != 'cyclic' and feat_selector != 'shuffle':
            raise ValueError(f"Unsupported feature selector: {feat_selector} "
                             f", for shotgun updater choose 'cyclic' or "
                             f"'shuffle'")
        self.selector_ = create_feature_selector(feat_selector)

    def update(self, gpair, fmat, model, sum_instance_weight=1):
        self.param_.denormalize_penalties(sum_instance_weight)
        ngroup = model.learner_model_param.num_output_group
        lr = self.param_.learning_rate
        for gid in range(ngroup):
            # grad wrt bias
            pos_hess = gpair[:, 1, gid] >= 0
            grad_bias_pair = get_bias_gradient(gpair[pos_hess, :, gid])
            dbias = lr * (- grad_bias_pair[0] / grad_bias_pair[1])

            # update bias
            model.weight[-1:, gid] += dbias

            # Updates the gradient vector based on a change in the bias.
            pos_hess = gpair[:, 1, gid] >= 0
            gpair[pos_hess, 0, gid] += gpair[pos_hess, 1, gid] * dbias

        self.selector_.setup(model, gpair, fmat,
                             self.param_.reg_alpha_denorm,
                             self.param_.reg_lambda_denorm, 0, self.rng)

        for feat in range(fmat.shape[1]):
            fid = self.selector_.next_feature(feat, model, 0, gpair, fmat,
                                              self.param_.reg_alpha_denorm,
                                              self.param_.reg_lambda_denorm)
            if fid < 0:
                return
            col = fmat[:, fid]
            gpair_c = gpair.copy()
            for gid in range(ngroup):
                pos_hess = gpair_c[:, 1, gid] >= 0
                sum_grad = (gpair_c[pos_hess, 0, gid] * col).sum()
                sum_hess = (gpair_c[pos_hess, 1, gid] * col * col).sum()
                w = model.weight[fid, gid]
                coord_delta = coordinate_delta(sum_grad, sum_hess, w,
                                               self.param_.reg_alpha_denorm,
                                               self.param_.reg_lambda_denorm)
                dw = lr * coord_delta
                if dw == 0:
                    continue
                model.weight[fid, gid] += dw
                pos_h = gpair_c[:, 1, gid] >= 0
                gpair_c[pos_h, 0, gid] += gpair_c[pos_h, 1, gid] * col * dw


class CoordinateUpdater(LinearUpdater):
    def __init__(self, random_state=0):
        super(CoordinateUpdater, self).__init__()
        self.random_state = check_random_state(random_state)
