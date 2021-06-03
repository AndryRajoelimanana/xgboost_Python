from params import thresholdL1
import numpy as np


class Evaluator:

    @staticmethod
    def get_weight(p, sum_grad, sum_hess):
        if sum_hess < p.min_child_weight or sum_hess <= 0:
            return 0
        dw = - thresholdL1(sum_grad, p.reg_alpha) / (sum_hess + p.reg_lambda)
        if p.max_delta_step != 0 and np.abs(dw) > p.max_delta_step:
            dw = np.copysign(p.max_delta_step, dw)
        return dw

    @staticmethod
    def get_gain_given_weight(p, sum_grad, sum_hess, w):
        if sum_hess <= 0:
            return 0
        if p.max_delta_step == 0:
            return (thresholdL1(sum_grad, p.reg_alpha))**2 / (sum_hess +
                                                              p.reg_lambda)
        return -(2.0 * sum_grad * w + (sum_hess + p.reg_lambda) * (w * w))

    @staticmethod
    def calc_split_gain(param, sleft, sright):
        constraint = 0
        negative_infinity = np.iinfo(int).min
        g_l, h_l = sleft
        g_r, h_r = sright
        wleft = Evaluator.get_weight(param, g_l, h_l)
        wright = Evaluator.get_weight(param, g_r, h_r)

        gain_l = Evaluator.get_gain_given_weight(param, g_l, h_l, wleft)
        gain_r = Evaluator.get_gain_given_weight(param, g_r, h_r, wright)
        gain = gain_l + gain_r

        if constraint == 0:
            return gain
        elif constraint > 0:
            return gain if wleft <= wright else negative_infinity
        else:
            return gain if wleft >= wright else negative_infinity

    @staticmethod
    def get_gain(p, sum_grad, sum_hess):
        if sum_hess < p.min_child_weight:
            return 0
        if p.max_delta_step == 0.0:
            if p.reg_alpha == 0.0:
                return (sum_grad * sum_grad) / (sum_hess + p.reg_lambda)
            else:
                dw = thresholdL1(sum_grad, p.reg_alpha)
                return (dw * dw) / (sum_hess + p.reg_lambda)
        else:
            w = Evaluator.get_weight(p, sum_grad, sum_hess)
            ret = Evaluator.get_gain_given_weight(p, sum_grad, sum_hess, w)
            if p.reg_alpha == 0:
                return ret
            else:
                return ret + p.reg_alpha * np.abs(w)

    @staticmethod
    def get_loss(dat, p, grad, hess):
        loss = np.finfo(np.float).min
        best_ind = best_val = None
        for icol in range(dat.shape[1]):
            col = dat[:, icol].copy()
            ind_sort = np.argsort(col)
            col = col[ind_sort]
            gain = Evaluator.get_gain(p, grad, hess)
            for i, d in enumerate(col):
                if i == 0:
                    continue
                if d == col[i-1]:
                    continue
                left = col < d
                new_loss = Evaluator.calc_split_gain(p, grad[ind_sort],
                                                     hess[ind_sort],left) - gain
                if new_loss > loss:
                    loss = new_loss
                    best_ind = icol
                    best_val = (d + col[i-1])/2
        return loss, best_ind, best_val
