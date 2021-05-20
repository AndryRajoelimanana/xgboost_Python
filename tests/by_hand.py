from sklearn import datasets
from params import *
from objective.loss_function import LinearSquareLoss
from scipy.sparse import csc_matrix
from tree.split_evaluator import TreeEvaluator
from updaters.update_colmaker import ThreadEntry
from tree.tree_model import RegTree
from param.gbt_tparam import GBTreeTrainParam



class Entry:
    def __init__(self, index, data):
        self.data = data
        self.index = index


class DMat:
    def __init__(self, data):
        self.data = data

    def get_col(self, i):
        col = self.data.getcol(i)
        entries = []
        for i in range(col.size):
            entries.append(Entry(col.indices[i], col.data[i]))
        return sorted(entries, key=lambda x: x.data)


def get_snode(positions, gpair):
    snode = GradStats()
    for i, pair in enumerate(gpair):
        if positions[i] < 0:
            continue
        snode.add(pair)
    return snode


def positions(gpair):
    pos = np.zeros(len(gpair), dtype=int)
    for i, pair in enumerate(gpair):
        if pair.get_hess() < 0:
            pos[i] = -1
    return list(pos)


def enumerate_split(p, pos, data, gpair, snode_stats, snode_root_gain, fid):
    e = GradStats()
    e.add(gpair[data[0].index])
    c = GradStats()
    last_fvalue = data[0].fvalue

    for i, dat in enumerate(data):
        idx = dat.index
        nid = pos[idx]
        if nid < 0:
            continue
        if (dat.fvalue == last_fvalue) or e.sum_hess < p.min_child_weight:
            continue
        c.set_substract(snode_stats, e)
        if c.sum_hess >= p.min_child_weight:
            loss_chg = ev.calc_split_gain(p, 0, fid, e, c) - snode_root_gain
        proposed_split = (dat.fvalue + last_fvalue) * 0.5
        return loss_chg, proposed_split, c, e.stats


def get_weight(p, grad, hess):
    sum_grad = grad.sum()
    sum_hess = hess.sum()
    if sum_hess < p.min_child_weight or sum_hess <=0:
        return 0
    dw = - thresholdL1(sum_grad, p.reg_alpha)/ (sum_hess + p.reg_lambda)
    if p.max_delta_step !=0 and np.abs(dw) > p.max_delta_step:
        dw = np.copysign(p.max_delta_step, dw)
    return dw


def get_gain_given_weight(p, grad, hess, w):
    sum_grad = grad.sum()
    sum_hess = hess.sum()
    if sum_hess <= 0:
        return 0
    return -(2.0 * sum_grad * w + (sum_hess + p.reg_lambda) * (w * w))


def calc_split_gain(param, grad, hess, left):
    constraint = 0
    negative_infinity = np.iinfo(int).min
    g_l = grad[left]
    g_r = grad[~left]
    h_l = hess[left]
    h_r = hess[~left]
    wleft = get_weight(param, g_l, h_l)
    wright = get_weight(param, g_r, h_r)

    gain_l = get_gain_given_weight(param, g_l, h_l, wleft)
    gain_r = get_gain_given_weight(param, g_r, h_r, wright)
    gain = gain_l + gain_r

    if constraint == 0:
        return gain
    elif constraint > 0:
        return gain if wleft <= wright else negative_infinity
    else:
        return gain if wleft >= wright else negative_infinity


def get_gain(p, grad, hess):
    sum_grad = grad.sum()
    sum_hess = hess.sum()
    if sum_hess < p.min_child_weight:
        return 0
    if p.max_delta_step == 0.0:
        if p.reg_alpha == 0.0:
            return (sum_grad * sum_grad) / (sum_hess + p.reg_lambda)
        else:
            dw = thresholdL1(sum_grad, p.reg_alpha)
            return (dw * dw) / (sum_hess + p.reg_lambda)
    else:
        w = calc_weight(p, sum_grad, sum_hess)
        ret = get_gain_given_weight(p, sum_grad, sum_hess, w)
        if p.reg_alpha == 0:
            return ret
        else:
            return ret + p.reg_alpha * np.abs(w)


def get_loss(dat, p, grad, hess):
    loss = np.finfo(np.float).min
    best_ind = best_val = None
    for icol in range(dat.shape[1]):
        col = dat[:, icol].copy()
        ind_sort = np.argsort(col)
        col = col[ind_sort]
        gain = get_gain(p, grad, hess)
        for i, d in enumerate(col):
            if i == 0:
                continue
            if d == col[i-1]:
                continue
            left = col < d
            new_loss = calc_split_gain(p, grad[ind_sort], hess[ind_sort],
                                       left) - gain
            if new_loss > loss:
                loss = new_loss
                best_ind = icol
                best_val = (d + col[i-1])/2
    return loss, best_ind, best_val


class Booster:
    def __init__(self, data, obj_fn=None, base_score=0.5):
        self.data = data
        self.tparam = Gr
        self.nrow = data.shape[0]
        self.ncol = data.shape[1]
        self.grad = self.hess = None
        self.obj_ = obj_fn if obj_fn is not None else LinearSquareLoss()

        self.pos = np.zeros(self.nrow, dtype=int)
        # initial
        self.set_gpair(base_score)


    def set_gpair(self, pred=0.5):
        if isinstance(pred, float) or isinstance(pred, int):
            pred = np.full(data.shape[0], pred)
        self.grad = self.obj_.gradient(pred, y)
        self.hess = self.obj_.hessian(pred, y)

    def update(self):
        tree = RegTree()
        for depth in range()








if __name__ == "__main__":
    boston = datasets.load_boston()
    data = boston['data']
    X = data[:, :-1]
    y = data[:, -1]
    gg = LinearSquareLoss()
    base_score = 0.5

    base_s = np.full(y.shape, base_score)
    grad = gg.gradient(base_s, y)
    hess = gg.hessian(base_s, y)

    gpair = []
    stats = GradStats()

    for i in range(len(grad)):
        gpair.append(GradientPair(grad[i], hess[i]))
        stats.add(GradientPair(grad[i], hess[i]))

    bb = DMat(csc_matrix(X))
    nnn = bb.get_col(1)

    pos = positions(gpair)

    p = TrainParam()
    ev = TreeEvaluator(p, 12, -1).get_evaluator()
    snode_stats = get_snode(pos, gpair)
    snode_weight = ev.calc_weight(-1, p, snode_stats)
    snode_root_gain = ev.calc_gain(-1, p, snode_stats)

    bbb = get_loss(X, p, grad, hess)

    print(bbb)
    # w = calc_weight(p, gpair)
    # gain = calc_gain(p, stats)

    print(0)



    print(2)


