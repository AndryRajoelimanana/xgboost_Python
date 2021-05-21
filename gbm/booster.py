from sklearn import datasets
from params import *
from objective.loss_function import LinearSquareLoss
# from scipy.sparse import csc_matrix
# from tree.split_evaluator import TreeEvaluator
# from updaters.update_colmaker import ThreadEntry
from tree.tree_model import RegTree
from param.gbt_tparam import GBTreeTrainParam
from param.gbt_mparam import GBTreeModelParam
from param.train_param import TrainParam, ColMakerTrainParam, NodeEntry
from param.train_param import SplitEntry
from utils.eval_fn import Evaluator
from utils.util import resize, GenericParameter


kRtEps = 1e-6


def train(params, dtrain, num_boost_round=10, evals=(), obj=None, feval=None,
          xgb_model=None, callbacks=None, evals_result=None, maximize=None,
          verbose_eval=None, early_stopping_rounds=None):
    bst = Booster(params, dtrain)
    for i in range(num_boost_round):
        bst.update(dtrain, i, obj)


class Booster:
    # def __init__(self, data, obj_fn=None, base_score=0.5, num_boost_round=2):
    def __init__(self, param={}, cache=(), model_file=None):

        self.data = data
        self.mparam = GBTreeModelParam()
        self.tparam = GBTreeTrainParam()

        self.nrow = data.shape[0]
        self.ncol = data.shape[1]

        self.booster = 'gbtree'

        self.num_boost_round = num_boost_round
        self.grad = self.hess = None
        self.obj_ = obj_fn if obj_fn is not None else LinearSquareLoss()

        self.pos = np.zeros(self.nrow, dtype=int)
        # initial
        self.set_gpair(base_score)

    def set_param(self, params, value=None):
        for k, v in params.items():
            self.params


    def set_gpair(self, pred=0.5):
        if isinstance(pred, float) or isinstance(pred, int):
            pred = np.full(self.data.shape[0], pred)
        self.grad = self.obj_.gradient(pred, y)
        self.hess = self.obj_.hessian(pred, y)

    def update(self, train, i, obj=None):
        trees = [RegTree() for _ in range(self.num_boost_round)]
        data = self.data
        for tree in trees:
            colmaker = ColMaker()
            colmaker.update(grad0, hess0, X, tree)

        # for depth in range(self.mparam.)


class ColMaker:
    def __init__(self):
        self.param_ = TrainParam()
        self.colmaker_train_param_ = ColMakerTrainParam()

    def set_param(self, k, v):
        self.param_.set_param(k, v)

    def update(self, grad, hess, fmat, trees):
        lr = self.param_.learning_rate
        self.param_.learning_rate = lr / len(trees)
        # for tree in trees:
        param = self.param_
        cparam = self.colmaker_train_param_
        builder = ColMaker.Builder(param, cparam)
        builder.update(grad, hess, fmat, trees)
        self.build = builder

    class Builder:
        def __init__(self, param, cparam):
            self.param = param
            self.cparam = cparam
            self.grad = self.hess = None
            self.nrow = 0
            self.pos = self.data = None
            self.snode_ = []
            self.eval_fn = Evaluator()
            self.q_expand_ = [0]
            self.tree = RegTree()

        def update(self, grad, hess, data, tree):
            self.tree = tree
            self.data = data
            self.grad = grad
            self.hess = hess
            self.nrow = grad.shape[0]
            self.pos = self.positions()
            self.snode_ = [NodeEntry() for _ in range(tree.param.num_nodes)]

            for depth in range(self.param.max_depth):
                self.init_node_stat()
                for nid in self.q_expand_:
                    lw, rw = self.nodes_stat(nid)
                    if lw is None:
                        self.tree[nid].set_leaf(self.snode_[nid].weight *
                                                self.param.learning_rate)
                    else:
                        e = self.snode_[nid]
                        self.tree.expandnode(nid, e.best.split_index(),
                                             e.best.split_value,
                                             e.best.default_left(), e.weight,
                                             lw,
                                             rw, e.best.loss_chg, e.sum_hess,
                                             e.best.left_sum[1],
                                             e.best.right_sum[1], 0)
                self.reset_position()
                self.update_queue_expand()

        def init_node_stat(self):
            resize(self.snode_, self.tree.param.num_nodes, NodeEntry())

        def nodes_stat(self, nid):
            lr = self.param.learning_rate
            curr_pos = self.pos == nid
            grad = self.grad[curr_pos]
            hess = self.hess[curr_pos]
            sum_grad = grad.sum()
            sum_hess = hess.sum()
            weight = calc_weight(self.param, sum_grad, sum_hess)
            root_gain = calc_gain(self.param, sum_grad, sum_hess)
            dat = self.data[curr_pos, :]
            loss, feat, val, l_s, r_s = self.find_split(dat, grad, hess)
            print(f'nid: {nid}, loss: {loss}')
            best = SplitEntry(loss, feat, val, l_s, r_s)
            self.snode_[nid] = NodeEntry(sum_grad, sum_hess,
                                         root_gain, weight, best)

            if loss > kRtEps:
                left_w = self.eval_fn.get_weight(self.param, l_s[0], l_s[1])
                right_w = self.eval_fn.get_weight(self.param, r_s[0], r_s[1])
                return left_w * lr, right_w * lr
            else:
                return None, None

        def find_split(self, dat, grad, hess):
            loss = np.finfo(np.float).min
            best_ind = best_val = left_sum = right_sum = None
            p = self.param
            for icol in range(dat.shape[1]):
                col = dat[:, icol].copy()
                ind_sort = np.argsort(col)
                col = col[ind_sort]
                gain = self.eval_fn.get_gain(p, grad.sum(), hess.sum())
                for i, d in enumerate(col):
                    if i == 0:
                        continue
                    if d == col[i - 1]:
                        continue
                    left = col < d
                    res = self.eval_fn.calc_split_gain(p, grad[ind_sort],
                                                       hess[ind_sort], left)
                    new_loss = res[0] - gain

                    l_stat = res[1]
                    r_stat = res[2]
                    if new_loss > loss:
                        loss = new_loss
                        best_ind = icol
                        best_val = (d + col[i - 1]) / 2
                        left_sum = l_stat
                        right_sum = r_stat
            return loss, best_ind, best_val, left_sum, right_sum

        def positions(self):
            pos = np.zeros(self.nrow, dtype=int)
            pos[self.hess < 0] = -1
            return pos

        def reset_position(self):
            for nid in self.q_expand_:
                feat = self.tree[nid].split_index()
                val = self.tree[nid].split_cond()
                curr_pos = (np.abs(self.pos) == nid)
                col = self.data[:, feat]

                is_leaf = self.tree[nid].is_leaf()
                if not is_leaf:
                    i_left = curr_pos & (col < val)
                    i_right = curr_pos & (col >= val)
                    sign_l = np.sign(self.pos[i_left]) + (self.pos[i_left] == 0)
                    sign_r = np.sign(self.pos[i_right]) + (
                                self.pos[i_right] == 0)
                    self.pos[i_left] = self.tree[nid].left_child() * sign_l
                    self.pos[i_right] = self.tree[nid].right_child() * sign_r

                else:
                    if self.tree[nid].right_child() == -1:
                        self.pos[curr_pos] = -1 * np.abs(self.pos[curr_pos])
                    else:
                        sign_curr_pos = np.sign(self.pos[curr_pos]).copy()
                        if self.tree[nid].default_left():
                            cleft = self.tree[nid].left_child()
                            self.pos[curr_pos] = cleft * sign_curr_pos
                        else:
                            cright = self.tree[nid].right_child()
                            self.pos[curr_pos] = cright * sign_curr_pos

        def update_queue_expand(self):
            p_newnodes = []
            for nid in self.q_expand_:
                if not self.tree[nid].is_leaf():
                    p_newnodes.append(self.tree[nid].left_child())
                    p_newnodes.append(self.tree[nid].right_child())
            self.q_expand_ = p_newnodes


if __name__ == "__main__":
    boston = datasets.load_boston()
    data0 = boston['data']
    X = data0[:, :-1]
    y = data0[:, -1]
    gg = LinearSquareLoss()
    base_scores = 0.5

    base_s = np.full(y.shape, base_scores)
    grad0 = gg.gradient(base_s, y)
    hess0 = gg.hessian(base_s, y)
    trees0 = [RegTree()]

    colmaker = ColMaker()
    colmaker.update(grad0, hess0, X, trees0)



    # gpair = []
    # stats = GradStats()
    #
    # for i in range(len(grad)):
    #     gpair.append(GradientPair(grad[i], hess[i]))
    #     stats.add(GradientPair(grad[i], hess[i]))
    #
    # bb = DMat(csc_matrix(X))
    # nnn = bb.get_col(1)
    #
    # # pos = positions(gpair)
    #
    # p = TrainParam()
    # ev = TreeEvaluator(p, 12, -1).get_evaluator()
    # snode_stats = get_snode(pos, gpair)
    # snode_weight = ev.calc_weight(-1, p, snode_stats)
    # snode_root_gain = ev.calc_gain(-1, p, snode_stats)
    #
    # bbb = get_loss(X, p, grad, hess)

    # w = calc_weight(p, gpair)
    # gain = calc_gain(p, stats)

    print(0)

