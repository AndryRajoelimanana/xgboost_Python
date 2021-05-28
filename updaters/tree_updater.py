from param.generic_param import GenericParameter
from params import *
from utils.random import ColumnSampler


from tree.tree_model import RegTree
from param.train_param import TrainParam, ColMakerTrainParam, NodeEntry
from param.train_param import SplitEntry
from utils.eval_fn import Evaluator
from utils.util import resize


rt_eps = 1e-5
rt_2eps = 2e-5
kRtEps = 1e-6


class TreeUpdater:
    def __init__(self):
        self.tparam_ = GenericParameter()

    def can_modify_tree(self):
        return False


def create_treeupdater(name, tparam):
    if name == 'grow_colmaker':
        updater = ColMaker()
    else:
        raise ValueError(f"Unknown GradientBooster: {name}")
    updater.tparam_ = tparam
    return updater


class ColMaker(TreeUpdater):
    def __init__(self):
        super(ColMaker, self).__init__()
        self.param_ = TrainParam()
        self.colmaker_train_param_ = ColMakerTrainParam()

    def configure(self, args):
        self.param_.update_allow_unknown(args)
        self.colmaker_train_param_.update_allow_unknown(args)

    def set_param(self, k, v):
        self.param_.set_param(k, v)

    def update(self, grad, hess, fmat, trees):
        # rescale learning rate according to size of trees

        lr = self.param_.learning_rate
        self.param_.learning_rate = lr / len(trees)
        param = self.param_
        cparam = self.colmaker_train_param_
        for tree in trees:
            builder = ColMaker.Builder(param, cparam)
            builder.update(grad, hess, fmat, tree)
        self.param_.learning_rate = lr

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
            self.col_samplers_ = ColumnSampler()

        def update(self, grad, hess, data, tree, f_weight=None):
            self.tree = tree
            self.data = data
            self.grad = grad
            self.hess = hess
            self.nrow = grad.shape[0]
            self.pos = self.positions()
            num_col = data.shape[1]
            if f_weight is None:
                f_weight = np.full(num_col, 1.0)
            self.snode_ = [NodeEntry() for _ in range(tree.param.num_nodes)]

            self.col_samplers_.init(num_col, f_weight,
                                    self.param.colsample_bynode,
                                    self.param.colsample_bylevel,
                                    self.param.colsample_bytree)

            for depth in range(self.param.max_depth):
                self.init_node_stat()
                for nid in self.q_expand_:
                    lw, rw = self.nodes_stat(nid, depth)
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

        def nodes_stat(self, nid, depth):
            lr = self.param.learning_rate
            curr_pos = self.pos == nid
            grad = self.grad[curr_pos]
            hess = self.hess[curr_pos]
            sum_grad = grad.sum()
            sum_hess = hess.sum()
            weight = calc_weight(self.param, sum_grad, sum_hess)
            root_gain = calc_gain(self.param, sum_grad, sum_hess)
            dat = self.data[curr_pos, :]
            loss, feat, val, l_s, r_s = self.find_split(depth, dat, grad, hess)
            best = SplitEntry(loss, feat, val, l_s, r_s)
            self.snode_[nid] = NodeEntry(sum_grad, sum_hess,
                                         root_gain, weight, best)

            if loss > kRtEps:
                left_w = self.eval_fn.get_weight(self.param, l_s[0], l_s[1])
                right_w = self.eval_fn.get_weight(self.param, r_s[0], r_s[1])
                return left_w * lr, right_w * lr
            else:
                return None, None

        def find_split(self, depth, dat, grad, hess):
            loss = np.finfo(np.float).min
            best_ind = best_val = left_sum = right_sum = None
            p = self.param
            feat_set = self.col_samplers_.get_feature_set(depth)

            for icol in feat_set:
                col = dat[:, icol].copy()
                ind_sort = np.argsort(col)
                col = col[ind_sort]
                new_grad = grad[ind_sort]
                new_hess = hess[ind_sort]
                gain = self.eval_fn.get_gain(p, grad.sum(), hess.sum())
                total_grad = grad.sum()
                total_hess = hess.sum()
                grad_right = total_grad
                hess_right = total_hess
                # statistics for nan missing value
                nan_stats = (grad[np.isnan(col)].sum(),
                             hess[np.isnan(col)].sum())

                for i, d in enumerate(col):
                    if np.isnan(d):
                        break
                    if i == 0 or d == col[i - 1]:
                        grad_right -= new_grad[i]
                        hess_right -= new_hess[i]
                        continue

                    # compute children gain for default left direction (
                    # missing values belong to left children node)
                    sleft = (total_grad - grad_right + nan_stats[0],
                             total_hess - hess_right + nan_stats[1])
                    sright = (grad_right, hess_right)
                    res_dleft = self.eval_fn.calc_split_gain(p, sleft, sright)

                    # compute children gain for default right direction (
                    # missing values belong to right children node)
                    sleft = (total_grad - grad_right, total_hess - hess_right)
                    sright = (grad_right + nan_stats[0], hess_right +
                              nan_stats[1])
                    res_dright = self.eval_fn.calc_split_gain(p, sleft, sright)

                    new_loss = max(res_dleft, res_dright) - gain

                    if new_loss > loss:
                        loss = new_loss
                        if res_dleft > res_dright:
                            best_ind = icol | (1 << 31)
                        else:
                            best_ind = icol
                        best_val = (d + col[i - 1]) / 2
                        left_sum = sleft
                        right_sum = sright

                    grad_right -= new_grad[i]
                    hess_right -= new_hess[i]

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

    @staticmethod
    def name():
        return "grow_colmaker"


