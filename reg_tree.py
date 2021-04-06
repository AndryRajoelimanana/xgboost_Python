import numpy as np


class RegTree:
    pass


class Node:
    def __init__(self, cleft, cright, parent, split_ind,
                 split_cond, default_left):
        self.parent_ = parent
        self.cleft_ = cleft
        self.cright_ = cright
        self.set_parent(parent)
        self.sindex_ = self.split_cond = None
        self.set_split(split_ind, split_cond, default_left)
        self.leaf_value = None

    def set_parent(self, parent, is_left_child=True):
        if is_left_child:
            parent |= (1 << 31)
        self.parent_ = parent

    def set_split(self, split_ind, split_cond, default_left=False):
        if default_left:
            split_ind |= (1 << 31)
        self.sindex_ = split_ind
        self.split_cond = split_cond

    @property
    def defaultchild(self):
        if self.defaultleft():
            return self.cleft_
        else:
            return self.cright_

    def split_index(self):
        return self.sindex_ & ((1 << 31) - 1)

    def defaultleft(self):
        return (self.sindex_ >> 31) != 0

    def set_left_child(self, nid):
        self.cleft_ = nid

    def set_right_child(self, nid):
        self.cright_ = nid

    def parent(self):
        return self.parent_ & ((1 << 31) - 1)

    def is_left_child(self):
        return (self.parent_ & (1 << 31)) != 0

    def is_root(self):
        return self.parent_ == -1

    def mark_delete(self):
        self.sindex_ = np.iinfo(np.uint32)

    def set_parent(self, pidx, is_left_child=True):
        if is_left_child:
            pidx |= (1 << 31)
        self.parent_ = pidx

    def __eq__(self, other):
        return (self.parent_ == other.parent_ and self.cleft_ == other.cleft_
                and self.cright_ == other.cright_ and self.sindex_ ==
                other.sindex_ and self.leaf_value == other.leaf_value)


class TreeModel:
    def __init__(self):
        self.param = ToObject(max_depth=2, size_leaf_vector=0)
        self.param.num_nodes = 1
        self.param.num_roots = 1;
        self.param.num_deleted = 0;
        # nodes.resize(1);

    def get_node(self, nid):
        return self.nodes[nid]

    def InitModel(self):
        self.param.num_nodes = self.param.num_roots

    def add_childs(self, nid):
        self.nodes[nid]


class ToObject(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class Booster:
    def __init__(self, data):
        self.silent = 1
        self.init_model()
        self.set_cache_data(data)

    def pred(self, dmat, option_mask, ntree_limit, length):
        self.check_init_model()
        ypred = self.predict(dmat)
        self.length = ypred.shape[0]
        self.ypred = ypred
        return ypred

    def boost_one_iter(self, train, grad, hess):
        self.gpair_


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






class GradStats:
    def __init__(self, param):
        self.sum_grad = self.sum_hess = 0

    def add(self, grad, hess):
        self.sum_grad += grad
        self.sum_hess += hess

    def add_stats(self, gpair, info, ridx):
        b = gpair[ridx]
        self.add_pair(b)

    def add_pair(self, b):
        self.add(b.sum_grad, b.sum_hess)

    def set_substract(self, a, b):
        self.sum_grad = a.sum_grad - b.sum_grad
        self.sum_hess = a.sum_hess - b.sum_hess

    def empty(self):
        return self.sum_hess == 0


class Builder:
    def __init__(self, param):
        self.param = param

    def update(self, gpair, p_fmat, info, p_tree):
        self.init_data(gpair, p_fmat, info.root_index, p_tree)

    def init_data(self, gpair, fmat, root_index, p_tree):
        rowset = fmat.shape[0]
        position = np.zeros(gpair.shape[0])
        if root_index.shape[0] == 0:
            for i in range(rowset):
                position[rowset[i]] = 0
        else:
            for i in range(rowset):
                ridx = rowset[i]
                if self.gpair[ridx].hess < 0:
                    position[ridx] = -1
                else:
                    position[ridx] = root_index[ridx]
        self.feat_index = np.arange(num_col)




if __name__ == "__main__":
    # y0 = np.array([1, 0, 1, 0])
    # ypred0 = np.array([0.3, 0.5, 0.2, 0.3])
    # loss = SquareErrorLoss()
    # yy = np.random.uniform(size=(5, 3))
    # print(yy, loss.softmax(yy))
    # print(loss.get_gradient(y0, ypred0))
    nn = np.array([[1, 0, 3, 4],
       [0, 1, 3, 0],
       [1, 0, 0, 0],
       [0, 0, 1, 5]])
    mmm = Sparse_csr(nn.T)
    print(mmm[1])



