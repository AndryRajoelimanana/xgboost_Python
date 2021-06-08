import numpy as np


# tree/model.h


class GBTree:
    def __init__(self):
        pass




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
