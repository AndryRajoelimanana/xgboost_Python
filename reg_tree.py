import numpy as np

## tree/model.h

class RegTree:
    pass


class GBTree:
    def __init__(self):
        pass


class TreeModel:
    def __init__(self):
        self.param = Param()
        self.nodes = np.array(Node(), dtype=object)
        self.deleted_nodes = None

        self.param.num_nodes = 1;
        self.param.num_roots = 1;
        self.param.num_deleted = 0;

        self.node_stat = None
        self.split_cond = None
        self.stats = None
        self.leaf_vector = None

    def delete_node(self, nid):
        assert nid >= self.param.num_roots, "cannot delete root"
        self.deleted_nodes = np.append(self.deleted_nodes, nid)
        self.nodes[nid].set_parent(-1)
        self.param.num_deleted += 1

    def change_to_leaf(self, rid, value):
        mssg = "cannot delete a non terminal child"
        assert self.nodes[self.nodes[rid].cleft()].is_leaf(), mssg
        assert self.nodes[self.nodes[rid].cright()].is_leaf(), mssg
        self.deleted_nodes(self.nodes[rid].cleft())
        self.deleted_nodes(self.nodes[rid].cright())
        self.nodes[rid].set_leaf(value)

    def collapse_to_leaf(self):





class Param:
    def __init__(self):
        self.max_depth = 0
        self.size_leaf_vector = 0
        self.num_roots = self.num_nodes = self.num_deleted = None
        self.num_feature = None

    def set_param(self, name, val):
        setattr(self, name, val)


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
        self.info_={'leaf_value':None, 'split_cond':None}

    def cleft(self):
        return self.cleft_

    def cright(self):
        return self.cright_

    def cdefault(self):
        if self.default_left():
            return self.cleft_
        else:
            return self.cright_

    def set_parent(self, parent, is_left_child=True):
        if is_left_child:
            parent |= (1 << 31)
        self.parent_ = parent

    def set_split(self, split_ind, split_cond, default_left=False):
        if default_left:
            split_ind |= (1 << 31)
        self.sindex_ = split_ind
        self.info_['split_cond'] = split_cond

    @property
    def defaultchild(self):
        if self.default_left():
            return self.cleft_
        else:
            return self.cright_

    def split_index(self):
        return self.sindex_ & ((1 << 31) - 1)

    def default_left(self):
        return (self.sindex_ >> 31) != 0

    def set_left_child(self, nid):
        self.cleft_ = nid

    def set_right_child(self, nid):
        self.cright_ = nid

    def parent(self):
        return self.parent_ & ((1 << 31) - 1)

    def is_left_child(self):
        return (self.parent_ & (1 << 31)) != 0

    def is_leaf(self):
        return self.cleft_ == -1

    def leaf_value(self):
        return self.info_['leaf_value']

    def split_cond(self):
        return self.info_['split_cond']

    def is_root(self):
        return self.parent_ == -1

    def mark_delete(self):
        self.sindex_ = np.iinfo(np.uint32)

    def set_parent(self, pidx, is_left_child=True):
        if is_left_child:
            pidx |= (1 << 31)
        self.parent_ = pidx

    def set_leaf(self, value, right=-1):
        self.info_['leaf_value'] = value
        self.cleft_ = -1
        self.cright_ = right





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



