from param.parameters import dmlcParameter


class TreeParam(dmlcParameter):
    def __init__(self):
        super(TreeParam, self).__init__()
        self.size_leaf_vector = 0
        self.num_nodes = 1
        self.num_deleted = 0
        self.num_feature = None

    def __eq__(self, b):
        return (self.num_nodes == b.num_nodes) and (
                self.num_deleted == b.num_deleted) and (
                       self.num_feature == b.num_feature) and (
                       self.size_leaf_vector == b.size_leaf_vector)
