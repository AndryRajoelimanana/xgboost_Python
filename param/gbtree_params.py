from param.parameters import XGBoostParameter, dmlcParameter


class GBTreeTrainParam(XGBoostParameter):
    def __init__(self, process_type=0, predictor=0, tree_method=0):
        super().__init__()
        self.updater_seq = 'grow_colmaker'
        self.num_parallel_tree = 1
        self.process_type = process_type
        self.predictor = predictor
        self.tree_method = tree_method


class GBTreeModelParam(dmlcParameter):
    def __init__(self):
        super(GBTreeModelParam, self).__init__()
        self.num_trees = 0
        self.size_leaf_vector = 0


class DartTrainParam(XGBoostParameter):
    def __init__(self, sample_type=0, normalize_type=0, rate_drop=0,
                 one_drop=False, skip_drop=0.0, learning_rate=0.3):
        super().__init__()
        self.sample_type = sample_type
        self.normalize_type = normalize_type
        self.rate_drop = rate_drop
        self.one_drop = one_drop
        self.skip_drop = skip_drop
        self.learning_rate = learning_rate
        self.eta = self.learning_rate
