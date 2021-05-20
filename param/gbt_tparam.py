class GBTreeTrainParam:
    def __init__(self, process_type=0, predictor=0, tree_method=0):
        self.nthread = 0
        self.updater_seq = ['grow_colmaker']
        self.num_parallel_tree = 1
        self.updater_initialized = 0
        self.process_type = process_type
        self.predictor = predictor
        self.tree_method = tree_method

    def set_param(self, name, val):
        if name == 'updater' and val not in self.updater_seq:
            self.updater_seq = [val]
            self.updater_initialized = 0
        elif name == 'num_parallel_tree':
            self.num_parallel_tree = val
