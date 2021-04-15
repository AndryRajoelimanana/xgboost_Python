from Booster import BoostLearner
from sklearn import datasets
from utils.simple_matrix import DMatrixSimple, DMatrix


class BoostLearnTask:
    def __init__(self):
        self.silent = 0
        self.use_buffer = 1
        self.num_round = 10
        self.save_period = 0
        self.eval_train = 0
        self.pred_margin = 0
        self.ntree_limit = 0
        self.dump_model_stats = 0
        self.task = "train"
        self.model_in = "NULL"
        self.model_out = "NULL"
        self.name_fmap = "NULL"
        self.name_pred = "pred.txt"
        self.name_dump = "dump.txt"
        self.model_dir_path = "./"
        self.data = None
        self.dcache = []
        self.learner = BoostLearner()

    def run(self, dmat):
        self.init_data(dmat)
        self.init_learner()
        self.task_train()

    def init_data(self, dmat):
        self.data = dmat
        self.dcache.append(dmat)
        self.learner.set_cache_data(self.dcache)

    def init_learner(self):
        self.learner.init_model()

    def task_train(self):
        self.learner.check_init(self.data)
        for i in range(self.num_round):
            self.learner.update_one_iter(i, self.data)
            # res = self.learner.eval_one_iter(i, )


if __name__ == '__main__':
    diabetes = datasets.load_diabetes()
    X, y = diabetes.data, diabetes.target
    dmat = DMatrix(X, label=y)
    # dmat.handle.fmat().init_col_access()
    bst = BoostLearnTask()
    bst.run(dmat.handle)
    print(3)
    # dmat.handle.fmat().init_col_access()
    # bst = Booster(params={}, cache=[dmat])
    # for i in range(3):
    #     bst.update(dmat)
    # bb = BoostLearner()
    # bb.set_cache_data(mattt)
    #    print('jj')
