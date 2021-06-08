from fako.info_class import MetaInfo

# learner/dmatrix.h


class DataMatrix:
    """ learner/dmatrix.h  xgboost.learner"""
    def __init__(self, magic):
        self.magic = magic
        self.cache_learner_ptr_ = None
        self.info = MetaInfo()

    def fmat(self):
        pass
