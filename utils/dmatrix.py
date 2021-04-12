from info_class import MetaInfo

# learner/dmatrix.h


class DMatrix:
    def __init__(self, magic):
        self.magic = magic
        self.cache_learner_ptr = None
        self.info = MetaInfo()

    def fmat(self):
        pass
