from param.parameters import XGBoostParameter
import numpy as np


class GenericParameter(XGBoostParameter):
    kCpuId = -1
    kDefaultSeed = 0

    def __init__(self, seed=kDefaultSeed, seed_per_iteration=False, nthread=0,
                 gpu_id=-1, fail_on_invalid_gpu_id=False, gpu_page_size=0,
                 n_gpus=0):
        super(GenericParameter, self).__init__()
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.seed_per_iteration = seed_per_iteration
        self.nthread = nthread
        self.gpu_id = gpu_id
        self.fail_on_invalid_gpu_id = fail_on_invalid_gpu_id
        self.gpu_page_size = gpu_page_size
        self.n_gpus = n_gpus


