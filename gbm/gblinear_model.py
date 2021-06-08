import numpy as np


class GBLinearModel:
    def __init__(self, learner_model_param):
        self.learner_model_param = learner_model_param
        self.num_boosted_rounds = 0
        self.weight = np.empty(0)

    def configure(self, args):
        pass

    def lazy_init_model(self):
        if self.weight.shape[0] != 0:
            return
        else:
            num_feature = self.learner_model_param.num_feature
            ngroup = self.learner_model_param.num_output_group
            self.weight = np.zeros((num_feature + 1, ngroup))

    def bias(self):
        return self.weight[-1:, :]

    def __getitem__(self, item):
        return self.weight[item, :]

