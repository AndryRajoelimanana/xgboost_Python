import numpy as np


def resize(lst, num, default=0):
    if num < len(lst):
        while len(lst) > num:
            lst.pop()
    else:
        for i in range(len(lst), int(num)):
            lst.append(default)


def sample_binary(p):
    return np.random.uniform() < p


def sigmoid(x):
    x = np.array(x)
    return 1.0/(1.0 + np.exp(-x))


def softmax(score):
    score = np.asarray(score, dtype=float)
    score_max = np.max(score)
    score = np.exp(score - score_max)
    score /= np.sum(score, axis=1)[:, np.newaxis]
    return score


def one_hot_encoding(y):
    y = np.asarray(y)
    y_labels = np.unique(y)
    return np.squeeze(np.eye(y_labels.shape[0])[y.reshape(-1)])


class GenericParameter:
    kCpuId = -1
    kDefaultSeed = 0

    def __init__(self, seed=0, seed_per_iteration=False, nthread=0,
                 fail_on_invalid_gpu_id=False, gpu_page_size=0,
                 validate_parameters=False):
        self.seed = seed
        self.seed_per_iteration = seed_per_iteration
        self.nthread = nthread
        self.gpu_id = -1
        self.fail_on_invalid_gpu_id = fail_on_invalid_gpu_id
        self.gpu_page_size = gpu_page_size
        self.validate_parameters = validate_parameters
        self.n_gpus = 0


class LearnerModelParam:
    def __init__(self, user_param, base_margin, base_score=0.5, num_feature=0,
                 num_output_group=0):

        self.base_margin
        self.base_score = base_score
        self.num_feature = num_feature
        self.num_output_group = num_output_group

