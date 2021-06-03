import numpy as np
import numbers


def resize(lst, num, default=0):
    if num < len(lst):
        while len(lst) > num:
            lst.pop()
    else:
        for i in range(len(lst), int(num)):
            lst.append(default)


def to_lower(in_str):
    return ' '.join(re.split("([A-Z][^A-Z]*)", in_str)).strip().replace('  ', '_').lower()


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


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance
    Parameters
    ----------
    seed : None, int or instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)