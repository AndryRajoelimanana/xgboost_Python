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