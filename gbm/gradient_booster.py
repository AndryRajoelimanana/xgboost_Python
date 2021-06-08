from gbm.gbtree import GBTree, Dart
from gbm.gblinear import GBLinear


def create_gbm(name, generic_param, learner_model_param):
    if name == 'gbtree':
        bst = GBTree(learner_model_param)
    elif name == 'dart':
        bst = Dart(learner_model_param)
    elif name == 'gblinear':
        bst = GBLinear(learner_model_param)
    else:
        raise ValueError(f"Unknown GradientBooster: {name}")
    bst.generic_param_ = generic_param
    return bst
