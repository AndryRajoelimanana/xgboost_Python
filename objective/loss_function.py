from objective.regression_obj import LinearSquareLoss
from objective.multiclass_obj import SoftmaxMultiClassObj


def create_objective(name):
    dict_obj = {"multi:softprob": SoftmaxMultiClassObj(True),
                "multi:softmax": SoftmaxMultiClassObj(False),
                "reg:squaredlerror": LinearSquareLoss()
                }
    return dict_obj[name]

