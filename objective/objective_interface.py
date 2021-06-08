from param.generic_param import GenericParameter


class ObjFunction:
    """
    IObjFunction xgboost.learner : learner/objective.h
    interface
    """

    def __init__(self):
        self.tparam_ = GenericParameter()

    def configure(self, args):
        pass

    def set_param(self, name, value):
        pass

    def get_gradient(self, preds, labels, weights, iters):
        pass

    def default_eval_metric(self):
        pass

    def pred_transform(self, io_preds):
        pass

    def eval_transform(self, io_pred):
        self.pred_transform(io_pred)

    def prob_to_margin(self, base_score):
        return base_score