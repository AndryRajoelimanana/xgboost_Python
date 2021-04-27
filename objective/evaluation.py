import numpy as np
from info_class import MetaInfo

# learner/evaluation.h


class IEvaluator:
    def __init__(self):
        pass

    def eval(self, preds, info):
        pass

    def name(self):
        pass


def create_evaluator(name):
    if name == 'rmse':
        return EvalRMSE()
    elif name == 'error':
        return EvalError()
    elif name == "merror":
        return EvalMatchError()
    elif name == "logloss":
        return EvalLogLoss()
    # elif name == "auc":
    #     return EvalAuc()
    # elif name[:4] == "ams@":
    #     return EvalAMS(name)
    # elif name[:4] == "pre@":
    #     return EvalPrecision(name)
    # elif name[:7] == "pratio@":
    #     return EvalPrecisionRatio(name)
    # elif name[:3] == "map":
    #     return EvalMAP(name)
    # elif name[:4] == "ndcg":
    #     return EvalNDCG(name)
    # elif name[:3] == "ct-":
    #     return EvalCTest(CreateEvaluator(name + 3), name)


class EvalSet:
    """ learner/evaluation.h """

    def __init__(self):
        self.evals_ = []

    def add_eval(self, evname):
        for evals in self.evals_:
            if evname == evals.name():
                return
        self.evals_.append(create_evaluator(evname))

    def eval(self, evname, preds, info):
        result = ''
        for evals in self.evals_:
            res = evals.eval(preds, info)
            result += f'{evname}-{evals.name()}:{res} \n'
        return result

    def size(self):
        return len(self.evals_)


class EvalEWiseBase(IEvaluator):
    def __init__(self):
        super().__init__()
        pass

    def eval(self, preds, info):
        ndata = len(info.labels_)
        assert len(preds) % ndata == 0, "preds and labels_ # size"
        e_sum = 0
        wsum = 0
        for i in range(ndata):
            wt = info.get_weight(i)
            e_sum += self.eval_row(info.labels_[i], preds[i]) * wt
            wsum += wt
        return self.get_final(e_sum, wsum)

    @staticmethod
    def eval_row(label, pred):
        pass

    @staticmethod
    def get_final(esum, wsum):
        return esum / wsum


class EvalRMSE(EvalEWiseBase):
    def __init__(self):
        super().__init__()

    def name(self):
        return "rmse"

    def eval_row(self, label, pred):
        diff = label - pred
        return diff ** 2

    def get_final(self, esum, wsum):
        return np.sqrt(esum / wsum)


class EvalLogLoss(EvalEWiseBase):
    def __init__(self):
        super().__init__()

    def name(self):
        return "logloss"

    def eval_row(self, y, py):
        return - y * np.log(py) - (1.0 - y) * np.log(1 - py)


class EvalError(EvalEWiseBase):
    def __init__(self):
        super().__init__()

    def name(self):
        return "error"

    def eval_row(self, label, pred):
        if pred > 0.5:
            return 1 - label
        else:
            return label


class EvalMatchError(EvalEWiseBase):
    def __init__(self):
        super().__init__()

    def name(self):
        return "merror"

    def eval_row(self, label, pred):
        return int(int(pred) != int(label))


class EvalCTest(IEvaluator):
    def __init__(self, base, name):
        super().__init__()
        self.base_ = base
        self.name_ = name

    def name(self):
        return str(self.name_)

    def eval(self, preds, info):
        ngroup = len(preds) / len(info.labels_) - 1
        ndata = len(info.labels_)
        wsum = 0
        tpred = []
        tinfo = MetaInfo()
        for k in range(ngroup):
            for i in range(ndata):
                if info.info.fold_index[i] == k:
                    tpred.append(preds[i])

