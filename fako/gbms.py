from utils.util import GenericParameter
from gbm.gbtree import GBTree
from gbm.gblinear import GBLinear


class GradientBooster:
    def __init__(self):
        self.generic_param_ = GenericParameter()

    def configure(self, cfg):
        pass

    def slice(self, layer_begin, layer_end, step, out, out_of_bound):
        raise Exception("Slice is not supported by current booster.")

    def allow_lazy_check_point(self):
        return False

    def boosted_rounds(self):
        pass

    def do_boost(self, p_fmat, in_gpair):
        pass

    def predict_batch(self, dmat, training, layer_begin, layer_end):
        pass

    def predict_leaf(self, dmat, layer_begin, layer_end):
        pass


def create_gbm(name, generic_param, learner_model_param):
    if name == 'gbtree':
        bst = GBTree(learner_model_param)
    elif name == 'gblinear':
        bst = GBLinear(learner_model_param)
    else:
        raise ValueError(f"Unknown GradientBooster: {name}")
    bst.generic_param_ = generic_param
    return bst


class ParamFieldInfo:
    def __init__(self, name=None, types=None, types_info_str=None, description='' ):
        self.name = name
        self.type = types
        self.type_info_str = types_info_str
        self.description = description


class PredictionCacheEntry:
    def __init__(self):
        pass


class dlmc_reg:
    def __init__(self):
        self.name = None
        self.arguments = []

    def set_body(self, body):
        self.body = body
        return self

    def describe(self, description):
        self.description = description
        return self

    def add_argument(self, name, types, description):
        info = ParamFieldInfo()
        info.name = name
        info.type = types
        info.description = description
        self.arguments.append(info)

    def add_arguments(self, args):
        self.arguments += args


class FunctionRegEntryBase:
    def __init__(self):
        self.arguments = []
        self.body = self.description = self.return_type = None

    def set_body(self, body):
        self.body = body
        return self

    def describe(self, description):
        self.description = description
        return self

    def add_argument(self, name, types, description):
        info = ParamFieldInfo()
        info.name = name
        info.type = types
        info.type_info_str = info.type
        info.description = description
        self.arguments.append(info)
        return self

    def add_arguments(self, args):
        self.arguments += args

    def set_return_type(self, types):
        self.return_type = types
        return self


class GradientBoosterReg(FunctionRegEntryBase):
    def __init__(self):
        pass


class Learner:
    pass


class Registry:
    def __init__(self):
        self.const_list_ = []
        self.entry_list_ = []
        self.fmap_ = {}

    def list(self):
        return self.const_list_

    def list_all_names(self):
        return self.fmap_.keys()

    def find(self, k):
        return self.fmap_[k]

    def get(self):
        pass

    def add_alias(self, k, alias):
        e = self.fmap_[k]
        if alias in self.fmap_.keys():
            assert e == self.fmap_[alias]
        else:
            self.fmap_[alias] = e

    def _register__(self, name):
        pass


