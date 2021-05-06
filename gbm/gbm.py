from utils.util import GenericParameter, LearnerModelParam


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

    def do_boost(self, p_fmat, in_gpair, PredictionCacheEntry):
        pass

    def predict_batch(self, dmat, out_preds, training, layer_begin, layer_end):
        pass

    def predict_leaf(dmat, out_preds, layer_begin, layer_end):
        pass

    def create(self, name, generic_param, learner_model_param):
        # TODO
        generic_param = GenericParameter()
        learner_model_param = LearnerModelParam()
        # end TODO


class ParamFieldInfo:
    def __init__(self, name=None, types=None, types_info_str=None, description='' ):
        self.name=name
        self.type = types
        self.type_info_str = types_info_str
        self.description = description



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



class GradientBoosterReg(dlmc_reg):
    pass

class Learner:

