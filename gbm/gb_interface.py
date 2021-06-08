

class GradientBooster:
    def __init__(self):
        self.generic_param_ = None

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
