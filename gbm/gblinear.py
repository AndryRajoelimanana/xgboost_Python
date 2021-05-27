from gbm.gbms import GradientBooster


class GBLinear(GradientBooster):
    def __init__(self, booster_config):
        super(GBLinear, self).__init__()