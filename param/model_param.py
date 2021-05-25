

class DataSplitMode:
    kAuto = 0
    kCol = 1
    kRow = 2


class LearnerModelParam:
    def __init__(self, user_param, base_margin=0.5):

        self.base_score = base_margin
        self.num_feature = user_param.num_feature
        self.num_output_group = user_param.num_class
        if user_param.num_class == 0:
            self.num_output_group = 1
        self.num_class = self.num_output_group

    def initialized(self):
        return self.num_feature != 0


class LearnerModelParamLegacy:
    def __init__(self, base_score=0.5, num_feature=0, num_class=0):

        self.base_score = base_score
        self.num_feature = num_feature
        self.num_class = num_class


class LearnerTrainParam:
    def __init__(self):
        self.dsplit = DataSplitMode.kAuto
        self.disable_default_eval_metric = False
        self.booster = 'gbtree'
        self.objective = 'reg:squarederror'
