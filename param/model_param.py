

class DataSplitMode:
    kAuto = 0
    kCol = 1
    kRow = 2

class ParamInitOption:
    kAllowUnknown = 0
    kAllMatch = 1
    kAllowHidden = 2


class dmlcParameter:
    def init(self, kwargs, option=ParamInitOption.kAllowHidden):
        self.run_init(kwargs, option)

    def init_allow_unknown(self, kwargs):
        unkonwn = self.run_init(kwargs, ParamInitOption.kAllowUnknown)
        return unkonwn

    def update_allow_unknown(self, kwargs):
        self.run_update(kwargs, ParamInitOption.kAllowUnknown, None)

    def run_init(self, kwargs, option):
        return 0

    def run_update(self, kwargs, option, selected_args):




class XGBoostParameter:
    def __init__(self):
        self.initialised_ = True

    def update_allow_uknown(self, kwargs):
        unknown = {}
        if self.initialised_:
            for k, v in kwargs.items():
                if hasattr(self, k):
                    setattr(self, k, v)
                else:
                    unknown[k] = v
        return unknown

    def get_initialised(self):
        return self.initialised_


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


class LearnerTrainParam(XGBoostParameter):
    def __init__(self):
        super(LearnerTrainParam, self).__init__()
        self.dsplit = DataSplitMode.kAuto
        self.disable_default_eval_metric = False
        self.booster = 'gbtree'
        self.objective = 'reg:squarederror'






