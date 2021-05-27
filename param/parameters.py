
class ParamInitOption:
    kAllowUnknown = 0
    kAllMatch = 1
    kAllowHidden = 2


class dmlcParameter:
    def init(self, kwargs, option=ParamInitOption.kAllowHidden):
        self.run_update(kwargs, option, None)

    def init_allow_unknown(self, kwargs):
        unknown_args = {}
        self.run_update(kwargs, ParamInitOption.kAllowUnknown, unknown_args)
        return unknown_args

    def update_allow_unknown(self, kwargs):
        unknown_args = {}
        self.run_update(kwargs, ParamInitOption.kAllowUnknown, unknown_args)
        return unknown_args

    def run_update(self, kwargs, option, unknown_args):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                if unknown_args is not None:
                    unknown_args[k] = v
                else:
                    if option != ParamInitOption.kAllowUnknown:
                        raise ValueError(f'Cannot find argument {k}')


class XGBoostParameter(dmlcParameter):
    def __init__(self):
        super(XGBoostParameter, self).__init__()
        self.initialised_ = True

    def get_initialised(self):
        return self.initialised_
