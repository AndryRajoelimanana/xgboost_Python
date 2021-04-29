from gbm.gbm import GradientBooster
from tree.gbtree import GBTreeTrainParam, TreeProcessType
from gbm.gbtree_model import GBTreeModel



class GBTree(GradientBooster):
    def __init__(self, booster_config):
        # TODO
        self.model_ = GBTreeModel()
        # end to do
        self.tparam_ = GBTreeTrainParam()
        self.showed_updater_warning_ = False
        self.specified_updater_ = False
        self.configured_ = False
        self.updaters_ = []
        self.cpu_predictor_ = None
        self.cfg_ = None

    def configure(self, cfg):
        self.cfg_ = cfg
        updater_seq = self.tparam_.updater_seq
        for k, v in cfg:
            setattr(self.tparam_, k, v)
        self.model_.configure(cfg)
        if self.tparam_.process_type == TreeProcessType.kUpdate:
            self.model_.init_trees_to_update()

    def do_boost(self, p_fmat, in_gpair, predt):
        new_trees = []
        ngroup = self.model_.learner_model_param.num_output_group

    def configure_with_known_data(self, cfg, fmat):
        updater_seq = self.tparam_.updater_seq
        for k, v in cfg:
            setattr(self.tparam_, k, v)