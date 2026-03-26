import copy
from ...utils.modulebuilder import BaseModuleBuilder

from .normal_featureinteraction_neck import FeatureInteractionNeck
from .temporal_fusion import TemporalFusionInteraction
from .crossattention_featureinteraction import CrossAttentionInteraction
from .mixingmaskattention_featureinteraction import MixingMaskAttentionInteraction

'''FeatureInteractionBuilder'''
class FeatureInteractionBuilder(BaseModuleBuilder):
    REGISTERED_MODULES = {
        'normal': FeatureInteractionNeck, 'TemporalFusionInteraction':TemporalFusionInteraction,
        'CrossAttentionInteraction':CrossAttentionInteraction, 'MixingMaskAttentionInteraction':MixingMaskAttentionInteraction,
    }
    '''build'''
    def build(self, featureinteraction_cfg):
        featureinteraction_cfg = copy.deepcopy(featureinteraction_cfg)
        if 'selected_indices' in featureinteraction_cfg: featureinteraction_cfg.pop('selected_indices')
        return super().build(featureinteraction_cfg)


'''BuildFeatureInteraction'''
BuildFeatureInteraction = FeatureInteractionBuilder().build