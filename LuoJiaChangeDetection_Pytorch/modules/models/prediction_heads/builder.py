import copy
from ...utils.modulebuilder import BaseModuleBuilder

from .fcn_head import FCNHead
from .identity_head import IdentityHead
from .metric_head import DistanceHead

'''PreheadBuilder'''
class PreheadBuilder(BaseModuleBuilder):
    REGISTERED_MODULES = {
        'FCNHead': FCNHead,  'IdentityHead':IdentityHead, 'DistanceHead': DistanceHead,
    }
    '''build'''
    def build(self, prehead_cfg):
        prehead_cfg = copy.deepcopy(prehead_cfg)
        if 'selected_indices' in prehead_cfg: prehead_cfg.pop('selected_indices')
        return super().build(prehead_cfg)


'''BuildPrehead'''
BuildPrehead = PreheadBuilder().build