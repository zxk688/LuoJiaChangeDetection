import copy
from ...utils.modulebuilder import BaseModuleBuilder
from .cd_encoder_decoder import CDEncoderDecoder



'''ChangeDetectorBuilder'''
class ChangeDetectorBuilder(BaseModuleBuilder):
    REGISTERED_MODULES = {
        'CDEncoderDecoder': CDEncoderDecoder, 
    }
    '''build'''
    def build(self, changedetector_cfg):
        changedetector_cfg = copy.deepcopy(changedetector_cfg)
        changedetector_type = changedetector_cfg.pop('type')

        changedetector = self.REGISTERED_MODULES[changedetector_type](changedetector_cfg = changedetector_cfg)
        # changedetector = self.REGISTERED_MODULES[changedetector_type]()
        return changedetector


'''BuildChangeDetector'''
BuildChangeDetector = ChangeDetectorBuilder().build
