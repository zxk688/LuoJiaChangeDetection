import copy
from ...utils.modulebuilder import BaseModuleBuilder
from .cd_encoder_decoder import CDEncoderDecoder
from .new_model import NewNet
from .new2_model import New2Net
from .new3_model import new3net

'''ChangeDetectorBuilder'''
class ChangeDetectorBuilder(BaseModuleBuilder):
    REGISTERED_MODULES = {
        'CDEncoderDecoder': CDEncoderDecoder,'NewNet':NewNet,'New2Net':New2Net,'new3net':new3net,
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
