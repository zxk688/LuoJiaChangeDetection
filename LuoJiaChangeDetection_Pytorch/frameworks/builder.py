import copy

from modules.utils.modulebuilder import BaseModuleBuilder

from .binary_cd_framework import BinaryChangeDetectionFramework


class FrameworkBuilder(BaseModuleBuilder):
    REGISTERED_MODULES = {
        'BinaryChangeDetectionFramework': BinaryChangeDetectionFramework, 
    }
    '''build'''
    def build(self, configs):
        configs = copy.deepcopy(configs)
        cd_cfg = {  'configs': configs, 'type': configs['type'],
        }
        return super().build(cd_cfg)

'''BuildFramework'''
BuildFramework = FrameworkBuilder().build