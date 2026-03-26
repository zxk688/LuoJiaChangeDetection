import copy

from modules.utils.modulebuilder import BaseModuleBuilder
from .binary_cd_framework import BinaryChangeDetectionFramework
from .multiclass_cd_framework import MulticlassChangeDetectionFramework
from .semantic_cd_framework import SemanticChangeDetectionFramework

class FrameworkBuilder(BaseModuleBuilder):
    REGISTERED_MODULES = {
        'BinaryChangeDetectionFramework': BinaryChangeDetectionFramework, 'MulticlassChangeDetectionFramework':MulticlassChangeDetectionFramework,
        'SemanticChangeDetectionFramework':SemanticChangeDetectionFramework,
    }   
    '''build'''
    def build(self, configs):
        configs = copy.deepcopy(configs)
        cd_cfg = {  'configs': configs, 'type': configs['type'],
        }
        return super().build(cd_cfg)

'''BuildFramework'''
BuildFramework = FrameworkBuilder().build