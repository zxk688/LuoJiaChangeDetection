import copy
from ..utils.modulebuilder import BaseModuleBuilder

from .base import BaseChangeDetectionDataset
from .folder_splitted import FolderSplittedDataset
from .second import SECONDDataset
from .s2looking import S2LookingDataset
from .multisource import MultisourceDataset
from .hiucd import HiUCDDataset
from .hrscd import HRSCDDataset
from .htcd import HTCDDataset
from .wusu import OpenWUSU512
from .text_splitted import TextSplittedDataset
from .hyperspectral import HyperspectralDataset
from .oscd import OSCDDataset
from .x2view import X2ViewDataset
class DatasetBuilder(BaseModuleBuilder):
    REGISTERED_MODULES = {
        'BaseChangeDetectionDataset': BaseChangeDetectionDataset,  'FolderSplittedDataset': FolderSplittedDataset, 
        'SECONDDataset':SECONDDataset, 'S2LookingDataset':S2LookingDataset, 'MultisourceDataset':MultisourceDataset,
        'HiUCDDataset':HiUCDDataset, 'HRSCDDataset':HRSCDDataset, 'HTCDDataset':HTCDDataset, 'OpenWUSU512':OpenWUSU512,
        'TextSplittedDataset':TextSplittedDataset,'HyperspectralDataset':HyperspectralDataset,'OSCDDataset':OSCDDataset,
        'X2ViewDataset':X2ViewDataset,
    }
    '''build'''
    def build(self, mode, logger_handle, dataset_cfg):
        dataset_cfg = copy.deepcopy(dataset_cfg)
        # train_cfg, test_cfg = dataset_cfg.pop('train', {}), dataset_cfg.pop('test', {})
        # dataset_cfg.update(train_cfg if mode == 'train' else test_cfg)
        module_cfg = {
            'mode': mode, 'logger_handle': logger_handle, 'dataset_cfg': dataset_cfg, 'type': dataset_cfg['type'],
        }
        return super().build(module_cfg)

'''BuildDataset'''
BuildDataset = DatasetBuilder().build