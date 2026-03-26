import copy

from .text_splitted import TextSplittedDataset
from .folder_splitted import FolderSplittedDataset

from .wusu import OpenWUSU512
from .htcd import HTCDDataset
from .onlineclip import OnlineClipDataset
from .multisource import MultisourceDataset
from .oscd import OSCDDataset
from .hrscd import HRSCDDataset
from .second import SECONDDataset
from .s2looking import S2LookingDataset
from .hiucd import HiUCDDataset
from .semi import SemiDataset
from .captioning import CaptionDataset
from .cdrlsa import CDRLDataset
from .hyperspectral import HyperspectralDataset
from .ssl import SSLCDDataset
from .x2view import X2ViewDataset


from ..utils.modulebuilder import BaseModuleBuilder
from .base import BaseChangeDetectionDataset

class DatasetBuilder(BaseModuleBuilder):
    REGISTERED_MODULES = {
        'BaseChangeDetectionDataset': BaseChangeDetectionDataset,  'FolderSplittedDataset': FolderSplittedDataset, 
        'TextSplittedDataset': TextSplittedDataset, 'OpenWUSU512': OpenWUSU512,'HTCDDataset':HTCDDataset, 
        'OnlineClipDataset': OnlineClipDataset, 'MultisourceDataset': MultisourceDataset,
        'OSCDDataset': OSCDDataset, 'HRSCDDataset': HRSCDDataset, 'SECONDDataset': SECONDDataset,
        'S2LookingDataset': S2LookingDataset, 'HiUCDDataset': HiUCDDataset, 'SemiDataset': SemiDataset,
        'CaptionDataset': CaptionDataset, 'CDRLDataset': CDRLDataset, 'HyperspectralDataset':HyperspectralDataset,
        'SSLCDDataset': SSLCDDataset, 'X2ViewDataset': X2ViewDataset,
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