from .l1loss import L1Loss
from .klloss import KLDivLoss
from .diceloss import DiceLoss
from .lovaszloss import LovaszLoss
from ...utils import BaseModuleBuilder
from .focalloss import SigmoidFocalLoss
from .cosinesimilarityloss import CosineSimilarityLoss
from .bcewlloss import BCEWLLoss
# from .celoss import CrossEntropyLoss, BinaryCrossEntropyLoss
from torch.nn import BCELoss, CrossEntropyLoss, MSELoss

'''LossBuilder'''
class LossBuilder(BaseModuleBuilder):
    REGISTERED_MODULES = {
        'L1Loss': L1Loss, 'DiceLoss': DiceLoss, 'KLDivLoss': KLDivLoss, 'LovaszLoss': LovaszLoss,
        'CrossEntropyLoss': CrossEntropyLoss, 'SigmoidFocalLoss': SigmoidFocalLoss,
        'CosineSimilarityLoss': CosineSimilarityLoss, 'BinaryCrossEntropyLoss': BCELoss,
        'MSELoss': MSELoss, 'BCEWLLoss': BCEWLLoss,
    }
    '''build'''
    def build(self, loss_cfg):
        return super().build(loss_cfg)


'''BuildLoss'''
BuildLoss = LossBuilder().build