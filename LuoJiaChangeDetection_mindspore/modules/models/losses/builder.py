
from ...utils import BaseModuleBuilder
from mindspore.nn import BCELoss, CrossEntropyLoss, MSELoss
from .userdefined import SCDLoss,BCEWLLoss
'''LossBuilder'''
class LossBuilder(BaseModuleBuilder):
    REGISTERED_MODULES = {

        'CrossEntropyLoss': CrossEntropyLoss, 'BCELoss': BCELoss,'SCDLoss':SCDLoss,
        'MSELoss': MSELoss,'BCEWLLoss': BCEWLLoss}
    '''build'''
    def build(self, loss_cfg):
        return super().build(loss_cfg)


'''BuildLoss'''
BuildLoss = LossBuilder().build