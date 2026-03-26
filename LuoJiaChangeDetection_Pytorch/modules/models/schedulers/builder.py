'''
Function:
    Implementation of SchedulerBuilder and BuildScheduler
Author:
    Zhenchao Jin
'''
import copy
from ...utils import BaseModuleBuilder
from .poly import Poly


from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

'''SchedulerBuilder'''
class SchedulerBuilder(BaseModuleBuilder):
    REGISTERED_MODULES = {
        'Poly': Poly, 'ReduceLROnPlateau': ReduceLROnPlateau,
        'StepLR': StepLR,
    }
    '''build'''
    def build(self, optimizer, scheduler_cfg):
        scheduler_cfg = copy.deepcopy(scheduler_cfg)
        scheduler_type = scheduler_cfg.pop('type')
        scheduler_cfg.pop('optimizer')
        scheduler = self.REGISTERED_MODULES[scheduler_type](optimizer=optimizer, **scheduler_cfg)
        return scheduler


'''BuildScheduler'''
BuildScheduler = SchedulerBuilder().build

