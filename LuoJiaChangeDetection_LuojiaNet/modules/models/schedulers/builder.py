import copy
from ...utils import BaseModuleBuilder
import luojianet.nn.learning_rate_schedule as lrs


'''SchedulerBuilder'''
class SchedulerBuilder(BaseModuleBuilder):
    REGISTERED_MODULES = {
        'ExponentialDecayLR': lrs.ExponentialDecayLR, 'CosineDecayLR': lrs.CosineDecayLR,
        'PolynomialDecayLR': lrs.PolynomialDecayLR,'InverseDecayLR': lrs.InverseDecayLR,
    }
    '''build'''
    def build(self, scheduler_cfg):
        scheduler_cfg = copy.deepcopy(scheduler_cfg)
        scheduler_type = scheduler_cfg.pop('type')

        scheduler = self.REGISTERED_MODULES[scheduler_type](**scheduler_cfg)
        return scheduler


'''BuildScheduler'''
BuildScheduler = SchedulerBuilder().build