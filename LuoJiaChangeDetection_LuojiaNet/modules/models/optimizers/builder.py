import copy
import luojianet.nn.optim as optim
from ...utils.modulebuilder import BaseModuleBuilder
from .paramsconstructor import BuildParamsconstructor


'''OptimizerBuilder'''
class OptimizerBuilder(BaseModuleBuilder):
    REGISTERED_MODULES = {
        'SGD': optim.SGD, 'Adam': optim.Adam, 'Adadelta': optim.Adadelta,
    }
    '''build'''
    def build(self, model, scheduler, optimizer_cfg):
        # parse config
        optimizer_cfg = copy.deepcopy(optimizer_cfg)
        optimizer_type = optimizer_cfg.pop('type')
        # params_rules, filter_params = optimizer_cfg.pop('params_rules', {}), optimizer_cfg.pop('filter_params', False)
        filter_params = False
        params_rules = {}
        # build params_constructor
        params_constructor = BuildParamsconstructor(params_rules=params_rules, filter_params=filter_params, optimizer_cfg=optimizer_cfg)
        # obtain params
        optimizer_cfg['params'] = params_constructor(model=model)
        # build optimizer
        if scheduler!=None:
            optimizer = self.REGISTERED_MODULES[optimizer_type](lr = scheduler, **optimizer_cfg)
        else:
            optimizer = self.REGISTERED_MODULES[optimizer_type](**optimizer_cfg)
        # optimizer = self.REGISTERED_MODULES[optimizer_type]
        # return
        return optimizer


'''BuildOptimizer'''
BuildOptimizer = OptimizerBuilder().build