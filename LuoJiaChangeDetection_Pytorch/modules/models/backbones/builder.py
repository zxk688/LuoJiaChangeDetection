import copy
from ...utils.modulebuilder import BaseModuleBuilder

from .resnet import ResnetMultilevelEncoder, Resnet18Encoder, AERNetEncoder, ResNetEncoder
from .mobilenet import MobileNetEncoder
from .vgg16 import VggEncoder
from .changeformer_encoder import TransEncoder
from .cnnattention_encoder import CNNAttentionEncoder
from .convmixer_encoder import ConvMixerEncoder
from .dualcnntrans_encoder import DualCNNandTransEncoder
from .efficientnet_encoder import EfficientNetEncoder
from .nested_encoder import NestedCNNEncoder
from .fcnpp_encoder import FCNPPEncoder
from .fc_encoder import FCEFEncoder, FCSiamEncoder
from .cnn_encoder import CNNEncoder
from .multiresolution_encoder import MultiresolutionEncoder
from .semanticreasoning_encoder import SemanticReasoningEncoder
from .semiencoder import SemiEncoder
from .mobilenet import MobileNetEncoder
'''BackboneBuilder'''
class BackboneBuilder(BaseModuleBuilder):
    REGISTERED_MODULES = {
        'ResnetMultilevelEncoder': ResnetMultilevelEncoder,  'MobilenetEncoder': MobileNetEncoder, 
        'VggEncoder': VggEncoder, 'Resnet18Encoder': Resnet18Encoder, 'TransEncoder': TransEncoder,
        'CNNAttentionEncoder': CNNAttentionEncoder, 'ConvMixerEncoder': ConvMixerEncoder,
        'DualCNNandTransEncoder': DualCNNandTransEncoder, 'EfficientNetEncoder': EfficientNetEncoder,
        'NestedCNNEncoder': NestedCNNEncoder, 'FCNPPEncoder': FCNPPEncoder, 'FCEFEncoder':FCEFEncoder,
        'FCSiamEncoder': FCSiamEncoder, 'CNNEncoder': CNNEncoder, 'AERNetEncoder':AERNetEncoder,
        'MultiresolutionEncoder': MultiresolutionEncoder, 'SemanticReasoningEncoder': SemanticReasoningEncoder,
        'SemiEncoder': SemiEncoder, 'ResNetEncoder': ResNetEncoder,'MobileNetEncoder':MobileNetEncoder,
    }
    '''build'''
    def build(self, backbone_cfg):
        backbone_cfg = copy.deepcopy(backbone_cfg)
        if 'selected_indices' in backbone_cfg: backbone_cfg.pop('selected_indices')
        return super().build(backbone_cfg)


'''BuildBackbone'''
BuildBackbone = BackboneBuilder().build