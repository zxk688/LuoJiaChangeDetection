import copy
from ...utils.modulebuilder import BaseModuleBuilder

from .fc_encoder import FCEFEncoder, FCSiamEncoder
from .fcnpp_encoder import FCNPPEncoder
from .convmixer_encoder import ConvMixerEncoder
from .cnn_encoder import CNNEncoder
from .vgg16 import VggEncoder
from .nested_encoder import NestedCNNEncoder
from .semanticreasoning_encoder import SemanticReasoningEncoder
from .resnet import ResnetMultilevelEncoder,ResNetEncoder, AERNetEncoder
from .mobilenet import MobileNetEncoder
from .multiresolution_encoder import MultiresolutionEncoder
from .dualcnntrans_encoder import DualCNNandTransEncoder
from .changeformer_encoder import TransEncoder
from .efficientnet_encoder import EfficientNetEncoder

from .cnnattention_encoder import CNNAttentionEncoder

'''BackboneBuilder'''
class BackboneBuilder(BaseModuleBuilder):
    REGISTERED_MODULES = {
    'FCEFEncoder':FCEFEncoder,
        'FCSiamEncoder': FCSiamEncoder, 'FCNPPEncoder':FCNPPEncoder,'CNNEncoder':CNNEncoder,'ConvMixerEncoder':ConvMixerEncoder,
        'NestedCNNEncoder':NestedCNNEncoder,'MobileNetEncoder':MobileNetEncoder,
        'ResnetMultilevelEncoder':ResnetMultilevelEncoder,'ResNetEncoder':ResNetEncoder,'VggEncoder':VggEncoder,
        'SemanticReasoningEncoder': SemanticReasoningEncoder, 'AERNetEncoder':AERNetEncoder,
        'MultiresolutionEncoder':MultiresolutionEncoder, 'DualCNNandTransEncoder':DualCNNandTransEncoder,
        'TransEncoder':TransEncoder, 'EfficientNetEncoder':EfficientNetEncoder,
        'CNNAttentionEncoder':CNNAttentionEncoder,
    }   
    '''build'''
    def build(self, backbone_cfg):
        backbone_cfg = copy.deepcopy(backbone_cfg)
        if 'selected_indices' in backbone_cfg: backbone_cfg.pop('selected_indices')
        return super().build(backbone_cfg)


'''BuildBackbone'''
BuildBackbone = BackboneBuilder().build