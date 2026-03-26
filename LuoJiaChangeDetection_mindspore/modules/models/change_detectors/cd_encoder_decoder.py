
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import ops as F
from ..backbones import BuildBackbone
from ..decoding_heads import BuildDecoder
from ..feature_interactions import BuildFeatureInteraction
from ..prediction_heads import BuildPrehead
import copy

class CDEncoderDecoder(nn.Cell):
    def __init__(self,
                 changedetector_cfg: dict = None):
        super(CDEncoderDecoder, self).__init__()
        
        backbone = changedetector_cfg['backbone']
        decode_head = changedetector_cfg['decode_head']
        feature_interaction = changedetector_cfg['feature_interaction']
        prediction_head = changedetector_cfg['prediction_head']
        self.CONNECTED = changedetector_cfg['CONNECTED']
        
        
        if self.CONNECTED:
            self.CONNECTED_temporal = changedetector_cfg['CONNECTED_Temporal']
        self.backbone = BuildBackbone(backbone_cfg=copy.deepcopy(backbone))
        self.decode_head = BuildDecoder(decoder_cfg=copy.deepcopy(decode_head))
        if feature_interaction != None:
            self.feature_interaction = BuildFeatureInteraction(featureinteraction_cfg=copy.deepcopy(feature_interaction))
        else:
            self.feature_interaction = None
        if prediction_head != None:
            self.prediction_head = BuildPrehead(prehead_cfg=copy.deepcopy(prediction_head))
        else: 
            self.prediction_head = None
            
    def construct(self,
                 inputs):
       

        # feed to backbone network
        backbone_outputs = self.backbone(inputs)
            
        if self.feature_interaction != None:
            # feed to feature interactions
            feaint_outputs = self.feature_interaction(backbone_outputs)
            
            if not self.CONNECTED:
                # feed to decoder
                decoder_outputs = self.decode_head(feaint_outputs)
            else:  #multiple connections AERNet, BiSRNet, siam
                if self.CONNECTED_temporal == 'T2':
                    #backbone connected into the decoder        
                    decoder_outputs = self.decode_head(backbone_outputs[-1],feaint_outputs)
                elif self.CONNECTED_temporal == 'T1T2':
                    decoder_outputs = self.decode_head(backbone_outputs,feaint_outputs)
            if self.prediction_head != None:
                # feed to the prediction heads
                predictions = self.prediction_head(decoder_outputs)
            else:
                predictions = decoder_outputs
        else:
            # feed to decoder
            decoder_outputs = self.decode_head(backbone_outputs)
            if self.prediction_head != None:
                # feed to the prediction heads
                predictions = self.prediction_head(decoder_outputs)
            else:
                predictions = decoder_outputs

        return predictions
        
