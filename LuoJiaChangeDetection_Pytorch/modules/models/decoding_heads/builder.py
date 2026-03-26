import copy
from ...utils.modulebuilder import BaseModuleBuilder

from .dminet_decoder import DMINetDecoder
from .a2net_decoder import A2NetDecoder
from .bit_decoder import BITTrans
from .dsamnet_decoder import DSAMNetDecoder
from .changeformer_decoder import TransDecoder
from .dsifn_decoder import CNNDecoderDeepSuper
from .tinycd_decoder import TinyCDDecoder
from .snunet_decoder import SNUNetDecoder
from .rdpnet_decoder import RDPNetDecoder
from .icifnet_decoder import ICIFDecoder
from .hanet_decoder import HANetDecoder
from .ussfcnet_decoder import CNNAttentionDecoder
from .resunet_decoder import ResUnetDecoder
from .fc_decoder import FCEFDecoder, FCSiamDecoder
from .fdcnn_decoder import FDCNNDecoder
from .fcnpp_decoder import FCNPPDecoder
from .aernet_decoder import AERNetDecoder
from .sunet_decoder import SUNetDecoder
from .bisrnet_decoder import BiSRNetDecoder


'''DecoderBuilder'''
class DecoderBuilder(BaseModuleBuilder):
    REGISTERED_MODULES = {
        'DMINetDecoder': DMINetDecoder,  'A2NetDecoder': A2NetDecoder,
        'BITTrans': BITTrans, 'DSAMNetDecoder': DSAMNetDecoder, 'TransDecoder': TransDecoder,
        'CNNDecoderDeepSuper': CNNDecoderDeepSuper, 'TinyCDDecoder': TinyCDDecoder,
        'SNUNetDecoder': SNUNetDecoder, 'RDPNetDecoder':RDPNetDecoder, 'ICIFDecoder': ICIFDecoder,
        'HANetDecoder': HANetDecoder, 'CNNAttentionDecoder': CNNAttentionDecoder,
        'ResUnetDecoder': ResUnetDecoder, 'FCEFDecoder': FCEFDecoder, 'FCSiamDecoder':FCSiamDecoder,
        'FDCNNDecoder': FDCNNDecoder, 'FCNPPDecoder': FCNPPDecoder, 'AERNetDecoder': AERNetDecoder,
        'SUNetDecoder': SUNetDecoder, 'BiSRNetDecoder': BiSRNetDecoder,
    }
    '''build'''
    def build(self, decoder_cfg):
        decoder_cfg = copy.deepcopy(decoder_cfg)
        if 'selected_indices' in decoder_cfg: decoder_cfg.pop('selected_indices')
        return super().build(decoder_cfg)


'''BuildDecoder'''
BuildDecoder = DecoderBuilder().build