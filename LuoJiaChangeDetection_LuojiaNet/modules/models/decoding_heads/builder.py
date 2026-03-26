import copy
from ...utils.modulebuilder import BaseModuleBuilder
from .fc_decoder import FCEFDecoder, FCSiamDecoder
from .fcnpp_decoder import FCNPPDecoder
from .fdcnn_decoder import FDCNNDecoder
from .hanet_decoder import HANetDecoder
from .snunet_decoder import SNUNetDecoder
from .dsamnet_decoder import DSAMNetDecoder
from .dminet_decoder import DMINetDecoder
from .bit_decoder import BITTrans
from .dsifn_decoder import CNNDecoderDeepSuper
from .rdpnet_decoder import RDPNetDecoder
from .a2net_decoder import A2NetDecoder
from .icifnet_decoder import ICIFDecoder
from .bisrnet_decoder import BiSRNetDecoder
from .aernet_decoder import AERNetDecoder
from .sunet_decoder import SUNetDecoder
from .changeformer_decoder import TransDecoder
from .tinycd_decoder import TinyCDDecoder

from .ussfcnet_decoder import CNNAttentionDecoder

'''DecoderBuilder'''
class DecoderBuilder(BaseModuleBuilder):
    REGISTERED_MODULES = {
        'FCEFDecoder': FCEFDecoder, 'FCSiamDecoder':FCSiamDecoder,'FCNPPDecoder':FCNPPDecoder,
        'FDCNNDecoder':FDCNNDecoder,'HANetDecoder':HANetDecoder, 'A2NetDecoder':A2NetDecoder,
        'SNUNetDecoder':SNUNetDecoder,'ICIFDecoder':ICIFDecoder,'TinyCDDecoder':TinyCDDecoder,
        'DSAMNetDecoder':DSAMNetDecoder,'DMINetDecoder':DMINetDecoder,'BITTrans':BITTrans,
        'CNNDecoderDeepSuper':CNNDecoderDeepSuper,'RDPNetDecoder':RDPNetDecoder, 'TransDecoder':TransDecoder,
        'BiSRNetDecoder':BiSRNetDecoder, 'AERNetDecoder':AERNetDecoder, 'SUNetDecoder':SUNetDecoder,
        'CNNAttentionDecoder':CNNAttentionDecoder
    }
    '''build'''
    def build(self, decoder_cfg):
        decoder_cfg = copy.deepcopy(decoder_cfg)
        if 'selected_indices' in decoder_cfg: decoder_cfg.pop('selected_indices')
        return super().build(decoder_cfg)


'''BuildDecoder'''
BuildDecoder = DecoderBuilder().build