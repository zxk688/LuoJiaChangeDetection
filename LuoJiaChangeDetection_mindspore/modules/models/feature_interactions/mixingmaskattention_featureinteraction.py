
import numpy as np
import mindspore
import mindspore.nn as nn
import mindspore.ops as F
from mindspore import Parameter,Tensor

class PixelwiseLinear(nn.Cell):
    def __init__(
        self,
        fin,
        fout,
        last_activation=None,
    ) -> None:
        assert len(fout) == len(fin)
        super().__init__()

        n = len(fin)
        self._linears = nn.SequentialCell(
            *[
                nn.SequentialCell(
                    nn.Conv2d(fin[i], fout[i], kernel_size=1, has_bias=True),
                    nn.PReLU()
                    if i < n - 1 or last_activation is None
                    else last_activation,
                )
                for i in range(n)
            ]
        )

    def construct(self, x: Tensor) -> Tensor:
        # Processing the tensor:
        return self._linears(x)


class MixingBlock(nn.Cell):
    def __init__(
        self,
        ch_in: int,
        ch_out: int,
    ):
        super().__init__()
        self._convmix = nn.SequentialCell(
            nn.Conv2d(ch_in, ch_out, 3, group=ch_out, padding=1,pad_mode='pad'),
            nn.PReLU(),
            nn.InstanceNorm2d(ch_out),
        )

    def construct(self, x: Tensor, y: Tensor) -> Tensor:
        # Packing the tensors and interleaving the channels:
        mixed = F.stack((x, y), axis=2)
        mixed = F.reshape(mixed, (x.shape[0], -1, x.shape[2], x.shape[3]))

        # Mixing:
        return self._convmix(mixed)


class MixingMaskAttentionBlock(nn.Cell):
    """use the grouped convolution to make a sort of attention"""

    def __init__(
        self,
        ch_in,
        ch_out,
        fin,
        fout,
        generate_masked: bool = False,
    ):
        super().__init__()
        self._mixing = MixingBlock(ch_in, ch_out)
        self._linear = PixelwiseLinear(fin, fout)
        self._final_normalization = nn.InstanceNorm2d(ch_out) if generate_masked else None
        self._mixing_out = MixingBlock(ch_in, ch_out) if generate_masked else None

    def construct(self, x: Tensor, y: Tensor) -> Tensor:
        z_mix = self._mixing(x, y)
        z = self._linear(z_mix)
        z_mix_out = 0 if self._mixing_out is None else self._mixing_out(x, y)

        return (
            z
            if self._final_normalization is None
            else self._final_normalization(z_mix_out * z)
        )
    
class MixingMaskAttentionInteraction(nn.Cell):
    def __init__(self):
        super().__init__()
        # Initialize mixing blocks:
        self._first_mix = MixingMaskAttentionBlock(6, 3, [3, 10, 5], [10, 5, 1])
        self._mixing_mask = nn.CellList(
            [
                MixingMaskAttentionBlock(48, 24, [24, 12, 6], [12, 6, 1]),
                MixingMaskAttentionBlock(64, 32, [32, 16, 8], [16, 8, 1]),
                MixingBlock(112, 56),
            ]
        )
        
    def construct(self, inputs):
        features1, features2 = inputs
        features = [self._first_mix(features1[0], features2[0])]

        for num in range(0,len(features1)-2):
            features.append(self._mixing_mask[num](features1[num+2], features2[num+2]))# interaction
        return features
    

