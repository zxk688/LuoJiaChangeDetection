import torch
import torchvision
from torch import Tensor, reshape, stack
from torch.nn import ( Module, ModuleList)


def _get_backbone(
    bkbn_name, pretrained, output_layer_bkbn, freeze_backbone
) -> ModuleList:
    # The whole model:
    entire_model = getattr(torchvision.models, bkbn_name)(
        pretrained=pretrained
    ).features

    # Slicing it:
    derived_model = ModuleList([])
    for name, layer in entire_model.named_children():
        derived_model.append(layer)
        if name == output_layer_bkbn:
            break

    # Freezing the backbone weights:
    if freeze_backbone:
        for param in derived_model.parameters():
            param.requires_grad = False
    return derived_model


class EfficientNetEncoder(Module):
    def __init__(
        self,
        bkbn_name="efficientnet_b4",
        pretrained=True,
        output_layer_bkbn="3",
        freeze_backbone=False,
    ):
        super().__init__()
        self._backbone = _get_backbone(
            bkbn_name, pretrained, output_layer_bkbn, freeze_backbone
        )
        
    def forward(self, ref: Tensor, test: Tensor) -> Tensor:
        features1, features2 = [], []
        features1.append(ref)
        features2.append(test)
        for num, layer in enumerate(self._backbone):
            ref, test = layer(ref), layer(test)#encoding
            features1.append(ref)
            features2.append(test)
        return features1, features2
