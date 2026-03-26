import mindspore.nn as nn
import mindspore.ops as F
from mindspore.nn import LossBase
from mindspore import Tensor
import numpy as np
from mindspore import dtype as mstype
from mindspore.nn import CrossEntropyLoss

class ChangeSimilarity(nn.Cell):
    """input: x1, x2 multi-class predictions, c = class_num
       label_change: changed part
    """
    def __init__(self, reduction='mean'):
        super(ChangeSimilarity, self).__init__()
        self.loss_f = nn.CosineEmbeddingLoss(margin=0., reduction=reduction)
        
    def construct(self, x1, x2, label_change):
        b,c,h,w = x1.shape
        x1 = F.Softmax(axis=1)(x1)
        x2 = F.Softmax(axis=1)(x2)
        x1 = x1.permute(0,2,3,1)
        x2 = x2.permute(0,2,3,1)
        x1 = F.reshape(x1,[b*h*w,c])
        x2 = F.reshape(x2,[b*h*w,c])
        
        label_unchange = ~label_change.bool()
        target = label_unchange.float()
        target = target - label_change.float()
        target = F.reshape(target,[b*h*w])
        
        loss = self.loss_f(x1, x2, target)
        return loss
class weighted_BCE_logits(nn.Cell):
    def __init__(self):
        self.weight_pos=0.25
        self.weight_neg=0.75
        super(weighted_BCE_logits, self).__init__()
    def construct(self,logit_pixel, truth_pixel):
        logit = logit_pixel.view(-1)
        truth = truth_pixel.view(-1)
        assert(logit.shape==truth.shape)

        loss = F.binary_cross_entropy_with_logits(logit, truth, Tensor(np.ones_like(logit.shape), mstype.float32),
                                                  Tensor(np.ones_like(logit.shape), mstype.float32),reduction='none')
        
        pos = (truth>0.5).float()
        neg = (truth<0.5).float()
        pos_num = pos.sum() + 1e-12
        neg_num = neg.sum() + 1e-12
        loss = (self.weight_pos*pos*loss/pos_num + self.weight_neg*neg*loss/neg_num).sum()

        return loss
    
class SCDLoss(LossBase):
    def __init__(self):
        super(SCDLoss, self).__init__()
        self.loss_seg = CrossEntropyLoss()
        self.loss_bn = weighted_BCE_logits()
        self.loss_sc = ChangeSimilarity()
        
    def construct(self, predict, target):

        labels_A, labels_B, labels_bn = target[:,0-7],target[:,7-14],target[:,-1]

        outputs_A, outputs_B, out_change = predict[0],predict[1],predict[2]
        loss_seg = self.loss_seg(outputs_A, labels_A.astype(mstype.int32)) * 0.5
        +  self.loss_seg(outputs_B, labels_B.astype(mstype.int32)) * 0.5 
        loss_bn = self.loss_bn(out_change, labels_bn)
        loss_sc = self.loss_sc(outputs_A[:,1:], outputs_B[:,1:], labels_bn)
        total_loss = loss_seg + loss_bn + loss_sc
        return total_loss

'''BCEWLLoss'''
class BCEWLLoss(nn.Cell):
    def __init__(self, scale_factor=1.0, temperature=1, reduction='mean', log_target=False, lowest_loss_value=None):
        super(BCEWLLoss, self).__init__()
        self.reduction = reduction
        self.log_target = log_target
        self.temperature = temperature
        self.scale_factor = scale_factor
        self.lowest_loss_value = lowest_loss_value
    '''forward'''
    def construct(self, prediction, target):
        loss = F.binary_cross_entropy_with_logits(prediction, target)
        return loss
    
# net = SCDLoss()
# np_array = np.random.random(size=(10,7,256,256))
# logits =  Tensor(np_array,mstype.float32)
# labels =  Tensor(np_array,mstype.float32)
# print(logits.shape)
# output = net(logits, labels)
# print(output)
