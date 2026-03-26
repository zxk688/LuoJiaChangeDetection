import torch
import torch.nn as nn


class FeatureInteractionNeck(nn.Module):
    """Feature Interaction Neck.

    Args:
        policy (str): The operation to fuse features. candidates 
            are `concat`, `sum`, `diff` and `Lp_distance`.
        in_channels (Sequence(int)): Input channels.
        channels (int): Channels after modules, before conv_seg.
        out_indices (tuple[int]): Output from which layer.
    """

    def __init__(self,
                 policy,
                 in_channels=None,
                 channels=None,
                 out_indices=[0,1,2,3,4]):
        super().__init__()
        self.policy = policy
        # self.in_channels = in_channels
        # self.channels = channels
        # out_indices = out_indices
        # lis  = [eval(i) for i in out_indices]
        # self.out_indices = lis
        self.out_indices = out_indices

    @staticmethod
    def fusion(x1, x2, policy):
        """Specify the form of feature fusion"""
        
        _fusion_policies = ['concat', 'sum', 'diff', 'abs_diff']
        assert policy in _fusion_policies, 'The fusion policies {} are ' \
            'supported'.format(_fusion_policies)
        
        if policy == 'concat':
            x = torch.cat([x1, x2], dim=1)
        elif policy == 'sum':
            x = x1 + x2
        elif policy == 'diff':
            x = x2 - x1
        elif policy == 'abs_diff':
            x = torch.abs(x1 - x2)

        return x

    def forward(self, inputs):
        
        """Forward function."""
        x1, x2 = inputs
        assert len(x1) == len(x2), "The features x1 and x2 from the" \
            "backbone should be of equal length"
        outs = []
        for i in range(len(x1)):
            out = self.fusion(x1[i], x2[i], self.policy)
            outs.append(out)
        
        outs = [outs[i] for i in self.out_indices]
        return tuple(outs)
    
# if __name__ =="__main__":
#     model=FeatureInterationNeck(policy='sum',out_indices=(0,))
#     a=torch.rand(2,18,256,256)
#     aa=(a,a,a,a,)
#     a1=torch.rand(2,18,256,256)
#     bb=(a1,a1,a1,a1,)
    
#     b=model(aa,bb)
#     print(len(b))
#     print(b[0].shape)



