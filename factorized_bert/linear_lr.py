import torch
import torch.nn as nn


def factorize(linear_layer, rank_ratio):
    '''Converts torch.nn.Linear into LinearLR'''
    weight = linear_layer.weight
    rank = min(weight.size()[0], weight.size()[1])
    sliced_rank = int(rank * rank_ratio)
    
    # factorize original weights
    u, s, v = torch.svd(weight)
    u_weight = (u * torch.sqrt(s))[:, 0:sliced_rank]
    v_weight = (torch.sqrt(s) * v)[:, 0:sliced_rank]
    res_weight = weight - u_weight.matmul(v_weight.t())

    # extract arguments
    in_features, out_features = linear_layer.in_features, linear_layer.out_features
    device, dtype = weight.device, weight.dtype
    bias = linear_layer.bias is not None

    lowrank_layer = LinearLR(
        in_features,
        out_features,
        rank_ratio,
        bias,
        device,
        dtype
    )
    
    # initialize lowrank layer weights with factorized weights
    with torch.no_grad():
        lowrank_layer.u.weight.copy_(u_weight)
        lowrank_layer.v.weight.copy_(v_weight.t())
        lowrank_layer.res.weight.copy_(res_weight)
        
        if bias:
            lowrank_layer.u.bias.copy_(linear_layer.bias)

    return lowrank_layer


class LinearLR(nn.Module):
    '''[u * v + res] version of torch.nn.Linear'''
    def __init__(self, in_features, out_features, rank_ratio=0.25, bias=True, device=None, dtype=None):
        super().__init__()

        sliced_rank = int(min(in_features, out_features) * rank_ratio)
        self.u = nn.Linear(
            sliced_rank,
            out_features,
            bias=bias,  # original bias is stored as the bias of u
            device=device,
            dtype=dtype
        )
        self.v = nn.Linear(
            in_features,
            sliced_rank,
            bias=False,
            device=device,
            dtype=dtype
        )
        self.res = nn.Linear(
            in_features,
            out_features,
            bias=False,
            device=device,
            dtype=dtype
        )
        
    def freeze(self):
        for param in self.res.parameters():
            param.requires_grad = False
            
    def unfreeze(self):
        for param in self.res.parameters():
            param.requires_grad = True

    def forward(self, input):
        return self.u(self.v(input)) + self.res(input)