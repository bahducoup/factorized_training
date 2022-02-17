import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.cuda.amp import autocast
import logging

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

__author__ = "Yu-Hsiang Huang"

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    @autocast()
    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            #attn = attn.masked_fill(mask == 0, -1e9)
            attn = attn.masked_fill(mask == 0, -(2 ** 15))

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn
