#!/usr/bin/python
# -*- coding:utf-8 -*-
from torch import nn
import warnings
import torch
from Signal_GCN import GCN
from MRF_GCN import MRF_GCN


import torch
import torch.nn as nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def hid(ht1,ht2):
    hidden_size = 50
    forget_gate_weights = nn.Parameter(torch.randn(hidden_size * 2, hidden_size)).to(torch.float64).to(device)
    forget_gate_bias = nn.Parameter(torch.randn(hidden_size)).to(torch.float64).to(device)
    concatenated_input = torch.cat((ht1, ht2), dim=1)
    linear_transform = torch.matmul(concatenated_input, forget_gate_weights) + forget_gate_bias
    a = torch.sigmoid(linear_transform)
    H = a*ht1

    return H


# 定义输入数据


# 定义遗忘门的权重和偏置


# 计算遗忘门的输出


class LGF(nn.Module):
    def __init__(self, pretrained=False, in_channel=8, out_channel=10):
        super(LGF, self).__init__()
        if pretrained == True:
            warnings.warn("Pretrained model is not available")

        self.layer1 = nn.Sequential(
            nn.Linear(400, 300)
            )

        self.model_GCN1 = GCN(nfeat1=50,  # 5*5adj1F1.shape[1]
                             nhid=50,
                             dropout=0.5)
        self.model_GCN1 = GCN(nfeat1=50,  # 5*5adj1F1.shape[1]
                              nhid=50,
                              dropout=0.5)
        self.model_GCN2 = GCN(nfeat1=50,  # 5*5adj1F1.shape[1]
                              nhid=50,
                              dropout=0.5)
        self.model_GCN3 = GCN(nfeat1=50,  # 5*5adj1F1.shape[1]
                              nhid=200,
                              dropout=0.5)


    def forward(self,x,AS,AF):
        x_t1 = x[:,:50]
        x_t2 = x[:, 50:100]
        x_t3 = x[:, 100:150]
        x_t4 = x[:, 150:200]

        t1_XF1 = x_t1 + self.model_GCN1(x_t1, AF)
        t1_XS1 = self.model_GCN1(x_t1, AS)
        t1_XF2 = 0.2*t1_XS1+0.8*self.model_GCN1(t1_XF1, AF)


        t2_XF1 = x_t2 + self.model_GCN1(x_t2, AF)
        t2_XS1 = self.model_GCN1(x_t2, AS)
        hS_t1 = hid(t1_XS1,t2_XS1 )
        t2_XS1 = t2_XS1+hS_t1
        t2_XF2 = 0.2*t2_XS1 + 0.8*self.model_GCN1(t2_XF1, AF)
        hF_t1 = hid(t1_XF2,t2_XF2 )
        t2_XF2 = hF_t1+ t2_XF2

        t3_XF1 = 0.2*x_t3 + 0.8*self.model_GCN1(x_t3, AF)
        t3_XS1 = self.model_GCN1(x_t3, AS)
        hS_t2 = hid(t2_XS1, t3_XS1)
        t3_XS1 = t3_XS1 + hS_t2
        t3_XF2 = 0.2*t3_XS1 + 0.8*self.model_GCN1(t3_XF1, AF)
        hF_t2 = hid(t2_XF2, t3_XF2)
        t3_XF2 = hF_t2 + t3_XF2


        t4_XF1 = x_t4 + self.model_GCN1(x_t4, AF)
        t4_XS1 = self.model_GCN1(x_t4, AS)
        hS_t3 = hid(t3_XS1, t4_XS1)
        t4_XS1 = t4_XS1 + hS_t3
        t4_XF2 = 0.2*t4_XS1 + 0.8*self.model_GCN1(t4_XF1, AF)
        hF_t3 = hid(t3_XF2, t4_XF2)
        t4_XF2 = hF_t3 + t4_XF2

        t1_XS1 = x_t1 + self.model_GCN1(x_t1, AS)
        t1_XF1 = self.model_GCN1(x_t1, AF)
        t1_XS2 = 0.2 * t1_XF1 + 0.8 * self.model_GCN1(t1_XS1, AS)

        t2_XS1 = x_t2 + self.model_GCN1(x_t2, AS)
        t2_XF1 = self.model_GCN1(x_t2, AF)
        hF_t1 = hid(t1_XF1, t2_XF1)
        t2_XF1 = t2_XF1 + hF_t1
        t2_XS2 = 0.2 * t2_XF1 + 0.8 * self.model_GCN1(t2_XS1, AS)
        hS_t1 = hid(t1_XS2, t2_XS2)
        t2_XS2 = hS_t1 + t2_XS2

        t3_XS1 = 0.2 * x_t3 + 0.8 * self.model_GCN1(x_t3, AS)
        t3_XF1 = self.model_GCN1(x_t3, AF)
        hF_t2 = hid(t2_XF1, t3_XF1)
        t3_XF1 = t3_XF1 + hF_t2
        t3_XS2 = 0.2 * t3_XF1 + 0.8 * self.model_GCN1(t3_XS1, AS)
        hS_t2 = hid(t2_XS2, t3_XS2)
        t3_XS2 = hS_t2 + t3_XS2

        t4_XS1 = x_t4 + self.model_GCN1(x_t4, AS)
        t4_XF1 = self.model_GCN1(x_t4, AF)
        hF_t3 = hid(t3_XF1, t4_XF1)
        t4_XF1 = t4_XF1 + hF_t3
        t4_XS2 = 0.2 * t4_XF1 + 0.8 * self.model_GCN1(t4_XS1, AS)
        hS_t3 = hid(t3_XS2, t4_XS2)
        t4_XS2 = hS_t3 + t4_XS2


        h1 = torch.cat([t1_XF2, t1_XS2], dim=1)
        h2 = torch.cat([t2_XF2, t2_XS2], dim=1)
        h3 = torch.cat([t3_XF2, t3_XS2], dim=1)
        h4 = torch.cat([t4_XF2, t4_XS2], dim=1)
        h_L = torch.cat([h1,h2,h3,h4],dim =1)
        h_L = self.layer1(h_L)

        return h_L

