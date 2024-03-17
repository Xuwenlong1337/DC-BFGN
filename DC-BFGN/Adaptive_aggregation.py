#!/usr/bin/python
# -*- coding:utf-8 -*-
from torch import nn
import warnings
import torch
from Signal_GCN import GCN
from frobenius import frob

class Adaptive(nn.Module):
    def __init__(self, pretrained=False, in_channel=8, out_channel=10):
        super(Adaptive, self).__init__()
        if pretrained == True:
            warnings.warn("Pretrained model is not available")

        self.layer1 = nn.Sequential(
            nn.Linear(200, 1),
            nn.Sigmoid())
        self.layer2 = nn.Sequential(
            nn.Linear(200, 1),
            nn.Sigmoid())
        self.layer3 = nn.Sequential(
            nn.Linear(200, 1),
            nn.Sigmoid())

        self.model_GCN1 = GCN(nfeat1=200,  # 5*5adj1F1.shape[1]
                              nhid=100,
                              dropout=0.5)
        self.model_GCN2 = GCN(nfeat1=200,  # 5*5adj1F1.shape[1]
                              nhid=100,
                              dropout=0.5)
        self.model_GCN3 = GCN(nfeat1=200,  # 5*5adj1F1.shape[1]
                              nhid=100,
                              dropout=0.5)
        self.frobenius = frob(pretrained)

    def forward(self,H1,H2,H3,A_S,A_F,x1,x2,x3,x_sensor):
        A_S = A_S.to_dense()

        a1 = self.layer1(H1)
        A1 = a1 * A_S +(1-a1) * A_F
        A1_1 = self.frobenius(A_S,A_F,A1,x_sensor,x1)
        A1 = A1 + A1_1
        H1 = self.model_GCN1(H1,A1)


        a2 = self.layer2(H2)
        A2 = a2 * A_S + (1 - a2) * A_F
        A2_1 = self.frobenius(A_S, A_F, A2, x_sensor,x2)
        A2 = A2 + A2_1
        H2 = self.model_GCN2(H2, A2)

        a3 = self.layer3(H3)
        A3 = a3 * A_S + (1 - a3) * A_F
        A3_1 = self.frobenius(A_S, A_F, A3,x_sensor,x3)
        A3 = A3 + A3_1
        H3 = self.model_GCN3(H3, A3)
        H = torch.cat([H1,H2,H3],dim=1)

        return H

