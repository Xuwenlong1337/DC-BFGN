#!/usr/bin/python
# -*- coding:utf-8 -*-
from torch import nn
import warnings
import torch
from torch_geometric.nn import  ChebConv, BatchNorm
from torch_geometric.utils import dropout_adj
from torch.nn.functional import cosine_similarity


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class GGL(torch.nn.Module):
    '''
    Grapg generation layer
    '''

    def __init__(self,):
        super(GGL, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(200,10),
            nn.Sigmoid())

    def forward(self, x):
        x = x.view(x.size(0), -1)
        atrr = self.layer(x)
        values, edge_index ,A_norm= Gen_edge(atrr)
        return values.view(-1), edge_index, A_norm

def Gen_edge(atrr):#attr：torch.size=[64,10]
    atrr = atrr.cpu()
    adjacency_matrix = torch.zeros(8, 8)  # 创建一个初始的8x8邻接矩阵
    for c in range(8):
        for v in range(8):
            similarity = cosine_similarity(atrr[c].unsqueeze(0), atrr[v].unsqueeze(0))
            adjacency_matrix[c, v] = similarity
    A = adjacency_matrix
    # A = torch.mm(atrr, atrr.T)
    maxval, maxind = A.max(axis=1)
    A_norm = A / maxval
    k = A.shape[0]#shape[0]就是读取矩阵第一维度的长度
    values, indices = A_norm.topk(k, dim=1, largest=True, sorted=False) #values：最大的k个值;indices：最大值所对应的下标
    edge_index = torch.tensor([[],[]],dtype=torch.long)

    for i in range(indices.shape[0]):
        index_1 = torch.zeros(indices.shape[1],dtype=torch.long) + i
        index_2 = indices[i]
        sub_index = torch.stack([index_1,index_2])
        edge_index = torch.cat([edge_index,sub_index],axis=1)

    return values, edge_index, A_norm

class MultiChev_1(torch.nn.Module):
    def __init__(self, in_channels,):
        super(MultiChev_1, self).__init__()
        self.scale_1 = ChebConv(in_channels, 200, K=1)
        self.scale_2 = ChebConv(in_channels, 200, K=2)
        self.scale_3 = ChebConv(in_channels, 200, K=3)

    def forward(self, x, edge_index,edge_weight ):
        scale_1 = self.scale_1(x, edge_index,edge_weight )
        scale_2 = self.scale_2(x, edge_index,edge_weight )
        scale_3 = self.scale_3(x, edge_index,edge_weight )
        return scale_1,scale_2,scale_3

class MultiChev_B(torch.nn.Module):
    def __init__(self, in_channels,):
        super(MultiChev_B, self).__init__()
        self.scale_1 = ChebConv(in_channels,100,K=2)
        self.scale_2 = ChebConv(in_channels,100,K=3)
        self.scale_3 = ChebConv(in_channels,100,K=4)
    def forward(self, x, edge_index,edge_weight ):
        scale_1 = self.scale_1(x, edge_index,edge_weight )
        scale_2 = self.scale_2(x, edge_index,edge_weight )
        scale_3 = self.scale_3(x, edge_index,edge_weight )
        return torch.cat([scale_1,scale_2,scale_3],1)



class MGCN(nn.Module):
    def __init__(self, pretrained=False, in_channel= 200, out_channel=10):
        super(MGCN, self).__init__()
        if pretrained == True:
            warnings.warn("Pretrained model is not available")

        self.atrr = GGL()
        self.conv1 = MultiChev_1(in_channel)
        self.bn1 = BatchNorm(200)
        self.conv2 = MultiChev_B(200)
        self.bn2 = BatchNorm(300)
        self.layer5 = nn.Sequential(
            nn.Linear(300, 200),
            nn.ReLU(inplace=True),
            nn.Dropout())


    def forward(self, x):
        edge_atrr, edge_index, A_norm = self.atrr(x)
        edge_atrr = edge_atrr.to(device)
        edge_index = edge_index.to(device)
        edge_index, edge_atrr = dropout_adj(edge_index,edge_atrr)
        x1,x2,x3 = self.conv1(x, edge_index, edge_weight =  edge_atrr)
        x1 = self.bn1(x1)
        x2 = self.bn1(x2)
        x3 = self.bn1(x3)
        return x1,x2,x3,A_norm