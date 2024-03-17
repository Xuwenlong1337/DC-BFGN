import torch.nn as nn
import torch.nn.functional as F
from layers_1 import GraphConvolution,GraphAttentionLayer, SpGraphAttentionLayer
import torch
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def BILinear_pooling(adj, XW):
    # step1 sum_squared
    sum = torch.spmm(adj, XW)
    sum_squared = torch.mul(sum, sum)

    # step2 squared_sum
    squared = torch.mul(XW, XW)
    squared_sum = torch.spmm(adj, squared)

    # step3
    new_embedding = 0.5 * (sum_squared - squared_sum)

    return new_embedding

class GCN(nn.Module):
    def __init__(self, nfeat1, nhid, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat1, nhid)


        self.dropout = dropout
        self.linear = nn.Linear(256, 552)
        self.attentions = [GraphAttentionLayer(nfeat1, nhid, dropout=dropout, alpha=0.2, concat=True) for _ in
                           range(1)]

        self.layer5 = nn.Sequential(
            nn.Linear(256, 552),
            nn.ReLU(inplace=True),
            nn.Dropout())
        self.out_att = GraphAttentionLayer(nhid * 1, 400, dropout=dropout, alpha=0.2, concat=False)
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)




    def forward(self, x,adj):
        x_signal = F.relu(self.gc1(x, adj))

        x_signal = F.dropout(x_signal, self.dropout)

        return x_signal
