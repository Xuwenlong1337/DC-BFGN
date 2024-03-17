from torch import nn
import warnings
import torch
import torch.nn.functional as F
from MGCN import MGCN
from CNN import CNN
from Signal_GCN import GCN
import numpy as np
import scipy.sparse as sp
from Adaptive_aggregation import Adaptive
from LGFM import LGF
from se_module import *
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
#
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

adj2 =    sp.coo_matrix([[0,1,1,1,0,0,0,0],
                         [1,0,0,0,1,0,0,0],
                         [1,0,0,0,0,1,0,0],
                         [1,0,0,0,0,0,1,1],
                         [0,1,0,0,0,0,0,0],
                         [0,0,1,0,0,0,0,0],
                         [0,0,0,1,0,0,0,0],
                         [0,0,0,1,0,0,0,0]],shape=(8,8),dtype=np.float32)
adj_I2 = normalize(adj2 + sp.eye(adj2.shape[0]))
adj2 = sparse_mx_to_torch_sparse_tensor(normalize(adj2))
adj_I2 = sparse_mx_to_torch_sparse_tensor(adj_I2)
adj_I2= adj_I2.to(torch.float64).to(device)
adj2= adj2.to(torch.float64).to(device)

class DC(nn.Module):
    def __init__(self, pretrained=False, in_channel=1, out_channel=10):
        super(DC, self).__init__()
        self.model_FGCN = MGCN(pretrained)
        self.model_cnn = CNN(pretrained)
        self.model_Adaptive = Adaptive(pretrained)
        self.linear = nn.Linear(4800, 1080)
        self.LGF = LGF(pretrained)
        self.model_SGCN = GCN(nfeat1=200,
            nhid=200,
            dropout=0.2)


    def forward(self, x):
        x = self.model_cnn(x)
        k = x.shape[0]
        G_output1 = torch.zeros(64, 8, 300)
        G_output2 = torch.zeros(64, 8, 300)
        for i in range(k):
            F2 = x[i, :, :]
            F2 = F2.to(torch.float64).to(device)
            x1, x2, x3, AF = self.model_FGCN(F2)  # 得到8，200
            x_sensor = self.model_SGCN(F2, adj_I2)
            H1 = (x1 + x_sensor) / 2
            H2 = (x2 + x_sensor) / 2
            H3 = (x3 + x_sensor) / 2
            AF = AF.to(torch.float64).to(device)
            H_h = self.model_Adaptive(H1, H2, H3, adj_I2, AF, x1, x2, x3, x_sensor)
            H_L = self.LGF(F2, adj_I2, AF)
            G_output1[i, :, :] = H_h
            G_output2[i, :, :] = H_L
            G_output1 = G_output1.to(torch.float64).to(device)
            G_output2 = G_output2.to(torch.float64).to(device)

        SE1 = SELayer(8).to(device)
        # no注意力
        G_output1h = SE1(G_output1.to(torch.float32), G_output2.to(torch.float32))
        G_output2L = G_output2.permute(0, 2, 1).to(torch.float64)
        G_output2h = G_output1.permute(0, 2, 1).to(torch.float64)

        SE2 = SELayer(300).to(device)
        G_output2 = SE2(G_output2L.to(torch.float32), G_output2h.to(torch.float32))
        G_output2V = G_output2.permute(0, 2, 1).to(torch.float64)
        G_output = torch.cat([G_output1h, G_output2V], dim=2)
        output = G_output.contiguous().view(G_output.size(0), -1)
        output = self.linear(output)
        output = F.log_softmax(output, dim=1)

        return output

