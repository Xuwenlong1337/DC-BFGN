from Dse_module import *
import torch
import warnings
import numpy as np
class frob(nn.Module):
    def __init__(self, pretrained=False, in_channel=8, out_channel=10):
        super(frob, self).__init__()
        if pretrained == True:
            warnings.warn("Pretrained model is not available")
        self.layer1 = nn.Sequential(
            nn.Linear(200, 1),
            nn.Sigmoid())
        self.layer2 = nn.Sequential(
            nn.Linear(200, 1),
            nn.Sigmoid())



    def forward(self,A_S,A_F,A1,X_sensor,H):
        # 计算差异矩阵
        D_diff1 = A1 - A_S
        D_diff2 = A1 - A_F
        # 计算差异矩阵的 Frobenius 范数
        frobenius_norm1 = torch.norm(D_diff1, p='fro')
        frobenius_norm2 = torch.norm(D_diff2, p='fro')
        a1 = frobenius_norm1/(frobenius_norm1+frobenius_norm2)
        a2 = frobenius_norm2 / (frobenius_norm1 + frobenius_norm2)
        A_S = A_S * torch.eye(8).to(device)
        A_F = A_F * torch.eye(8).to(device)
        if frobenius_norm1>frobenius_norm2:
            aF = self.layer1(H)
            A = aF*(a2*A_S+a1*A_F)
        else:
            aS = self.layer2(X_sensor)
            A = aS*(a2*A_S+a1*A_F)

        return A



