from torch import nn
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 一个均池化和一个瓶颈全连接层
class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # b、c不变，output_size(w, h) = 1
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x,x1):
        b, c, _ = x.size()
        b1, c1, _ = x1.size()# b = batch_size, c = channel

        y = self.avg_pool(x).view(b, c)
        y1 = self.avg_pool(x1).view(b1, c1)
        y = y.to(device)
        y1 = y1.to(device)
        y = self.fc(y).view(b, c, 1)
        y1 = self.fc(y1).view(b1, c1, 1)
        y = y+y1
        y = self.sigmoid(y)
        y = y.to(device)
        x = x.to(device)
        y = x * y.expand_as(x)
        return y

# if __name__ == '__main__':
#
#     model = SELayer(16)
#     a = torch.randn(128,16,552)
#     print(a)
#     output = model(a)
#     print(output)