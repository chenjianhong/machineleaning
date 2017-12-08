#coding:utf-8
import torch
from torch.autograd import Variable

class TwoLayerNet(torch.nn.Module):

    def __init__(self,d_in,h,d_out):
        super(TwoLayerNet,self).__init__()
        self.linear1 = torch.nn.Linear(d_in,h)
        self.linear2 = torch.nn.Linear(h,d_out)

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred

def main():
    dtype = torch.FloatTensor
    N, d_in, H, d_out = 64, 1000, 100, 10  # d_in表示输入维度,d_out输出维度,H是隐藏层维度数

    x = Variable(torch.randn(N, d_in).type(dtype), requires_grad=False)
    y = Variable(torch.randn(N, d_out).type(dtype), requires_grad=False)

    model = TwoLayerNet(d_in,H,d_out)

    loss_fn = torch.nn.MSELoss(size_average=False)

    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for t in range(500):
        y_pred = model(x)

        loss = loss_fn(y_pred,y)

        model.zero_grad()
        loss.backward()

        optimizer.step()
    print(loss.data[0])


if __name__=="__main__":
    main()