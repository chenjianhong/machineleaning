#coding:utf-8
import torch
from torch.autograd import Variable

class MyRelu(torch.autograd.Function):
    '''
    自定义relu函数
    '''

    def forward(self, input):
        self.save_for_backward(input) # 缓存起来backward阶段使用
        return input.clamp(min=0)

    def backward(self, grad_output):
        input = self.saved_tensors[0]
        grad_input = grad_output.clone()
        grad_input[input<0] = 0
        return grad_input

def main():
    dtype = torch.FloatTensor
    N, d_in, H, d_out = 64, 1000, 100, 10  # d_in表示输入维度,d_out输出维度,H是隐藏层维度数

    x = Variable(torch.randn(N, d_in).type(dtype), requires_grad=False)
    y = Variable(torch.randn(N, d_out).type(dtype), requires_grad=False)

    w1 = Variable(torch.randn(d_in, H).type(dtype), requires_grad=True)
    w2 = Variable(torch.randn(H, d_out).type(dtype), requires_grad=True)

    learning_rate = 1e-6
    for t in range(500):

        relu = MyRelu()

        y_pred = relu(x.mm(w1)).mm(w2)

        loss = (y_pred - y).pow(2).sum()

        loss.backward()

        w1.data -= learning_rate * w1.grad.data
        w2.data -= learning_rate * w2.grad.data

        w1.grad.data.zero_()
        w2.grad.data.zero_()
    print(loss.data[0])

if __name__=="__main__":
    main()