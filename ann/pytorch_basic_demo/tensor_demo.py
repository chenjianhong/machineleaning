#coding:utf-8
import torch
import plotly.graph_objs as go
import plotly

def main():
    dtype = torch.FloatTensor
    N,d_in,H,d_out = 64,1000,100,10 # d_in表示输入维度,d_out输出维度,H是隐藏层维度数

    x = torch.randn(N,d_in).type(dtype)
    y = torch.randn(N,d_out).type(dtype)

    w1 = torch.randn(d_in,H).type(dtype)
    w2 = torch.randn(H, d_out).type(dtype)

    learning_rate = 1e-6
    for t in range(500):

        # 定义模型
        h = x.mm(w1)
        h_relu = h.clamp(min=0)
        y_pred = h_relu.mm(w2)

        # 定义损失函数
        loss = (y_pred - y).pow(2).sum()

        grad_y_pred = 2.0 * (y_pred - y)
        grad_w2 = h_relu.t().mm(grad_y_pred)
        grad_h_relu = grad_y_pred.mm(w2.t())
        grad_h = grad_h_relu.clone()
        grad_h[h<0] = 0
        grad_w1 = x.t().mm(grad_h)

        w1 -= learning_rate * grad_w1
        w2 -= learning_rate * grad_w2

    print(type(loss))
    print(loss)
    # trace1 = go.Scatter(
    #     x=x[:,0],
    #     y=y[:,0],
    #     mode='markers',
    #     name='markers1'
    # )
    # trace2 = go.Scatter(
    #     x=x[:,0],
    #     y=y_pred[:,0],
    #     mode='markers',
    #     name='markers2'
    # )
    # data = [trace1, trace2]
    #
    # plotly.offline.plot(data, filename='tensor_demo.html')

if __name__=="__main__":
    main()