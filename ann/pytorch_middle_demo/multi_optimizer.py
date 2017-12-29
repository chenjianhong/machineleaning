#coding: utf-8
import torch
from torch.autograd import Variable
import torch.utils.data as Data
import matplotlib.pyplot as plt
import torch.nn.functional as F

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(1, 20)   # hidden layer
        self.predict = torch.nn.Linear(20, 1)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x


def main():
    LR = 1e-2

    BATCH_SIZE=4

    x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
    y = x.pow(2) + 0.2 * torch.rand(x.size())


    torch_dataset = Data.TensorDataset(data_tensor=x, target_tensor=y)
    loader = Data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, )

    net_SGD = Net()
    net_Momentum = Net()
    net_RMSprop = Net()
    net_Adam = Net()
    nets = [net_SGD, net_Momentum, net_RMSprop, net_Adam]

    loss_func = torch.nn.MSELoss()

    optimizer_list = [
        torch.optim.SGD(net_SGD.parameters(),lr=LR),
        torch.optim.SGD(net_Momentum.parameters(),lr=LR, momentum=0.8),
        torch.optim.RMSprop(net_RMSprop.parameters(),lr=LR, alpha=0.9),
        torch.optim.Adam(net_Adam.parameters(),lr=LR, betas=(0.9, 0.99)),
    ]

    losses_his = [[], [], [], []]  # record loss

    for i in range(BATCH_SIZE):
        for step, (batch_x, batch_y) in enumerate(loader):
            b_x = Variable(x)
            b_y = Variable(y)
            for net, opt, l_his in zip(nets, optimizer_list, losses_his):
                output = net(b_x)  # get output for every net
                loss = loss_func(output, b_y)  # compute loss for every net
                opt.zero_grad()  # clear gradients for next train
                loss.backward()  # backpropagation, compute gradients
                opt.step()  # apply gradients
                l_his.append(loss.data[0])  # loss recoder

    labels = ['SGD', 'Momentum', 'RMSprop', 'Adam']
    for i, l_his in enumerate(losses_his):
        plt.plot(l_his, label=labels[i])
    plt.legend(loc='best')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.ylim((0, 0.2))
    plt.show()

if __name__ == '__main__':
    main()