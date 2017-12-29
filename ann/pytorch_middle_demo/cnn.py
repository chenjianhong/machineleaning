#coding: utf-8
import torch
import os
import torchvision
import matplotlib.pyplot as plt
import torch.utils.data as Data
from torch.autograd import Variable

class CNN(torch.nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Sequential(   # 输入尺寸 (1,28,28)
            torch.nn.Conv2d(
                in_channels=1,
                out_channels=16, # 几个卷积核
                kernel_size=5,
                stride=1,
                padding=2,       # 为了保证图像输入和输出的尺寸一致,需要左右填充0,padding=(kernel_size-1)/2,stride为1的情况下
            ),                   # 输出尺寸 (16,28,28)
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2), # 最大值池化,输出尺寸 (16,14,14)
        )

        self.conv2 = torch.nn.Sequential( # 输入尺寸 (16,14,14)
            torch.nn.Conv2d(16,32,5,1,2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2) # 输出尺寸 (32,7,7)
        )
        self.out = torch.nn.Linear(32*7*7,10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0),-1)
        output = self.out(x)
        return output,x


def main():
    EPOCH = 1  # train the training data n times, to save time, we just train 1 epoch
    BATCH_SIZE = 50
    DOWNLOAD_MNIST = False

    if not (os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
        DOWNLOAD_MNIST = True

    train_data = torchvision.datasets.MNIST(
        root='../data_set/mnist/',
        train=True,  # this is training data
        transform=torchvision.transforms.ToTensor(),  # 把图片的每个像素值归一化到(0,1)
        download=DOWNLOAD_MNIST,
    )
    test_data = torchvision.datasets.MNIST(root='../data_set/mnist/', train=False)

    plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
    plt.title('%i' % train_data.train_labels[0])
    plt.show()

    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

    test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1), volatile=True).type(torch.FloatTensor)[:2000] / 255.  # volatile 禁用自动梯度,变量的后续计算都不会有梯度
    test_y = test_data.test_labels[:2000]

    cnn = CNN()

    loss_func = torch.nn.CrossEntropyLoss()

    optm = torch.optim.Adam(cnn.parameters())

    for epoch in range(EPOCH):
        for step,(x,y) in enumerate(train_loader):
            b_x = Variable(x)
            b_y = Variable(y)

            o = cnn(b_x)[0]
            loss = loss_func(o,b_y)
            optm.zero_grad()
            loss.backward()
            optm.step()

            if step % 50 == 0:
                test_output, last_layer = cnn(test_x)
                pred_y = torch.max(test_output, 1)[1].data.squeeze()
                accuracy = sum(pred_y == test_y) / float(test_y.size(0))
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0], '| test accuracy: %.2f' % accuracy)



if __name__ == '__main__':
    main()

