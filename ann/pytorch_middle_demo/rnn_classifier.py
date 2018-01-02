#coding: utf-8
import os
import torch
import torch.utils.data
import torchvision.datasets as dsets
from torch.autograd import Variable
import torchvision.transforms as transforms

EPOCH = 1               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 64
TIME_STEP = 28          # rnn time step / image height
INPUT_SIZE = 28         # rnn input size / image width
LR = 0.01               # learning rate
DOWNLOAD_MNIST = True   # set to True if haven't download the data

if not (os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
    DOWNLOAD_MNIST = True

class RNN(torch.nn.Module):

    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = torch.nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )

        self.out = torch.nn.Linear(64,10)

    def forward(self, x):
        r_out, (h_n,h_c) = self.rnn(x,None)

        out = self.out(r_out[:,-1,:])
        return out


def main():
    train_data = dsets.MNIST(
        root='../data_set/mnist/',
        train=True,  # this is training data
        transform=transforms.ToTensor(),  # Converts a PIL.Image or numpy.ndarray to
        # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
        download=DOWNLOAD_MNIST,  # download it if you don't have it
    )

    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

    # convert test data into Variable, pick 2000 samples to speed up testing
    test_data = dsets.MNIST(root='../data_set/mnist/', train=False, transform=transforms.ToTensor())
    test_x = Variable(test_data.test_data, volatile=True).type(torch.FloatTensor)[
             :2000] / 255.  # shape (2000, 28, 28) value in range(0,1)
    test_y = test_data.test_labels.numpy().squeeze()[:2000]  # covert to numpy array

    rnn = RNN()

    optm = torch.optim.Adam(rnn.parameters())

    loss_func = torch.nn.CrossEntropyLoss()

    for epoch in range(EPOCH):
        for step,(x,y) in enumerate(train_loader):
            b_x = Variable(x.view(-1,28,28))
            b_y = Variable(y)

            o = rnn(b_x)

            loss = loss_func(o,b_y)

            optm.zero_grad()
            loss.backward()
            optm.step()
            if step % 50 == 0:
                test_output = rnn(test_x)                   # (samples, time_step, input_size)
                pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
                accuracy = sum(pred_y == test_y) / float(test_y.size)
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0], '| test accuracy: %.2f' % accuracy)






if __name__ == '__main__':
    main()