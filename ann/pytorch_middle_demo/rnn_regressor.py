#coding: utf-8
import torch
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt

TIME_STEP = 10      # rnn time step
INPUT_SIZE = 1      # rnn input size

class RNN(torch.nn.Module):

    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = torch.nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=32,
            num_layers=2,
            batch_first=True
        )

        self.out = torch.nn.Linear(32,1)

    def forward(self, x, h_state):
        r_out,h_state = self.rnn(x,h_state)

        outs = []
        for t in range(r_out.size(1)):
            outs.append(self.out(r_out[:,t,:]))
        return torch.stack(outs,dim=1),h_state


def main():
    rnn = RNN()
    print(rnn)

    optm = torch.optim.Adam(rnn.parameters())

    loss_func = torch.nn.MSELoss()

    h_state = None

    plt.ion()

    for step in range(100):
        start, end = step * np.pi, (step + 1) * np.pi  # time range
        # use sin predicts cos
        steps = np.linspace(start, end, TIME_STEP, dtype=np.float32)
        x_np = np.sin(steps)  # float32 for converting torch FloatTensor
        y_np = np.cos(steps)

        x = Variable(torch.from_numpy(x_np[np.newaxis, :, np.newaxis]))  # shape (batch, time_step, input_size)
        y = Variable(torch.from_numpy(y_np[np.newaxis, :, np.newaxis]))

        o,h_s = rnn(x,h_state)

        h_state = Variable(h_s.data)


        loss = loss_func(o,y)

        optm.zero_grad()
        loss.backward()
        optm.step()


        plt.plot(steps, y_np.flatten(), 'r-')
        plt.plot(steps, o.data.numpy().flatten(), 'b-')
        plt.draw()
        plt.pause(0.05)

    plt.ioff()
    plt.show()



if __name__ == '__main__':
    main()

