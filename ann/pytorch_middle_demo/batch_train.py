#coding: utf-8
import torch
import torch.utils.data

torch.manual_seed(1)    # 设置随机数种子,保证每次的网络参数初始化一致

x = torch.linspace(1, 10, 10)       # this is x data (torch tensor)
y = torch.linspace(10, 1, 10)       # this is y data (torch tensor)

torch_dataset = torch.utils.data.TensorDataset(data_tensor=x,target_tensor=y)
loader = torch.utils.data.DataLoader(
    dataset=torch_dataset,
    batch_size=3,
    shuffle=True,
    num_workers=2
)

for e in range(3):
    for step,(batch_x,batch_y) in enumerate(loader):
        print('Epoch: ', e, '| Step: ', step, '| batch x: ',
              batch_x.numpy(), '| batch y: ', batch_y.numpy())