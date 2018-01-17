#coding: utf-8
import torch
import torch.utils.data
import torchvision
import argparse
import os
import time
import shutil
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('-b', '--batch_size', help='batch_size', default=120, type=int)
parser.add_argument('-w', '--workers', default=8, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('-e', '--epoch', default=90, type=int)
parser.add_argument('--print-freq', '-p', default=100, type=int, help='print frequency (default: 100)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--fine_tune_fc', default=False, type=bool, help='fine_tune_fc')


args = parser.parse_args()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True) # pred是top k的索引值
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred)) # target每个样本只有一个值,表示具体类别值,expand之后比较是否相等,相等的就是对的

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)  # top几的分类正确数量累加,然后除以batch_size就是准确率
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def save_checkpoint(state, is_best, filename='fineturn_epoch.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'fineturn_model_best.pth.tar')

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    prev_end_time = time.time()

    model.train()

    for i, (x, y) in enumerate(train_loader):
        y = y.cuda()
        b_x = torch.autograd.Variable(x).cuda()
        b_y = torch.autograd.Variable(y).cuda()

        o = model(b_x)

        # for nets that have multiple outputs such as inception
        if isinstance(o, tuple):
            loss = sum((criterion(i, b_y) for i in o))
        else:
            loss = criterion(o, b_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        prec1, prec5 = accuracy(sum(o).data, y, topk=(1, 2))
        batch_time.update(time.time() - prev_end_time)
        data_time.update(time.time() - prev_end_time)
        losses.update(loss.data[0], b_x.size(0))
        top1.update(prec1[0], b_x.size(0))
        top5.update(prec5[0], b_x.size(0))

        prev_end_time = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))




def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True).cuda()
        target_var = torch.autograd.Variable(target, volatile=True).cuda()

        # compute output
        o = model(input_var)
        if isinstance(o, tuple):
            loss = sum((criterion(i, target_var) for i in o))
        else:
            loss = criterion(o, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(sum(o).data, target, topk=(1, 2))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg


def run():
    # model = torchvision.models.inception_v3(pretrained=False)


    train_dir = os.path.join(args.data,'train')
    val_dir = os.path.join(args.data,'val')

    # normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    normalize = torchvision.transforms.Normalize(mean=[0.38924074,0.37072024,0.36058131],std=[0.37184399,0.36313567,0.36288691])

    train_dataset = torchvision.datasets.ImageFolder(
        train_dir,
        torchvision.transforms.Compose([
            torchvision.transforms.RandomSizedCrop(299),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            normalize,
        ])
    )

    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,num_workers=args.workers,pin_memory=False)
    val_loader = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        val_dir,
        torchvision.transforms.Compose([
            torchvision.transforms.Scale(320),
            torchvision.transforms.CenterCrop(299),
            torchvision.transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size,shuffle=True,num_workers=args.workers,pin_memory=False
    )

    # model = torchvision.models.resnet18(pretrained=False,num_classes=2)
    model = torchvision.models.inception_v3(pretrained=True)
    labels = len(train_dataset.classes)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, labels)

    if args.fine_tune_fc:
        for param in model.parameters():
            param.requires_grad = False
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, labels)

    model = torch.nn.DataParallel(model).cuda()
    loss_func = torch.nn.CrossEntropyLoss().cuda()
    optm = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

    best_prec1 = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (best_prec1 {})"
                  .format(args.resume, checkpoint['best_prec1']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    for epoch in range(args.epoch):
        train(train_loader, model, loss_func, optm, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, loss_func)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1

        best_prec1 = max(prec1, best_prec1)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optm.state_dict(),
        }, is_best,filename='fineturn_epoch_%s.pth.tar'%epoch)


def calculate_image_mean_and_std():
    train_dir = os.path.join(args.data, 'train')
    val_dir = os.path.join(args.data, 'val')

    train_dataset = torchvision.datasets.ImageFolder(
        train_dir,
        torchvision.transforms.Compose([
            torchvision.transforms.RandomSizedCrop(299),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
        ])
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=5000, shuffle=True,
                                               num_workers=5, pin_memory=True)

    for i, (x, y) in enumerate(train_loader):
        np_x = x.numpy()
        pop_channel_mean = np.mean(np_x, axis=(0, 2, 3))  # Shape (3,)
        pop_channel_std = np.std(np_x, axis=(0, 2, 3))  # Shape (3,)

        print(pop_channel_mean)
        print(pop_channel_std)


if __name__ == '__main__':
    run()