import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from utils.dataloader import voice_dataset_collate, VoiceDataset
from nets.mobilenetv3 import mobilenetv3_large
from torch.utils.data import DataLoader
import argparse
import time
import os

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def get_args():
    parser = argparse.ArgumentParser("parameters")

    parser.add_argument("--dataset-mode", type=str, default="VOICE",
                        help="(example: CIFAR10, CIFAR100, IMAGENET, VOICE), (default: IMAGENET)")
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs, (default: 100)")
    parser.add_argument("--batch-size", type=int, default=32, help="number of batch size, (default, 512)")
    parser.add_argument("--learning-rate", type=float, default=1e-2, help="learning_rate, (default: 1e-1)")
    parser.add_argument("--dropout", type=float, default=1, help="dropout rate, not implemented yet, (default: 0.8)")
    parser.add_argument('--model-mode', type=str, default="SMALL", help="(example: LARGE, SMALL), (default: LARGE)")
    parser.add_argument("--load-pretrained", type=bool, default=False, help="(default: False)")
    parser.add_argument('--evaluate', type=bool, default=False, help="Testing time: True, (default: False)")
    parser.add_argument('--multiplier', type=float, default=1.0, help="(default: 1.0)")
    parser.add_argument('--print-interval', type=int, default=5,
                        help="training information and evaluation information output frequency, (default: 5)")
    parser.add_argument('--data', default='D:/ILSVRC/Data/CLS-LOC')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--distributed', type=bool, default=False)

    args = parser.parse_args()

    return args


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.learning_rate * (0.1 ** (epoch // 30))
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(train_loader, model, criterion, optimizer, epoch):
    # switch to train mode
    model.train()
    acc = []
    total_loss = 0
    for i, (data, target) in enumerate(train_loader):
        # measure data loading time
        with torch.no_grad():
            if torch.cuda.is_available():
                data = Variable(torch.from_numpy(data).type(torch.FloatTensor)).cuda()
                target = Variable(torch.from_numpy(target).type(torch.FloatTensor)).cuda()
            else:
                data = Variable(torch.from_numpy(data).type(torch.FloatTensor))
                target = Variable(torch.from_numpy(target).type(torch.FloatTensor))

        # compute output
        output = model(data)
        loss = criterion(output, target.long())
        # loss = sum(loss)
        total_loss += loss.detach().item()
        # measure accuracy and record loss
        acc.append(accuracy(output, target.long()))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("epoch{} acc：{} loss: {}".format(epoch, np.mean(np.array(acc)), total_loss))


def validate(val_loader, model, criterion):
    # switch to evaluate mode
    model.eval()
    valid_total_loss = 0
    acc = []
    with torch.no_grad():
        for i, (data, target) in enumerate(val_loader):
            with torch.no_grad():
                if torch.cuda.is_available():
                    data = Variable(torch.from_numpy(data).type(torch.FloatTensor)).cuda()
                    target = Variable(torch.from_numpy(target).type(torch.FloatTensor)).cuda()
                else:
                    data = Variable(torch.from_numpy(data).type(torch.FloatTensor))
                    target = Variable(torch.from_numpy(target).type(torch.FloatTensor))

                # compute output
                output = model(data)
                loss = criterion(output, target.long())
                valid_total_loss += loss.detach().item()
                # measure accuracy and record loss
                acc.append(accuracy(output, target.long()))
        print("val loss is ", valid_total_loss)

    return np.mean(np.array(acc))


def accuracy(output, target):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        # maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(k=1, dim=1, largest=True, sorted=True)  # sorted：返回的结果按照顺序返回
        # 指明是得到前k个数据以及其index; 指定在dim个维度上排序， 默认是最后一个维度;
        # largest：如果为True，按照大到小排序； 如果为False，按照小到大排序;
        pred = pred.t()  # shape is (1, batch_size)
        correct = pred.eq(target.view(1, -1).expand_as(pred))  #

        correct_k = correct[:1].view(-1).float().sum(0, keepdim=True)
        acc = correct_k.mul_(1 / batch_size).cpu().detach().numpy()[0]
        return acc


def remove_prefix(state_dict, prefix):
    """
    Old style model is stored with all names of parameters sharing common prefix 'module.'
    """
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x  # 去除带有prefix的名字
    return {f(key): value for key, value in state_dict.items()}


def add_prefix(state_dict, prefix):
    """
    Old style model is stored with all names of parameters sharing common prefix 'module.'
    """
    print('add prefix \'{}\''.format(prefix))
    f = lambda x: x + prefix  # 去除带有prefix的名字
    return {f(key): value for key, value in state_dict.items()}


def main():
    args = get_args()
    # train_loader, test_loader = load_data(args)  # 返回迭代器
    # TODO: 加载自己的数据
    Batch_size = 16
    num_workers = 8
    annotation_path = './voice_data/train_data.txt'
    annotation_path1 = './voice_data/test_data.txt'
    with open(annotation_path) as f:
        lines = f.readlines()
    with open(annotation_path1) as f1:
        lines1 = f1.readlines()
    # TODO
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    train_dataset = VoiceDataset(lines, (224, 224))
    test_dataset = VoiceDataset(lines1, (224, 224))
    train_loader = DataLoader(train_dataset, batch_size=Batch_size, num_workers=num_workers,
                              pin_memory=True, drop_last=True, collate_fn=voice_dataset_collate)
    test_loader = DataLoader(test_dataset, batch_size=Batch_size, num_workers=num_workers,
                             pin_memory=True, drop_last=True, collate_fn=voice_dataset_collate)

    if args.dataset_mode == "CIFAR10":
        num_classes = 10
    elif args.dataset_mode == "CIFAR100":
        num_classes = 100
    elif args.dataset_mode == "IMAGENET":
        num_classes = 1000
    elif args.dataset_mode == "VOICE":
        num_classes = 3
    print('num_classes: ', num_classes)

    class h_sigmoid(nn.Module):
        def __init__(self, inplace=True):
            super(h_sigmoid, self).__init__()
            self.relu = nn.ReLU6(inplace=inplace)

        def forward(self, x):
            return self.relu(x + 3) / 6

    class h_swish(nn.Module):
        def __init__(self, inplace=True):
            super(h_swish, self).__init__()
            self.sigmoid = h_sigmoid(inplace=inplace)

        def forward(self, x):
            return x * self.sigmoid(x)

    # TODO: 模型加载
    # model = MobileNetV3(model_mode=args.model_mode, num_classes=num_classes, multiplier=args.multiplier,
    #                     dropout_rate=args.dropout).to(device)

    model = mobilenetv3_large()
    model.load_state_dict(torch.load('pretrained/mobilenetv3-large-1cd25616.pth'))
    model.features[0][0] = nn.Conv2d(1, 16, 3, 2, 1, bias=False)
    model.classifier[-1] = nn.Linear(1280, num_classes)
    model.to(device)

    if torch.cuda.device_count() >= 1:
        print("num GPUs: ", torch.cuda.device_count())
        model = nn.DataParallel(model).to(device)

    # TODO: 是否做finetune
    if args.load_pretrained or args.evaluate:
        filename = "best_model_" + str(args.model_mode)
        checkpoint = torch.load('./checkpoint/' + filename + '_ckpt.t7')

        model.load_state_dict(checkpoint['model'], strict=False)
        epoch = checkpoint['epoch']
        acc1 = checkpoint['best_acc1']
        best_acc1 = acc1
        print("Load Model Accuracy1: %.3f" % (acc1.detach().item()), " Load Model end epoch: ", epoch)
    else:
        print("init model load ...")
        epoch = 1
        best_acc1 = 0
    # TODO: 构建优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 5, eta_min=0.000001)
    criterion = nn.CrossEntropyLoss().to(device)

    # TODO: 是否做验证
    if args.evaluate:
        acc1 = validate(test_loader, model, criterion, args)
        print("Acc1: ", acc1)
        return

    if not os.path.isdir("reporting"):
        os.mkdir("reporting")
    # TODO: 训练代码

    for epoch in range(epoch, args.epochs):
        # adjust_learning_rate(optimizer, epoch, args)
        train(train_loader, model, criterion, optimizer, epoch)
        scheduler.step()
        print("lr is ", optimizer.state_dict()['param_groups'][0]['lr'])
        acc1 = validate(test_loader, model, criterion)

        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if is_best:
            print('Saving..')
            state = {
                'model': model.state_dict(),
                'best_acc1': best_acc1,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            filename = "best_model_" + str(args.model_mode)
            torch.save(state, './checkpoint/' + filename + '_ckpt.t7')

        print("val acc: ", acc1)


if __name__ == "__main__":
    main()
