import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from utils.dataloader import voice_dataset_collate, VoiceDataset
from model import MobileNetV3
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
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
    parser.add_argument("--load-pretrained", type=bool, default=True, help="(default: False)")
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
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    # top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1, prefix="Epoch: [{}]".format(epoch))
    # progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1, top5,
    #                          prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    end = time.time()
    for i, (data, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        with torch.no_grad():
            if torch.cuda.is_available():
                data = Variable(torch.from_numpy(data).type(torch.FloatTensor)).cuda()
                target = Variable(torch.from_numpy(target).type(torch.FloatTensor)).cuda()
                # target = [Variable(torch.from_numpy(np.array(ann)).type(torch.FloatTensor)) for ann in target]
            else:
                data = Variable(torch.from_numpy(data).type(torch.FloatTensor))
                target = Variable(torch.from_numpy(target).type(torch.FloatTensor))

        # compute output
        output = model(data)
        loss = criterion(output, target.long())

        # measure accuracy and record loss
        acc1 = accuracy(output, target.long(), topk=(1, 5))
        losses.update(loss.item(), data.size(0))
        top1.update(acc1[0], data.size(0))
        # top5.update(acc5[0], data.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_interval == 0:
            progress.print(i)


def validate1(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5,
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (data, target) in enumerate(val_loader):
            with torch.no_grad():
                if torch.cuda.is_available():
                    data = Variable(torch.from_numpy(data).type(torch.FloatTensor)).cuda()
                    target = Variable(torch.from_numpy(target).type(torch.FloatTensor)).cuda()
                    # target = [Variable(torch.from_numpy(np.array(ann)).type(torch.FloatTensor)) for ann in target]
                else:
                    data = Variable(torch.from_numpy(data).type(torch.FloatTensor))
                    target = Variable(torch.from_numpy(target).type(torch.FloatTensor))

            # compute output
            output = model(data)
            loss = criterion(output, target.long())

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target.long(), topk=(1, 5))
            losses.update(loss.item(), data.size(0))
            top1.update(acc1[0], data.size(0))
            top5.update(acc5[0], data.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_interval == 0:
                progress.print(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
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

            # measure accuracy and record loss
            acc1 = accuracy(output, target.long(), topk=(1, 5))
            losses.update(loss.item(), data.size(0))
            top1.update(acc1[0], data.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_interval == 0:
                progress.print(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        # maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(k=1, dim=1, largest=True, sorted=True)  # sorted：返回的结果按照顺序返回
        # 指明是得到前k个数据以及其index; 指定在dim个维度上排序， 默认是最后一个维度;
        # largest：如果为True，按照大到小排序； 如果为False，按照小到大排序;
        pred = pred.t()  # shape is (1, batch_size)
        correct = pred.eq(target.view(1, -1).expand_as(pred))  #

        res = []  # 存放准确率的
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


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
    Batch_size = 32
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

    # TODO: 模型加载
    model = MobileNetV3(model_mode=args.model_mode, num_classes=num_classes, multiplier=args.multiplier,
                        dropout_rate=args.dropout).to(device)

    for para_tensor in model.state_dict():
        print(model.state_dict()[para_tensor].size())

    if torch.cuda.device_count() >= 1:
        print("num GPUs: ", torch.cuda.device_count())
        model = nn.DataParallel(model).to(device)

    # TODO: 是否做finetune
    if args.load_pretrained or args.evaluate:
        filename = "best_model_" + str(args.model_mode)
        checkpoint = torch.load('./checkpoint/' + filename + '_ckpt.t7')
        # # TODO: 将model中的module.去掉
        # if "state_dict" in model.keys():
        #     pretrained_dict = remove_prefix(model['state_dict'], 'module.')
        # else:
        #     pretrained_dict = remove_prefix(model, 'module.')
        # model.load_state_dict(pretrained_dict, strict=False)
        #
        # # TODO: 恢复权重
        # checkpoint['model'] = add_prefix(model, 'module.')
        model.load_state_dict(checkpoint['model'], strict=False)
        epoch = checkpoint['epoch']
        acc1 = checkpoint['best_acc1']
        # acc5 = checkpoint['best_acc5']
        best_acc1 = acc1
        # print("Load Model Accuracy1: ", acc1, " acc5: ", acc5, "Load Model end epoch: ", epoch)
        print("Load Model Accuracy1: ", acc1, "Load Model end epoch: ", epoch)
    else:
        print("init model load ...")
        epoch = 1
        best_acc1 = 0
    # TODO: 构建优化器和损失函数
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=1e-5, momentum=0.9)
    # optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss().to(device)

    # TODO: 是否做验证
    if args.evaluate:
        acc1 = validate(test_loader, model, criterion, args)
        # acc1, acc5 = validate(test_loader, model, criterion, args)
        # print("Acc1: ", acc1, "Acc5: ", acc5)
        print("Acc1: ", acc1)
        return

    if not os.path.isdir("reporting"):
        os.mkdir("reporting")
    # TODO: 训练代码
    start_time = time.time()
    with open("./reporting/" + "best_model_" + args.model_mode + ".txt", "w") as f:
        for epoch in range(epoch, args.epochs):
            adjust_learning_rate(optimizer, epoch, args)
            train(train_loader, model, criterion, optimizer, epoch, args)
            acc1 = validate(test_loader, model, criterion, args)
            # acc1, acc5 = validate(test_loader, model, criterion, args)

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

            time_interval = time.time() - start_time
            time_split = time.gmtime(time_interval)
            print("Training time: ", time_interval, "Hour: ", time_split.tm_hour, "Minute: ", time_split.tm_min,
                  "Second: ", time_split.tm_sec, end='')
            print(" Test best acc1:", best_acc1, " acc1: ", acc1)

            f.write("Epoch: " + str(epoch) + " " + " Best acc: " + str(best_acc1) + " Test acc: " + str(acc1) + "\n")
            f.write("Training time: " + str(time_interval) + " Hour: " + str(time_split.tm_hour) + " Minute: " + str(
                time_split.tm_min) + " Second: " + str(time_split.tm_sec))
            f.write("\n")


if __name__ == "__main__":
    main()
