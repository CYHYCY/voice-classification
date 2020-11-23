import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from torch.autograd import Variable
from utils.dataloader import voice_dataset_collate, VoiceDataset
from nets.mobilenetv3 import mobilenetv3_large
from torch.utils.data import DataLoader
from utils.loss import ClassificationLosses
import argparse
import time
import os


class train_model(object):
    def __init__(self):
        args = self.get_args()
        self.create_dir(args)
        self.load_dataset(args)
        args.num_classes = self.get_class(args.dataset_mode)
        self.gpu = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.gpu else "cpu")
        self.model = self.load_model(args)
        self.build_opt_loss(args)
        self.train(args)

    def get_args(self):
        parser = argparse.ArgumentParser("parameters")

        parser.add_argument("--dataset-mode", type=str, default="VOICE",
                            help="(example: CIFAR10, CIFAR100, IMAGENET, VOICE), (default: IMAGENET)")
        parser.add_argument("--epochs", type=int, default=200, help="number of epochs, (default: 100)")
        parser.add_argument("--batch-size", type=int, default=32, help="number of batch size, (default, 512)")
        parser.add_argument("--learning-rate", type=float, default=1e-2, help="learning_rate, (default: 1e-1)")
        parser.add_argument("--dropout", type=float, default=1,
                            help="dropout rate, not implemented yet, (default: 0.8)")
        parser.add_argument('--model-mode', type=str, default="SMALL", help="(example: LARGE, SMALL), (default: LARGE)")
        parser.add_argument("--load-pretrained", type=bool, default=False, help="(default: False)")
        parser.add_argument('--evaluate', type=bool, default=False, help="Testing time: True, (default: False)")
        parser.add_argument('--multiplier', type=float, default=1.0, help="(default: 1.0)")
        parser.add_argument('--print-interval', type=int, default=5,
                            help="training information and evaluation information output frequency, (default: 5)")
        # loss
        parser.add_argument('--loss_type', type=str, default='focal')
        # optimizer
        parser.add_argument('--opt_weight_decay', type=float, default=5e-4)
        # scheduler
        parser.add_argument('--CosineAnnealingLR_T_max', type=int, default=5)
        parser.add_argument('--CosineAnnealingLR_eta_min', type=float, default=1e-6)
        #
        parser.add_argument('--save_num_epoch', type=int, default=1)
        parser.add_argument('--model_weight', type=str, default='./model_weight/')
        parser.add_argument('--pretrained_weight', default='./pretrained/mobilenetv3-large-1cd25616.pth')
        #
        parser.add_argument('--annotation_train_txt', default='./voice_data/train_data.txt')
        parser.add_argument('--annotation_val_txt', default='./voice_data/val_data.txt')
        parser.add_argument('--workers', type=int, default=8)
        parser.add_argument('--distributed', type=bool, default=False)

        args = parser.parse_args()

        return args

    def load_dataset(self, args, ipt_size=(224, 224)):
        Batch_size = args.batch_size
        num_workers = args.workers
        annotation_path = args.annotation_train_txt
        annotation_val_path = args.annotation_val_txt
        with open(annotation_path) as f:
            lines_train = f.readlines()
        with open(annotation_val_path) as f:
            lines_val = f.readlines()
        # TODO
        np.random.seed(101010)
        np.random.shuffle(lines_train)
        np.random.seed(None)
        self.num_train = len(lines_train)
        self.num_val = len(lines_val)
        train_dataset = VoiceDataset(lines_train, ipt_size)
        val_dataset = VoiceDataset(lines_val, ipt_size)
        self.train_loader = DataLoader(train_dataset, batch_size=Batch_size, num_workers=num_workers,
                                       pin_memory=True, drop_last=True, collate_fn=voice_dataset_collate)
        self.val_loader = DataLoader(val_dataset, batch_size=Batch_size, num_workers=num_workers,
                                     pin_memory=True, drop_last=True, collate_fn=voice_dataset_collate)

    def load_model(self, args):
        model = mobilenetv3_large()
        model.load_state_dict(torch.load(args.pretrained_weight))
        model.features[0][0] = nn.Conv2d(1, 16, 3, 2, 1, bias=False)
        model.classifier[-1] = nn.Linear(1280, args.num_classes)
        model.to(self.device)
        if torch.cuda.device_count() >= 1:
            print("num GPUs: ", torch.cuda.device_count())
            model = nn.DataParallel(model).to(self.device)
        return model

    def build_opt_loss(self, args):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate,
                                          weight_decay=args.opt_weight_decay)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                       T_max=args.CosineAnnealingLR_T_max,
                                                                       eta_min=args.CosineAnnealingLR_eta_min)
        loss = ClassificationLosses(cuda=self.gpu)
        self.criterion = loss.build_loss(args.loss_type)
        # self.criterion = nn.CrossEntropyLoss().to(self.device)

    def train_stage(self, train_loader, model, criterion, optimizer, epoch, epoch_size, Epoch):

        model.train()
        acc = []
        total_loss = 0
        start_time = time.time()
        with tqdm(total=epoch_size, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
            for i, (data, target) in enumerate(train_loader):
                if i >= epoch_size:
                    break
                # measure data loading time
                with torch.no_grad():
                    if torch.cuda.is_available():
                        data = Variable(torch.from_numpy(data).type(torch.FloatTensor)).cuda()
                        target = Variable(torch.from_numpy(target).type(torch.FloatTensor)).cuda()
                    else:
                        data = Variable(torch.from_numpy(data).type(torch.FloatTensor))
                        target = Variable(torch.from_numpy(target).type(torch.FloatTensor))

                # compute output
                output = model(data)  # shape is (N, n_classes), target shape is (N), type is tensor
                loss = criterion(output, target.long())
                # loss = sum(loss)
                total_loss += loss.detach().item()
                # measure accuracy and record loss
                acc.append(self.accuracy(output, target.long()))
                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                waste_time = time.time() - start_time
                pbar.set_postfix(**{'total_loss': total_loss / (i + 1),
                                    'lr': self.get_lr(optimizer),
                                    'step/s': waste_time})
                pbar.update(1)
                start_time = time.time()
            # print("epoch{} acc：{} loss: {}".format(epoch, np.mean(np.array(acc)), total_loss))
        return total_loss

    def validate_stage(self, val_loader, model, criterion, epoch, epoch_size, Epoch):
        model.eval()
        valid_total_loss = 0
        acc = []
        with tqdm(total=epoch_size, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
            with torch.no_grad():
                for i, (data, target) in enumerate(val_loader):
                    if i >= epoch_size:
                        break
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
                        acc.append(self.accuracy(output, target.long()))
                    pbar.set_postfix(**{'total_loss': valid_total_loss / (i + 1)})
                    pbar.update(1)
                # print("val loss is ", valid_total_loss)
        return valid_total_loss, np.mean(np.array(acc))

    def accuracy(self, output, target):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            # maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(k=1, dim=1, largest=True, sorted=True)  # sorted：返回的结果按照顺序返回
            # 指明是得到前k个数据以及其index; 指定在dim个维度上排序， 默认是最后一个维度;
            # largest：如果为True，按照大到小排序； 如果为False，按照小到大排序;
            pred = pred.t()  # shape is (1, batch_size)
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            correct_k = correct[:1].view(-1).float().sum(0, keepdim=True)
            acc = correct_k.mul_(1 / batch_size).cpu().detach().numpy()[0]
        return acc

    def remove_prefix(self, state_dict, prefix):
        """
        Old style model is stored with all names of parameters sharing common prefix 'module.'
        """
        print('remove prefix \'{}\''.format(prefix))
        f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x  # 去除带有prefix的名字
        return {f(key): value for key, value in state_dict.items()}

    def create_dir(self, args):
        if not os.path.exists(args.model_weight):
            os.mkdir(args.model_weight)

    def add_prefix(self, state_dict, prefix):
        """
        Old style model is stored with all names of parameters sharing common prefix 'module.'
        """
        print('add prefix \'{}\''.format(prefix))
        f = lambda x: x + prefix  # 去除带有prefix的名字
        return {f(key): value for key, value in state_dict.items()}

    def adjust_learning_rate(self, optimizer, epoch, args):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = args.learning_rate * (0.1 ** (epoch // 30))
        # print(lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def get_class(self, dataset_mode):
        if dataset_mode == "CIFAR10":
            num_classes = 10
        elif dataset_mode == "CIFAR100":
            num_classes = 100
        elif dataset_mode == "IMAGENET":
            num_classes = 1000
        elif dataset_mode == "VOICE":
            num_classes = 3
        return num_classes

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def train(self, args):
        epoch_size_train = max(1, self.num_train // args.batch_size)
        epoch_size_val = self.num_val // args.batch_size
        for epoch in range(args.epochs):
            train_total_loss = self.train_stage(self.train_loader, self.model, self.criterion, self.optimizer,
                                                epoch, epoch_size_train, args.epochs)
            valid_total_loss, val_acc = self.validate_stage(self.val_loader, self.model, self.criterion, epoch,
                                                            epoch_size_val, args.epochs)
            self.lr_scheduler.step()
            if (epoch + 1) % args.save_num_epoch == 0:
                print('Saving state, iter:', str(epoch + 1))
                torch.save(self.model.state_dict(), './model_weight/Epoch%d-Train_Loss%.4f-Val_Loss%.4f.pth' % (
                    (epoch + 1), train_total_loss / (epoch_size_train + 1), valid_total_loss / (epoch_size_val + 1)))
            print('Finish Validation')
            print('Epoch:' + str(epoch + 1) + '/' + str(args.epochs))
            print('Train Total Loss: %.4f || Val Total Loss: %.4f || Val acc: %.4f ' % (
                train_total_loss / (epoch_size_train + 1), valid_total_loss / (epoch_size_val + 1), val_acc))


if __name__ == "__main__":
    model = train_model()
