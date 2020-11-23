from __future__ import print_function, division
import os
import librosa
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from dataloaders import custom_transforms as tr


class Segmentation(Dataset):
    """
    own_matting dataset
    """
    NUM_CLASSES = 1

    def __init__(self, args, split='train'):
        super().__init__()
        self.args = args
        self.split = split
        self.image_paths = []
        self.label_paths = []

        with open(os.path.join(split + '_data', '.txt'), "r") as f:  # 打开txt文件路径
            lines = f.read().splitlines()

        for sum_path in lines:
            self.image_paths.append(sum_path.split(' ')[0])
            self.label_paths.append(sum_path.split(' ')[1])

        assert (len(self.image_paths) == len(self.label_paths))

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.image_paths)))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target}

        if self.split == "train":
            return self.transform_tr(sample)
        elif self.split == 'val':
            return self.transform_val(sample)

    def _make_img_gt_point_pair(self, index, n_mfcc=224):
        """
        对数据进行读取操作，返回数据和标签
        """
        # TODO: 读取数据
        x, sr = librosa.load(self.image_paths[index], sr=None)  # 音频信息和采样频率
        mfcc = librosa.feature.mfcc(x, sr=sr, n_mfcc=n_mfcc)  # shape is (n_mfcc, t),t为帧数，一般与音频时长相关
        # TODO: 将数据大小进行转换
        # 1.将数据转为3维数据；2.对数据进行归一化操作；3.对数据进行数据增强的操作
        _img = Image.open(self.image_paths[index]).convert('RGB')
        _target = int(self.image_paths[index].strip().split()[-1])
        return _img, _target

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            transforms.Resize(size=(self.args.base_size, self.args.crop_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])
        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            transforms.Resize(size=(self.args.base_size, self.args.crop_size)),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])
        return composed_transforms(sample)

    def __str__(self):
        return 'own_matting(split=' + str(self.split) + ')'


if __name__ == '__main__':
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 600
    args.crop_size = 600

    voc_train = Segmentation(args, split='train')

    dataloader = DataLoader(voc_train, batch_size=5, shuffle=True, num_workers=0)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='pascal')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)
