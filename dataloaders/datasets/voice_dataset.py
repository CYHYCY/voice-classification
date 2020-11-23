from random import shuffle
import numpy as np
import os
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import librosa


class VoiceDataset(Dataset):
    def __init__(self, train_lines, image_size=(224, 224), split='train'):
        """

        :param image_size: 输入图片的大小
        :param mosaic: 是否使用mosaic的标志位
        """
        super(VoiceDataset, self).__init__()
        self.train_lines = train_lines
        self.train_batches = len(train_lines)
        self.image_size = image_size
        self.train_lines = []
        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.train_lines)))

    def __len__(self):
        return self.train_batches

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, annotation_line, input_shape):
        """
        实时数据增强的随机预处理
        :param annotation_line:
        :param input_shape: 网络的输入
        :return: image shape is (C, H, W)
        """
        line = annotation_line.split('.jpg')  # 应对./data/JPGImages/img ().jpg 12 13 44 56 0的情况
        line_path = line[0] + ".jpg"
        line_label = line[-1].strip().split()
        # TODO: 根据路径读取音频信息
        x, sr = librosa.load(line_path, sr=None)  # 采样频率，默认22050
        mfcc = librosa.feature.mfcc(x, sr=sr, n_mfcc=64) / 1000  # n_mfcc为每帧多少维特征
        # TODO: 做归一化操作
        image = Image.fromarray(mfcc)
        image = image.resize(input_shape, Image.BILINEAR)  # input_shape is (width, height)
        # image shape is input_shape
        # 是否翻转图片
        if self.rand() < .5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)  # 左右翻转
        image = np.expand_dims(np.asarray(image), axis=0)  # shape is (C, H, W), 即(1, 48, t)

        return image, np.array(line_label, dtype=np.float32)

    def __getitem__(self, index):
        """
        :param index:
        :return: tmp_inp shape is (image_size, image_size)
        tmp_targets shape is (N, 5),(x, y, w, h, label)归一化的值
        """
        if index == 0:
            shuffle(self.train_lines)
        lines = self.train_lines
        n = self.train_batches
        index = index % n
        tmp_inp, tmp_targets = self.get_random_data(lines[index], self.image_size)
        return tmp_inp, tmp_targets


# DataLoader中collate_fn使用
def yolo_dataset_collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = np.asarray(images)
    bboxes = np.asarray(bboxes)
    return images, bboxes


if __name__ == "__main__":
    Batch_size_first = 16
    num_workers = 0
    annotation_path = '../data/fire/test1.txt'
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * 0.1)
    num_train = len(lines) - num_val
    train_dataset = VoiceDataset(lines[:num_train], (416, 416), mosaic=True)
    gen = DataLoader(train_dataset, batch_size=Batch_size_first, num_workers=num_workers,
                     pin_memory=True, drop_last=True, collate_fn=yolo_dataset_collate)
    for _ in range(2):
        for iteration, batch in enumerate(gen):
            # print("iter11111:", iteration)
            pass
