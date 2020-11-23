import torch
import librosa
from PIL import Image
import numpy as np
from torch.autograd import Variable
from model import MobileNetV3


class voice_classficication(object):
    def __init__(self):
        self.classes = self.read_class_names("./place.names")
        self.classes_list = [str(key) for key, _ in self.classes.items()]
        self.num_classes = len(self.classes)
        self.device = torch.cuda.is_available()
        self.load_model()

    def inference(self, path):
        data = self.load_input(path)
        with torch.no_grad():
            if self.device:
                data = Variable(torch.from_numpy(data).type(torch.FloatTensor)).cuda()
            else:
                data = Variable(torch.from_numpy(data).type(torch.FloatTensor))
            output = self.model(data)
            _, pred = output.topk(k=1, dim=1, largest=True, sorted=True)  # sorted：返回的结果按照顺序返回
            # 指明是得到前k个数据以及其index; 指定在dim个维度上排序， 默认是最后一个维度;
            # largest：如果为True，按照大到小排序； 如果为False，按照小到大排序;
        return pred.cpu().numpy()

    def load_model(self):
        self.model = MobileNetV3(model_mode="SMALL", num_classes=self.num_classes, multiplier=1.0,
                                 dropout_rate=1)
        if self.device:
            self.model.cuda()
        checkpoint = torch.load("./checkpoint/best_model_SMALL_ckpt.t7")
        self.model.load_state_dict(checkpoint['model'], strict=False)
        self.model.eval()

    def load_input(self, path, input_shape=(224, 224)):
        x, sr = librosa.load(path, sr=None)  # 采样频率，默认22050
        mfcc = librosa.feature.mfcc(x, sr=sr, n_mfcc=64) / 1000  # n_mfcc为每帧多少维特征
        # TODO: 做归一化操作
        image = Image.fromarray(mfcc)
        image = image.resize(input_shape, Image.BILINEAR)  # input_shape is (width, height)
        # 是否翻转图片
        image = np.expand_dims(np.asarray(image), axis=0)  # shape is (C, H, W), 即(1, 48, t)
        image = np.expand_dims(np.asarray(image), axis=0)
        return image

    def read_class_names(self, class_file_name):
        """
        loads class name from a file
        加载id和类别名称映射关系，文件中是每行一个类别名称
        :param class_file_name:
        :return: names：key is 阿拉伯数字 value is 具体种类
        """
        names = {}
        with open(class_file_name, 'r') as data:
            for ID, name in enumerate(data):
                names[ID] = name.strip('\n')
        return names

    def index_2_label(self, pre, char_set):
        """
        将预测值转换为真实标签
        :param pre: shape is (batch_size, num_classes)
        :param char_set:类别字典
        :return:
        """
        pre_str = []
        nums, lens = np.shape(pre)
        for i in range(nums):  # 遍历所有数据
            a = []
            for j in range(lens):  # 遍历每条数据中的值
                a.append(char_set[pre[i][j]])
            pre_str.append(a)
        return np.asarray(pre_str)


if __name__ == "__main__":
    vc_model = voice_classficication()
    while True:
        path = input("input path:")
        y_predict = vc_model.inference(path)
        print(y_predict)
