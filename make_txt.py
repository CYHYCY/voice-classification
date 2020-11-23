import os
import random

dict1 = {
    'fatigue': '0',
    'sober': '1',
    "silence": "2"
}
class_list = ['fatigue', 'sober', "silence"]
path = './voice_data/'
datas_txt = os.path.join(path + 'train_data.txt')  # 得到txt文件路径和名字
all_img_path = []
for (root, dirs, files) in os.walk(path):
    if root.split('/')[-1] in class_list:
        img_path = [root + "/" + file for file in files]
        all_img_path.extend(img_path)
random.shuffle(all_img_path)
print(len(all_img_path))
with open(datas_txt, "a+") as file:
    for data in all_img_path:
        char_index = data.strip().split('/')[-2]  # 提取路径中的字段，是类似标签
        label = dict1[char_index]
        file.write(data + ' ' + label + '\n')
