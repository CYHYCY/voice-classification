import matplotlib.pyplot as plt
import librosa.display
import pandas as pd
import numpy as np

# 音乐时域信息
x, sr = librosa.load("./out000.wav.", sr=8000)
print(max(x), min(x), sr, x.shape)
plt.figure(figsize=(14, 5))
librosa.display.waveplot(x, sr=sr)
plt.show()

# 保存csv
# music = pd.DataFrame(data=x, columns=['acoustic_data'])
# music.to_csv('Toxic_music.csv',index=0)

# 时变频谱图
z = librosa.stft(x)
zdb = librosa.amplitude_to_db(abs(z))  # 把幅度转成分贝格式
plt.figure(figsize=(14, 5))
librosa.display.specshow(zdb, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar()
plt.show()

# #过零率
# plt.figure(figsize=(14, 5))
# plt.plot(x[100:150000])
# plt.grid()
# plt.show()
# zero_crossings = librosa.zero_crossings(x[100:150000], pad=False)
# print("零点个数", sum(zero_crossings))


# mfcc特征
mfcc = librosa.feature.mfcc(x, sr=sr, n_mfcc=32)  # n_mfcc为每帧多少维特征，那一帧指的什么？
# mfcc2 = librosa.feature.mfcc(x, sr=sr, n_mfcc=1)
print(mfcc.shape)

mfcc1 = librosa.feature.mfcc(x, sr=sr)

librosa.display.specshow(mfcc, sr=sr, x_axis='time')
plt.show()
librosa.display.specshow(mfcc1, sr=sr, x_axis='time')
plt.show()

# Log-Mel Spectrogram 特征
melspec = librosa.feature.melspectrogram(x, sr, n_fft=1024, hop_length=512, )
logmelspec = librosa.power_to_db(melspec)  # convert to log scale 7
print(logmelspec.shape)  # 这里帧数和mfcc一样，还不错
plt.show()
zhaohedong = 777
