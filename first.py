import scipy.io.wavfile as wav
import numpy as np
import matplotlib.pyplot as plt
import os
import wave


"""
读取双通道波形并绘制波形图
"""

# 打开WAV音频
f = wave.open("E:\Code\PyCharm\Video\Data\Summer.wav", "rb")
# 读取格式信息
# (声道数、     量化位数、   采样频率、  采样点数、 压缩类型、  压缩类型的描述)
# (nchannels, sampwidth, framerate, nframes, comptype, compname)
params = f.getparams()
nchannels, sampwidth, framerate, nframes = params[:4]
print("声道数---", nchannels)
print("量化位数---", sampwidth)
print("采样频率---", framerate)
print("采样点数---", nframes)
# 读取nframes个数据，返回字符串格式
str_data = f.readframes(nframes)
f.close()
# 将字符串转换为数组，得到一维的short类型的数组
wave_data = np.fromstring(str_data, dtype=np.short)
# 赋值的归一化
wave_data = wave_data * 1.0 / (max(abs(wave_data)))
# 整合左声道和右声道的数据
wave_data = np.reshape(wave_data, [nframes, nchannels])
# wave_data.shape = (-1, 2)   # -1的意思就是没有指定,根据另一个维度的数量进行分割
# 最后通过采样点数和取样频率计算出每个取样的时间
time = np.arange(0, nframes) * (1.0 / framerate)

plt.figure()
# 左声道波形
plt.subplot(3, 1, 1)
plt.plot(time, wave_data[:, 0])
plt.xlabel("time (seconds)")
plt.ylabel("Amplitude")
plt.title("Left channel")
plt.grid()  # 标尺

plt.subplot(3, 1, 3)
# 右声道波形
plt.plot(time, wave_data[:, 1], c="g")
plt.xlabel("time (seconds)")
plt.ylabel("Amplitude")
plt.title("Left channel")
plt.title("right channel")
plt.grid()

plt.show()





