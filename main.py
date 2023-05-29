import numpy as np
import cv2
from tools import show_picture
import matplotlib.pyplot as plt
import pandas as pd


img = cv2.imread('data/6.png') #read image
gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)#convert to hsv
mads = []
for i in range(gray_img.shape[0]):
    # 逐行扫描，计算平均绝对偏差
    mad = np.mean(np.abs(gray_img[i] - np.mean(gray_img[i])))
    mads.append(mad)
    print('Mean Absolute Deviation (MAD):', mad)
plt.plot(mads)
plt.show()
cv2.imwrite('data/processed.png', gray_img)

df = pd.DataFrame(mads)
df.to_csv('data/mads.csv', index=False)

# 构建差值曲线，横坐标为图像的高，纵坐标为每一横向的平均绝对偏差
curve = np.array(mads)

peaks = []
valleys = []
# 找到差分符号变化的位置，即波峰和波谷的位置
for i in range(1, len(mads)-1):
    if mads[i-1] < mads[i] and mads[i] > mads[i+1]:
        peaks.append(i)
    elif mads[i-1] > mads[i] and mads[i] < mads[i+1]:
        valleys.append(i)
peaks = np.array(peaks)
valleys = np.array(valleys)

# 过滤波峰
filtered_peaks = []
filtered_valleys = []
for i in range(len(peaks)):
    peak = peaks[i]

    previous_peak = filtered_peaks[-1] if filtered_peaks else -1
    if i == 0:
        filtered_peaks.append(peak)
        continue

    if peak - previous_peak > 7:
        filtered_peaks.append(peak)

        if len(filtered_peaks) > 1:
            # 找到当前波峰前的波谷
            previous_valleys = valleys[(peak > valleys) & (valleys > previous_peak)]

            minimum_valley = 100
            for v in previous_valleys:
                if mads[v] < minimum_valley:
                    previous_valley = v

            filtered_valleys.append(previous_valley)

# 绘制曲线和波峰
plt.plot(curve)
plt.plot(filtered_peaks, curve[filtered_peaks], 'ro')  # 使用红色圆点标记波峰
plt.plot(filtered_valleys, curve[filtered_valleys], 'bo')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Curve with Peaks (Filtered by Distance)')
plt.show()

# 统计波峰数量
peak_count = len(filtered_peaks)
print(filtered_peaks)
print(filtered_valleys)
print('Number of Peaks:', peak_count)