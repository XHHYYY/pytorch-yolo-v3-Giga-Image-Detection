# %%
import cv2
from test_detector import detector
import numpy as np

# %%
flag = 2

# %%



det_path = './det/{}/'.format(flag)

Detect = detector(det_path, confidence=0.8)
for i in range(9 + flag):
    for j in range(9 + flag):
        Detect.read_images('./cutted/{0}/{1}_{2}_{3}.jpg'.format(flag, flag, i, j))
        Detect.detect_objects()
        Detect.output_result()


# %%

# todo 读取输出的数据
result = np.loadtxt('./results/original_result{}.csv'.format(flag), dtype=int, delimiter=',')
np.savetxt('./results/final_result{}.csv'.format(flag), result.reshape(-1, 8), fmt='%d', delimiter=',')

# todo 将所有bbox坐标转化为全局坐标，再 1、根据IOU去重 2、合并同一物体不同部位的bbox


