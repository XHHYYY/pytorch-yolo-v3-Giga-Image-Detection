import cv2
from test_detector import detector
import numpy as np

# img = cv2.imread('./imgs/eagle.jpg')
# Detect = detector('./det/', confidence=0.8)
# # todo 将原代码中的图片读取转换为传入图片矩阵
# Detect.read_images('./imgs/eagle.jpg')
# Detect.detect_objects()
# Detect.output_result()
flag = 1
det_path = './det/{}/'.format(flag)

Detect = detector(det_path, confidence=0.8)
for i in range(9 + flag):
    for j in range(9 + flag):
        Detect.read_images('./cutted/{0}/{1}_{2}_{3}.jpg'.format(flag, flag, i, j))
        Detect.detect_objects()
        Detect.output_result()



# todo 读取输出的数据
result = np.loadtxt('./results/result{}.csv'.format(det_path[-2]), dtype=int, delimiter=',')
result.reshape(-1, 8)
np.savetxt('./results/final_result{}.csv'.format([det_path[-2]]), fmt='%d', delimiter=',')

# todo 将所有bbox坐标转化为全局坐标，再 1、根据IOU去重 2、合并同一物体不同部位的bbox