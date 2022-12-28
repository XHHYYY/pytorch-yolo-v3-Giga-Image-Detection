import cv2
from test_detector import detector
import numpy as np

# img = cv2.imread('./imgs/eagle.jpg')
# Detect = detector('./det/', confidence=0.8)
# # todo 将原代码中的图片读取转换为传入图片矩阵
# Detect.read_images('./imgs/eagle.jpg')
# Detect.detect_objects()
# Detect.output_result()

Detect = detector('./det/', confidence=0.8)
for i in range(10):
    Detect.read_images('./cutted/1/5_' + str(i) + '.jpg')
    Detect.detect_objects()
    Detect.output_result()



# todo 读取输出的数据
# result = np.loadtxt('./results/result.csv', dtype=int, delimiter=',')
# result.reshape(-1, 8)