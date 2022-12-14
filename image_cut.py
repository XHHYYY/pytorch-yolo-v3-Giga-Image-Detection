import cv2
from test_detector import detector

img = cv2.imread('./imgs/eagle.jpg')
Detect = detector('./det/', confidence=0.8)
# todo 将原代码中的图片读取转换为传入图片矩阵
Detect.read_images('./imgs/eagle.jpg')
Detect.detect_objects()
Detect.output_result()
