import cv2
import numpy as np
for k in range(2):
    flag = k + 1
    # 将检测后画好bbox的图像拼合为完整图像：detected1.jpg 或 detected2.jpg
    for i in range(9 + flag):
        for j in range(9 + flag):
            temp_img = cv2.imread('./det/{0}/det_{1}_{2}_{3}.jpg'.format(flag, flag, i, j))
            himg = temp_img if j == 0 else np.concatenate((himg, temp_img), axis=1)
        vimg = himg if i == 0 else np.concatenate((vimg, himg), axis=0)
    cv2.imwrite('./det/detected{}.jpg'.format(flag), vimg)