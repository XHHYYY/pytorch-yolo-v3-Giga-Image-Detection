import numpy as np
import cv2
for i in range(10):
    temp = cv2.imread('./det/det_1_4_'+str(i)+'.jpg')
    img1 = temp if i == 0 else np.hstack((img1, temp))
for i in range(10):
    temp = cv2.imread('./det/det_1_5_'+str(i)+'.jpg')
    img2 = temp if i == 0 else np.hstack((img2, temp))
for i in range(9):
    temp = cv2.imread('./det/det_2_4_'+str(i)+'.jpg')
    img3 = temp if i == 0 else np.hstack((img3, temp))
    
cv2.imwrite('./det/ori.jpg', np.vstack((img1, img2)))
cv2.imwrite('./det/down.jpg', img3)