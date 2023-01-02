import cv2
import numpy as np
import json

def cut(a = 10, b = 10, img = None) -> None:
    if img is None:
        origin = cv2.imread('./imgs/test.jpg')
    else:
        origin = img
    H, W = origin.shape[0:2]
    h = H // a
    w = W // b
    
    data = {"a" : a, "b" : b, "H" : H, "W": W, "h" : h, "w" : w}
    with open('./results/config.json', 'w') as file:
        json.dump(data, file, indent=4)
    file.close()
    
    for i in range(a):
        for j in range(b):
            cv2.imwrite('./cutted/' + (str(1) if a == 10 else str(2))  + '/' + (str(1) if a == 10 else str(2)) + '_' + str(i) + '_' + str(j) + '.jpg', 
                        origin[i*h : i*h+h, j*w : j*w+w] if (i!=11 and j !=11) else
                        origin[i*h : i*h+h, j*w : ] if j == 11 else
                        origin[i*h : , j*w : j*w+w] if i == 11 else
                        origin[i*h : , j*w : ])
    if a == 11:
        return
    temp1 = np.zeros((H, w//2, 3))
    temp2 = np.zeros((h//2, W+2*(w//2), 3))
    padding = np.hstack((temp1, origin, temp1))
    padding = np.vstack((temp2, padding, temp2))
    # cut(a = 9, b = 9, img = origin[h//2 : -1-h//2, w//2 : -1-w//2])
    cut(a = 11, b = 11, img = padding)
    return



# 演示窗口重叠
# def draw_windows():
#     origin = cv2.imread('./imgs/test.jpg')
#     H, W = origin.shape[0:2]
#     a = 10
#     b = 10
#     h = H // a
#     w = W // b
    
#     H_s = H - h
#     W_s = W - w
    
#     for i in range(a):
#         for j in range(b):
            

if __name__ == '__main__':
    cut()