import cv2
import numpy as np
import json

def cut(a = 10, b = 10, img = None) -> None:
    '''
    - 输入：
        1. 图片切割数量：默认切成10*10
        2. 图片，若不输入则为默认读取路径( './imgs/test.jpg' )
    - 输出：将切好的图片保存至默认路径( './cutted/1/' 或 './cutted/2/' )
    '''
    
    if img is None:
        # 默认图片路径
        origin = cv2.imread('./imgs/test.jpg')
    else:
        origin = img
    # origin为待处理图片
    
    H, W = origin.shape[0:2]
    # h, w 为切分后图片的高、宽
    h = H // a
    w = W // b
    
    

    
    # 切割图片
    for i in range(a):
        for j in range(b):
            cv2.imwrite('./cutted/' + (str(1) if a == 10 else str(2))  + '/' + (str(1) if a == 10 else str(2)) + '_' + str(i) + '_' + str(j) + '.jpg', 
                        origin[i*h : i*h+h, j*w : j*w+w] if (i!=11 and j !=11) else
                        origin[i*h : i*h+h, j*w : ] if j == 11 else
                        origin[i*h : , j*w : j*w+w] if i == 11 else
                        origin[i*h : , j*w : ])
    # 若切完第二组图片则返回
    if a == 11:
        return
    # 否则对原图做padding
    
    # 将切割图片信息存储进`config.json`
    data = {"a" : a, "b" : b, "H" : H, "W": W, "h" : h, "w" : w}
    with open('./results/config.json', 'w') as file:
        json.dump(data, file, indent=4)
    file.close()
    
    temp1 = np.zeros((H, w//2, 3))
    temp2 = np.zeros((h//2, W+2*(w//2), 3))
    padding = np.hstack((temp1, origin, temp1))
    padding = np.vstack((temp2, padding, temp2))
    
    
    data = {"a" : a, "b" : b, "H" : H, "W": W, "h" : h, "w" : w}
    with open('./results/config.json', 'w') as file:
        json.dump(data, file, indent=4)
    file.close()

    # 切分padding后的图片
    cut(a = 11, b = 11, img = padding)
    return

if __name__ == '__main__':
    # 函数自动读取路径并裁剪图片保存在默认文件夹中
    cut()