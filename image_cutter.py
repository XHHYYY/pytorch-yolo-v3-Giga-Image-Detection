import cv2

def cut(a = 10, b = 10, img = None) -> None:
    if img is None:
        origin = cv2.imread('./imgs/test.jpg')
    else:
        origin = img
    H, W = origin.shape[0:2]
    h = H // a
    w = W // b
    
    for i in range(a):
        for j in range(b):
            cv2.imwrite('./cutted/' + (str(1) if a == 10 else str(2))  + '/' + str(i) + '_' + str(j) + '.jpg', 
                        origin[i*h : i*h+h, j*w : j*w+w] if (i!=9 and j !=9) else
                        origin[i*h : i*h+h, j*w : ] if j == 9 else
                        origin[i*h : , j*w : j*w+w] if i == 9 else
                        origin[i*h : , j*w : ])
    if a == 9:
        return
    cut(a = 9, b = 9, img = origin[h//2 : -1-h//2, w//2 : -1-w//2])
    
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