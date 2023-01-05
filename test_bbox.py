import numpy as np
import json

class rect():
    def __init__(self, c1, c2) -> None:
        self.c1 = c1
        self.c2 = c2
        self.points = [c1, (c1[0], c2[1]), c2, (c2[0], c1[1])] # 左上起逆时针
        self.area = (c1[0] - c2[0]) * (c1[1] - c2[1])
        

def compute_iou(rec1: tuple, rec2: tuple) -> int:

    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return intersect
        
        
class check_img_Exception(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)
        self.name = args[0]

# img = cv2.imread('./imgs/test.jpg')
with open('./results/config.json', 'r') as f:
    config = json.load(f)
    f.close()
Bb1 = np.loadtxt('./results/final_result1.csv', dtype=int, delimiter=',')
h, _ = Bb1.shape
Bb1[:, 0] = Bb1[:, 0] + Bb1[:, -1] * config['w'] 
Bb1[:, 2] = Bb1[:, 2] + Bb1[:, -1] * config['w'] 
Bb1[:, 1] = Bb1[:, 1] + Bb1[:, -2] * config['h'] 
Bb1[:, 3] = Bb1[:, 3] + Bb1[:, -2] * config['h'] 

Bb2 = np.loadtxt('./results/final_result2.csv', dtype=int, delimiter=',')
h, _ = Bb2.shape
Bb2[:, 0] = Bb2[:, 0] + Bb2[:, -1] * config['w'] - config['w'] // 2 
Bb2[:, 2] = Bb2[:, 2] + Bb2[:, -1] * config['w'] - config['w'] // 2 
Bb2[:, 1] = Bb2[:, 1] + Bb2[:, -2] * config['h'] - config['h'] // 2 
Bb2[:, 3] = Bb2[:, 3] + Bb2[:, -2] * config['h'] - config['h'] // 2 

Bb = np.concatenate((Bb1, Bb2), axis=0)


h, _ = Bb.shape
for i in range(h):
    for j in range(h):
        if i == j:
            continue
        rect1 = rect((Bb[i, 0], Bb[i, 1]), (Bb[i, 2], Bb[i, 3]))
        rect2 = rect((Bb[j, 0], Bb[j, 1]), (Bb[j, 2], Bb[j, 3]))
        
        area = compute_iou((rect1.c1[0], rect1.c1[1], rect1.c2[0], rect1.c2[1]), (rect2.c1[0], rect2.c1[1], rect2.c2[0], rect2.c2[1]))
        
        if area == 0:
            continue
        elif (area / rect1.area) > 0.8:
            Bb[i] = np.zeros((1, 8))