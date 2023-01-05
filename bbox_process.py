# %% [markdown]
# ## 匹配去割裂

# %%
import numpy as np
import cv2
import json
from typing import List
import pickle as pkl
import random


# %%


class check_img_Exception(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)
        self.name = args[0]
        


class Bbox():
    def __init__(self, data: np.ndarray) -> None:
        self.c1 = data[0:2]
        self.c2 = data[2:4]
        self.label = data[4]
        self.Iposi = data[-2:]
        self.area = (self.c2[0] - self.c1[0]) * (self.c2[1] - self.c1[1])


class block_image():
    def __init__(self, bbox: List[Bbox]) -> None:
        self.Bboxes = bbox
        if len(self.Bboxes) != len(list(filter(lambda x: (x.Iposi == self.Bboxes[0].Iposi).all(), self.Bboxes))):
            raise check_img_Exception('Contain Bbox from other image!')
        self.position = self.Bboxes[0].Iposi

def run_process(threshold: tuple, good_Bbox: block_image, block: block_image, down: block_image,right: block_image,
            right_up: block_image, right_down: block_image, left_down: block_image) -> None:
    '''
    # 功能说明
    ## 输入
    - 待处理图像
    - 第一组的下图、右图
    - 第二组的左下、右下、右上图
    ## 结果
    - 删除割裂Bbox
    - 保留原图完整Bbox
    - 记录第二组图完整的Bbox并返回

    # 注意
    未考虑双图冒险及人物过大导致的一个人在两组图中均有割裂的问题
    '''

    # flag = 0,1对应下右
    # 逻辑：对待处理图像：检查下方和右方图像匹配bbox，存在匹配则先使用第二组图像去割裂
    # todo 暂时不考虑双图冒险
    # todo 如何处理第二组图中有而地组图中没有的框
    # 2675 * 1505

    with open('./results/config.json', 'r') as f:
        config = json.load(f)
        f.close()

    # down:
    if down is not None:
        # ! 参数修改！
        pro_current_list = list(filter(lambda x: x.c2[1]>threshold[0]*config['h'], block.Bboxes))
        if len(pro_current_list) != 0:
            pro_down_list    = list(filter(lambda x: x.c1[1]<threshold[1]*config['h'], down.Bboxes))
            
            pro_ld_list = None if left_down  == None else list(filter(lambda x: x.c1[1]<0.5*config['h'] and x.c2[1]>0.5*config['h'] and x.c1[0]>0.5*config['w'], left_down.Bboxes))
            pro_rd_list = None if right_down == None else list(filter(lambda x: x.c1[1]<0.5*config['h'] and x.c2[1]>0.5*config['h'] and x.c1[0]<0.5*config['w'], right_down.Bboxes))
            
            # left down:
            if type(pro_ld_list) == list and len(pro_ld_list) != 0:
                for candidate in pro_ld_list:
                    # relative position
                    p1 = np.zeros((1,2))
                    p2 = np.zeros((1,2))
                    p1[0, 0] = candidate.c1[0] - config['w'] // 2
                    p1[0, 1] = candidate.c1[1] + config['h'] // 2
                    p2[0, 0] = candidate.c2[0] - config['w'] // 2
                    p2[0, 1] = candidate.c2[1] + config['h'] // 2
                    
                    # del bad in current image
                    for to_del in pro_current_list:
                        if (to_del.c1[0] - p1[0, 0])**2 + (to_del.c1[1] - p1[0, 1])**2 < 10000:
                            for i in range(len(block.Bboxes)):
                                if i >= len(block.Bboxes) - 1:
                                    break
                                if (block.Bboxes[i].c1 == to_del.c1).all():
                                    block.Bboxes.pop(i)
                                    good_Bbox.append(candidate)
                                    break
                                    
                    # del bad in down image
                    for to_del in pro_down_list:
                        if (to_del.c2[0] - p2[0, 0])**2 + (to_del.c2[1] + config['h'] - p2[0, 1])**2 < 10000:
                            for i in range(len(down.Bboxes)):
                                if i >= len(down.Bboxes) - 1:
                                    break
                                if (down.Bboxes[i].c1 == to_del.c1).all():
                                    down.Bboxes.pop(i)
        
                            
                                
            # right down:
            if pro_rd_list is not None and len(pro_rd_list) != 0:
                for candidate in pro_rd_list:
                    # relative position
                    p1 = np.zeros((1,2))
                    p2 = np.zeros((1,2))
                    p1[0, 0] = candidate.c1[0] + config['w'] // 2
                    p1[0, 1] = candidate.c1[1] + config['h'] // 2
                    p2[0, 0] = candidate.c2[0] + config['w'] // 2
                    p2[0, 1] = candidate.c2[1] + config['h'] // 2
                    
                    # del bad in current image
                    for to_del in pro_current_list:
                        if (to_del.c1[0] - p1[0, 0])**2 + (to_del.c1[1] - p1[0, 1])**2 < 10000:
                            for i in range(len(block.Bboxes)):
                                if i >= len(block.Bboxes) - 1:
                                    break
                                if (block.Bboxes[i].c1 == to_del.c1).all():
                                    block.Bboxes.pop(i)
                                    good_Bbox.append(candidate)
                                    
                    # del bad in down image
                    for to_del in pro_down_list:
                        if (to_del.c2[0] - p2[0, 0])**2 + (to_del.c2[1] + config['h'] - p2[0, 1])**2 < 10000:
                            for i in range(len(down.Bboxes)):
                                if i >= len(down.Bboxes) - 1:
                                    break
                                if (down.Bboxes[i].c1 == to_del.c1).all():
                                    down.Bboxes.pop(i)
                            
    # right:
    if right is not None:
        pro_current_list = list(filter(lambda x: x.c2[0]>threshold[0]*config['w'], block.Bboxes))
        if pro_current_list is not None:
            pro_right_list   = list(filter(lambda x: x.c1[0]<threshold[1]*config['w'], right.Bboxes))
            # todo 已忽略双图冒险
            # 右上右下
            pro_ru_list = None if right_up   == None else list(filter(lambda x: x.c1[0]<0.5*config['w'] and x.c2[0]>0.5*config['w'] and x.c1[1]>0.5*config['h'], right_up.Bboxes))
            pro_rd_list = None if right_down == None else list(filter(lambda x: x.c1[0]<0.5*config['w'] and x.c2[0]>0.5*config['w'] and x.c1[1]<0.5*config['h'], right_down.Bboxes))
            
            # right up:
            if pro_ru_list is not None and len(pro_ru_list) != 0:
                for candidate in pro_ru_list:
                    # relative position
                    p1 = np.zeros((1,2))
                    p2 = np.zeros((1,2))
                    p1[0, 0] = candidate.c1[0] + config['w'] // 2
                    p1[0, 1] = candidate.c1[1] - config['h'] // 2
                    p2[0, 0] = candidate.c2[0] + config['w'] // 2
                    p2[0, 1] = candidate.c2[1] - config['h'] // 2
                    
                    # del bad in current image
                    for to_del in pro_current_list:
                        if (to_del.c1[0] - p1[0, 0])**2 + (to_del.c1[1] - p1[0, 1])**2 < 10000:
                            for i in range(len(block.Bboxes)):
                                if i >= len(block.Bboxes) - 1:
                                    break
                                if (block.Bboxes[i].c1 == to_del.c1).all():
                                    block.Bboxes.pop(i)
                                    good_Bbox.append(candidate)
                                    break
                                    
                    # del bad in right image
                    for to_del in pro_right_list:
                        if (to_del.c2[0] - p2[0, 0])**2 + (to_del.c2[1] + config['w'] - p2[0, 1])**2 < 10000:
                            for i in range(len(down.Bboxes)):
                                if i >= len(down.Bboxes) - 1:
                                    break
                                if (down.Bboxes[i].c1 == to_del.c1).all():
                                    down.Bboxes.pop(i)
                                
                                
            # right down:
            if pro_rd_list is not None and len(pro_rd_list) != 0:
                for candidate in pro_rd_list:
                    # relative position
                    p1 = np.zeros((1,2))
                    p2 = np.zeros((1,2))
                    p1[0, 0] = candidate.c1[0] + config['w'] // 2
                    p1[0, 1] = candidate.c1[1] + config['h'] // 2
                    p2[0, 0] = candidate.c2[0] + config['w'] // 2
                    p2[0, 1] = candidate.c2[1] + config['h'] // 2
                    
                    # del bad in current image
                    for to_del in pro_current_list:
                        if (to_del.c1[0] - p1[0, 0])**2 + (to_del.c1[1] - p1[0, 1])**2 < 10000:
                            for i in range(len(block.Bboxes)):
                                if i >= len(block.Bboxes) - 1:
                                    break
                                if (block.Bboxes[i].c1 == to_del.c1).all():
                                    block.Bboxes.pop(i)
                                    good_Bbox.append(candidate)
                                    
                    # del bad in down image
                    for to_del in pro_right_list:
                        if (to_del.c2[0] - p2[0, 0])**2 + (to_del.c2[1] + config['w'] - p2[0, 1])**2 < 10000:
                            for i in range(len(down.Bboxes)):
                                if i >= len(down.Bboxes) - 1:
                                    break
                                if (down.Bboxes[i].c1 == to_del.c1).all():
                                    down.Bboxes.pop(i)

            
def draw(img, c1, c2, label):
    colors = pkl.load(open("pallete", "rb"))
    color = random.choice(colors)
    cv2.rectangle(img, c1, c2,color, 5)

    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)

def read_data_pair(path1: str, path2:str, config_path:str) ->List[block_image]:
    with open(config_path, 'r') as f:
        config = json.load(f)
        f.close()
    temp_data = np.loadtxt(path1, dtype=int, delimiter=',')
    h, _ = temp_data.shape
    Bboxes1 = [Bbox(temp_data[i]) for i in range(h)]

    temp_data = np.loadtxt(path2, dtype=int, delimiter=',')
    h, _ = temp_data.shape
    Bboxes2 = [Bbox(temp_data[i]) for i in range(h)]

    blocks1 = [[None] * 10 for _ in range(10)]
    blocks2 = [[None] * 11 for _ in range(11)]

    block_bbox_list = []
    for i in Bboxes1:
        if block_bbox_list == []:
            block_bbox_list = [i]
        elif (block_bbox_list[0].Iposi == i.Iposi).all():
            block_bbox_list.append(i)
        else:
            blocks1[block_bbox_list[0].Iposi[0]][block_bbox_list[0].Iposi[1]] = block_image(block_bbox_list)
            block_bbox_list = []
            block_bbox_list.append(i)
    blocks1[9][9] = block_image(block_bbox_list)
            
    block_bbox_list = []
    for i in Bboxes2:
        if block_bbox_list == []:
            block_bbox_list = [i]
        elif (block_bbox_list[0].Iposi == i.Iposi).all():
            block_bbox_list.append(i)
        else:
            blocks2[block_bbox_list[0].Iposi[0]][block_bbox_list[0].Iposi[1]] = block_image(block_bbox_list)
            block_bbox_list = []
            block_bbox_list.append(i)
    blocks2[10][10] = block_image(block_bbox_list)
    
    return blocks1, blocks2
        
# 已将所有bbox对象存入block_img对象，再将对应block存入blocks列表矩阵中
# todo 完成run_process，将blocks列表矩阵中的对象逐个送入其中处理bbox

# %%

# // todo 目前good_Bbox未加入任何信息——截止到2_7图像，debug检查原因，检查运行过程
def main_pair():
    blocks1, blocks2 = read_data_pair('./results/final_result1.csv', './results/final_result2.csv', './results/config.json')
    good_Bbox = []
    threshold=(0.9,0.1)
    for i in range(10):
        for j in range(10):
            block   = blocks1[i][j]
            if block == None:
                continue
            up      = blocks1[i-1][j] if i>0 else None
            right   = blocks1[i][j+1] if j<9 else None
            down    = blocks1[i+1][j] if i<9 else None
            left    = blocks1[i][j-1] if j>0 else None
            lu = blocks2[i][j]
            ru = blocks2[i][j+1]
            rd = blocks2[i+1][j+1]
            ld = blocks2[i+1][j]
            run_process(threshold, good_Bbox, block, down, right, ru, rd, ld)


    Bbox_list = []
    for i in range(10):
        for j in range(10):
            if blocks1[i][j] is None:
                continue
            Bbox_list = Bbox_list + blocks1[i][j].Bboxes
            
    # 输出至文件
    with open('./results/processed.csv', 'w') as f:
        f.truncate()
        f.close()
    for bbox in Bbox_list:
        with open('./results/processed.csv', 'a') as f:
            np.savetxt(f, np.column_stack((bbox.c1[0], bbox.c1[1], bbox.c2[0], bbox.c2[1], bbox.label, 0, bbox.Iposi[0], bbox.Iposi[1])), fmt='%d', delimiter=',')
        f.close()


# %% 读取数据

def read_data_repeat(path1: str, path2: str, config_path:str) -> np.ndarray:
    
    with open('./results/config.json', 'r') as f:
        config = json.load(f)
        f.close()
    Bb1 = np.loadtxt('./results/processed.csv', dtype=int, delimiter=',')
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
    return Bb

# %% 去重复
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

def main_repeat():
    
    Bb = read_data_repeat('./results/processed.csv', './results/final_result2.csv', './results/config.json')
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
            elif area / rect1.area > 0.7:
                Bb[i] = np.zeros((1, 8))
                

    #  画框
    img = cv2.imread('./imgs/test.jpg')
    for i in range(h):
        if (Bb[i] == np.zeros((1, 8))).all():
            continue
        c1 = (Bb[i, 0], Bb[i, 1])
        c2 = (Bb[i, 2], Bb[i, 3])
        label = 'person' if Bb[i, 4] == 0 else 'car'
        draw(img, c1, c2, label)
    cv2.imwrite('./det/test.jpg', img)

# %% 
if __name__ == '__main__':
    main_pair()
    main_repeat()