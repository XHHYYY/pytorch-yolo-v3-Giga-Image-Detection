# %%
import numpy as np
import cv2
import json
from typing import List

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


class block_image():
    def __init__(self, bbox: List[Bbox]) -> None:
        self.Bboxes = bbox
        if len(self.Bboxes) != len(list(filter(lambda x: (x.Iposi == self.Bboxes[0].Iposi).all(), self.Bboxes))):
            raise check_img_Exception('Contain Bbox from other image!')
        self.position = self.Bboxes[0].Iposi
    
    

# %%
def run_process(good_Bbox: block_image, block: block_image, down: block_image,right: block_image,
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
    # 2675 * 1505
    
    with open('./results/config.json', 'r') as f:
        config = json.load(f)
    
    # down:
    if down is not None:
        # ! 参数修改！
        pro_current_list = list(filter(lambda x: x.c2[1]>0.99*config['h'], block.Bboxes))
        if pro_current_list is not None:
            pro_down_list    = list(filter(lambda x: x.c1[1]<0.01*config['h'], down.Bboxes))
            
            pro_ld_list = None if left_down  == None else list(filter(lambda x: x.c1[1]<0.5*config['h'] and x.c2[1]>0.5*config['h'] and x.c1[0]>0.5*config['w'], left_down.Bboxes))
            pro_rd_list = None if right_down == None else list(filter(lambda x: x.c1[1]<0.5*config['h'] and x.c2[1]>0.5*config['h'] and x.c1[0]<0.5*config['w'], right_down.Bboxes))
            
            # left down:
            if pro_ld_list is not None and len(pro_ld_list) != 0:
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
                                if (block.Bboxes[i].c1 == to_del.c1).all():
                                    block.Bboxes.pop(i)
                                    good_Bbox.append(candidate)
                                    break
                                    
                    # del bad in down image
                    for to_del in pro_down_list:
                        if (to_del.c2[0] - p2[0, 0])**2 + (to_del.c2[1] + config['h'] - p2[0, 1])**2 < 10000:
                            for i in range(len(down.Bboxes)):
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
                                if i == len(down.Bboxes) - 1:
                                    break
                                if (block.Bboxes[i].c1 == to_del.c1).all():
                                    block.Bboxes.pop(i)
                                    good_Bbox.append(candidate)
                                    
                    # del bad in down image
                    for to_del in pro_down_list:
                        if (to_del.c2[0] - p2[0, 0])**2 + (to_del.c2[1] + config['h'] - p2[0, 1])**2 < 10000:
                            for i in range(len(down.Bboxes)):
                                if i == len(down.Bboxes) - 1:
                                    break
                                if (down.Bboxes[i].c1 == to_del.c1).all():
                                    down.Bboxes.pop(i)
                            
    # right:
    if right is not None:
        pro_current_list = list(filter(lambda x: x.c2[0]>0.9*config['w'], block.Bboxes))
        if pro_current_list is not None:
            pro_right_list   = list(filter(lambda x: x.c1[0]<0.1*config['w'], right.Bboxes))
            # todo 已忽略双图冒险
            # 右上右下
            pro_ru_list = None if right_up   == None else list(filter(lambda x: x.c1[0]<0.5*config['w'] and x.c2[0]>0.5*config['w'] and x.c1[1]>0.5*config['h'], right_up.Bboxes))
            pro_rd_list = None if right_down == None else list(filter(lambda x: x.c1[0]<0.5*config['w'] and x.c2[0]>0.5*config['w'] and x.c1[1]>0.5*config['h'], right_down.Bboxes))
            
            # left down:
            if pro_ru_list is not None:
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
                                if i == len(down.Bboxes) - 1:
                                    break
                                if (block.Bboxes[i].c1 == to_del.c1).all():
                                    block.Bboxes.pop(i)
                                    good_Bbox.append(candidate)
                                    break
                                    
                    # del bad in down image
                    for to_del in pro_right_list:
                        if (to_del.c2[0] - p2[0, 0])**2 + (to_del.c2[1] + config['w'] - p2[0, 1])**2 < 10000:
                            for i in range(len(down.Bboxes)):
                                if i == len(down.Bboxes) - 1:
                                    break
                                if (down.Bboxes[i].c1 == to_del.c1).all():
                                    down.Bboxes.pop(i)
                                
                                
            # right down:
            if pro_rd_list is not None:
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
                                if i == len(down.Bboxes) - 1:
                                    break
                                if (block.Bboxes[i].c1 == to_del.c1).all():
                                    block.Bboxes.pop(i)
                                    good_Bbox.append(candidate)
                                    
                    # del bad in down image
                    for to_del in pro_right_list:
                        if (to_del.c2[0] - p2[0, 0])**2 + (to_del.c2[1] + config['w'] - p2[0, 1])**2 < 10000:
                            for i in range(len(down.Bboxes)):
                                if i == len(down.Bboxes) - 1:
                                    break
                                if (down.Bboxes[i].c1 == to_del.c1).all():
                                    down.Bboxes.pop(i)



                # todo 实现返回所有列表（可行性？）
                # todo 实现距离判断与删除相应Bbox


# %%
with open('./results/config.json', 'r') as f:
    config = json.load(f)

temp_data = np.loadtxt('./results/final_result1.csv', dtype=int, delimiter=',')
h, _ = temp_data.shape
Bboxes1 = [Bbox(temp_data[i]) for i in range(h)]

temp_data = np.loadtxt('./results/final_result2.csv', dtype=int, delimiter=',')
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
        
block_bbox_list = []
for i in Bboxes2:
    if block_bbox_list == []:
        block_bbox_list = [i]
    elif (block_bbox_list[0].Iposi == i.Iposi).all():
        block_bbox_list.append(i)
    else:
        blocks2[block_bbox_list[0].Iposi[0]][block_bbox_list[0].Iposi[1]] = block_image(block_bbox_list)
        block_bbox_list = []
        
# 已将所有bbox对象存入block_img对象，再将对应block存入blocks列表矩阵中
# todo 完成run_process，将blocks列表矩阵中的对象逐个送入其中处理bbox

# %%
# todo 目前good_Bbox未加入任何信息——截止到2_7图像，debug检查原因，检查运行过程
good_Bbox = []
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
        run_process(good_Bbox, block, down, right, ru, rd, ld)


# %% [markdown]
# ## PlayGround

# %%
(1,2) == (1,3)

# %%
# 相对坐标转绝对坐标：
# temp_data[:, 0] = temp_data[:, 0] + temp_data[:, -1] * config['w'] - config['w']/2
# temp_data[:, 2] = temp_data[:, 2] + temp_data[:, -1] * config['w'] - config['w']/2
# temp_data[:, 1] = temp_data[:, 1] + temp_data[:, -2] * config['h'] - config['h']/2
# temp_data[:, 3] = temp_data[:, 3] + temp_data[:, -2] * config['h'] - config['h']/2

# %%
class check_img(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)
        self.name = args[0]
        
raise check_img('Wrong!')

# %%
class Bboxes():
    def __init__(self, path: str) -> None:
        
        with open('./results/config.json', 'r') as file:
            self.config = json.load(file) # 切割数a, b, 原图大小H, W, 切割后大小h, w
        file.close()
        
        temp = np.loadtxt(path, dtype=int, delimiter=',')
        self.c1 = temp[:, 0:2]
        self.c2 = temp[:, 2:4]
        self.label = temp[:, 4]
        self.position = temp[:, -2:-1]
        
    

# %%
class test():
    def __init__(self, b) -> None:
        self.b = b
        
def run_test(a: test):
    a.b = 10
    
c = test(20)
run_test(c)
print(c.b)

# %%
from typing import List
vector = List[Bbox]
def test(temp: vector):
    pass


