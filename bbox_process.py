import numpy as np
import cv2
import json
from typing import List
import pickle as pkl
import random

class check_img_Exception(Exception):
    '''
    报错类
    '''
    def __init__(self, *args: object) -> None:
        super().__init__(*args)
        self.name = args[0]

class Bbox():
    '''
    Bbox类，储存bbox信息
    '''
    def __init__(self, data: np.ndarray) -> None:
        '''
        - 输入：1*8矩阵
        '''
        self.c1 = data[0:2] # 左上角坐标
        self.c2 = data[2:4] # 右下角坐标
        self.label = data[4] # label
        self.Iposi = data[-2:] # 所在图片的图片坐标
        self.area = (self.c2[0] - self.c1[0]) * (self.c2[1] - self.c1[1]) # 面积

class block_image():
    '''
    图片类，储存切分后的图片的信息
    '''
    def __init__(self, bbox: List[Bbox]) -> None:
        self.Bboxes = bbox # 该图片包含的bbox列表
        # 若包含的bbox列表中有bbox的图片坐标与其他的不一致则报错
        if len(self.Bboxes) != len(list(filter(lambda x: (x.Iposi == self.Bboxes[0].Iposi).all(), self.Bboxes))):
            raise check_img_Exception('Contain Bbox from other image!')
        self.position = self.Bboxes[0].Iposi # 该图片的图片坐标

# 去重复
class rect():
    '''
    去重复算法中的矩形类，与bbox类类似，但包含该bbox四个点的坐标及该bbox的面积
    '''
    def __init__(self, c1, c2) -> None:
        self.c1 = c1
        self.c2 = c2
        # 该bbox四个点的坐标
        self.points = [c1, (c1[0], c2[1]), c2, (c2[0], c1[1])] # 左上起逆时针
        self.area = (c1[0] - c2[0]) * (c1[1] - c2[1])

def run_process(threshold: tuple, good_Bbox: block_image, block: block_image, down: block_image,right: block_image,
            right_up: block_image, right_down: block_image, left_down: block_image) -> None:
    '''
    # 功能说明
    使用bbox匹配算法处理割裂bbox
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

    # 读取图片配置
    with open('./results/config.json', 'r') as f:
        config = json.load(f)
        f.close()

    # down:删除位于本图和下图的割裂bbox
    if down is not None:
        
        # 本图bbox列表
        pro_current_list = list(filter(lambda x: x.c2[1]>threshold[0]*config['h'], block.Bboxes))
        if len(pro_current_list) != 0:
            # 下图bbox列表
            pro_down_list    = list(filter(lambda x: x.c1[1]<threshold[1]*config['h'], down.Bboxes))
            # 第二组左下图及右下图bbox列表
            pro_ld_list = None if left_down  == None else list(filter(lambda x: x.c1[1]<0.5*config['h'] and x.c2[1]>0.5*config['h'] and x.c1[0]>0.5*config['w'], left_down.Bboxes))
            pro_rd_list = None if right_down == None else list(filter(lambda x: x.c1[1]<0.5*config['h'] and x.c2[1]>0.5*config['h'] and x.c1[0]<0.5*config['w'], right_down.Bboxes))
            
            # 使用第二组左下图处理割裂bbox
            if type(pro_ld_list) == list and len(pro_ld_list) != 0:
                for candidate in pro_ld_list:
                    # 将左下图bbox坐标变换至本图坐标
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
        
                            
                                
            # 流程同上，使用右下图bbox处理本图右下半部分割裂bbox
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
                            
    # 流程同上，处理位于本图和右图边缘的割裂bbox
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
    '''
    对输入图像画框
    '''
    colors = pkl.load(open("pallete", "rb"))
    color = random.choice(colors)
    cv2.rectangle(img, c1, c2,color, 5)

    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)

def read_data_pair(path1: str, path2:str, config_path:str) -> List[block_image]:
    '''
    为匹配bbox算法读取bbox数据
    ## 输入
    - path1：第一组bbox结果路径
    - path2：第二组bbox结果路径
    - config_path：切分图像参数路径
    ## 输出：
    - blocks1：图像1数据
    - blocks2：图像2数据
    ## 说明：
    - path1、path2存储的bbox结果为n*8的矩阵，其中1~8列分别为：
        1. c1[0]
        2. c1[1]
        3. c2[0]
        4. c2[1]
        5. label
        6. 0（占位符）
        7. 图像坐标[0]
        8. 图像坐标[1]
    - 输出blocks的数据结构为：
        - blocks_image对象的二维列表，对象在二维列表中的位置与该对象对应图像在原图中的位置相对应
        - blocks_image对象包含该对象对应切分图像的所有bbox，即bbox列表
    '''
    # 读取切分图片参数
    with open(config_path, 'r') as f:
        config = json.load(f)
        f.close()
    # 读取第一组图像的所有bbox
    temp_data = np.loadtxt(path1, dtype=int, delimiter=',')
    h, _ = temp_data.shape
    Bboxes1 = [Bbox(temp_data[i]) for i in range(h)]

    # 读取第二组图像的所有bbox
    temp_data = np.loadtxt(path2, dtype=int, delimiter=',')
    h, _ = temp_data.shape
    Bboxes2 = [Bbox(temp_data[i]) for i in range(h)]

    blocks1 = [[None] * 10 for _ in range(10)]
    blocks2 = [[None] * 11 for _ in range(11)]

    # 将第一组图像的所有bbox按照图片位置存入block_image对象，并将该对象存入blocks列表的对应位置
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
            
    # 将第二组图像的所有bbox按照图片位置存入block_image对象，并将该对象存入blocks列表的对应位置
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

def main_pair():
    '''
    bbox匹配算法处理割裂bbox的主函数
    '''
    # 读取所有bbox
    blocks1, blocks2 = read_data_pair('./results/final_result1.csv', './results/final_result2.csv', './results/config.json')
    good_Bbox = [] # 割裂bbox对应的在第二组图中的完整bbox列表
    threshold=(0.9,0.1) # 即若bbox对应点位于图像的90%以外则认为该bbox为割裂bbox
    # 对每一个图像处理bbox
    for i in range(10):
        for j in range(10):
            block   = blocks1[i][j]
            if block == None:
                continue
            up      = blocks1[i-1][j] if i>0 else None
            right   = blocks1[i][j+1] if j<9 else None
            down    = blocks1[i+1][j] if i<9 else None
            left    = blocks1[i][j-1] if j>0 else None
            lu      = blocks2[i][j]
            ru      = blocks2[i][j+1]
            rd      = blocks2[i+1][j+1]
            ld      = blocks2[i+1][j]
            run_process(threshold, good_Bbox, block, down, right, ru, rd, ld)


    # 将处理后的第一组图所有bbox收集起来
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

def read_data_repeat(path1: str, path2: str, config_path:str) -> np.ndarray:
    '''
    为去重复算法读取bbox
    读取数据为bbox匹配算法处理后的所有bbox
    '''
    # 读取切分图像参数
    with open('./results/config.json', 'r') as f:
        config = json.load(f)
        f.close()
    # 匹配算法处理后的bbox
    Bb1 = np.loadtxt('./results/processed.csv', dtype=int, delimiter=',')
    h, _ = Bb1.shape
    # 将块内坐标变换为图像1全局坐标
    Bb1[:, 0] = Bb1[:, 0] + Bb1[:, -1] * config['w'] 
    Bb1[:, 2] = Bb1[:, 2] + Bb1[:, -1] * config['w'] 
    Bb1[:, 1] = Bb1[:, 1] + Bb1[:, -2] * config['h'] 
    Bb1[:, 3] = Bb1[:, 3] + Bb1[:, -2] * config['h'] 
    # 读取第二组图（辅助图像）的所有bbox
    Bb2 = np.loadtxt('./results/final_result2.csv', dtype=int, delimiter=',')
    h, _ = Bb2.shape
    # 将跨内坐标变换为图像1全局坐标
    Bb2[:, 0] = Bb2[:, 0] + Bb2[:, -1] * config['w'] - config['w'] // 2 
    Bb2[:, 2] = Bb2[:, 2] + Bb2[:, -1] * config['w'] - config['w'] // 2 
    Bb2[:, 1] = Bb2[:, 1] + Bb2[:, -2] * config['h'] - config['h'] // 2 
    Bb2[:, 3] = Bb2[:, 3] + Bb2[:, -2] * config['h'] - config['h'] // 2 
    # 合并为整个列表
    Bb = np.concatenate((Bb1, Bb2), axis=0)
    return Bb

def compute_iou(rec1: tuple, rec2: tuple) -> int:
    '''
    计算输入的两个矩形的交集面积
    - 输入：两个矩形的四点坐标
    - 输出：交叠面积
    '''
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
    '''
    去重复算法的处理主函数
    ## 主要思想
        对任意两个bbox，计算交叠面积，若该面积占一个bbox面积的70%以上则删除该bbox
    '''
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

if __name__ == '__main__':
    main_pair()
    main_repeat()