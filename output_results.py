import json
import numpy as np

# 读取数据
Bboxes = np.loadtxt('./results/processed.csv', dtype=float, delimiter=',')
c1 = [Bboxes[:, 0], Bboxes[:, 1]]
c2 = [Bboxes[:, 2], Bboxes[:, 3]]
label = Bboxes[:, 4]
Iposi = [Bboxes[:, 6], Bboxes[:, 7]]

# 读取参数
with open('./results/config.json') as f:
    config = json.load(f)
    f.close()
w = config['w']
h = config['h']

# 生成结果列表
h, _ = Bboxes.shape
results = []
for i in range(h):
    width  = c2[0][i] - c1[0][i]
    height = c2[1][i] - c1[1][i]
    x = c1[0][i] + Iposi[0][i] * w
    y = c1[1][i] + Iposi[1][i] * h
    toFile = list(map(lambda x: int(x), [x, y, width, height]))
    results.append({'bbox': toFile})
    
# 输出至文件
with open('./det_results.json', 'w') as f:
    json.dump(results, f, indent=4)
    f.close()