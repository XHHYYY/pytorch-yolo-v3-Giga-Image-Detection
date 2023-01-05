from test_detector import detector

flag = 2

det_path = './det/{}/'.format(flag)
with open('./results/final_result{}.csv'.format(flag), 'w') as f:
    f.truncate()
    f.close()
Detect = detector(det_path, confidence=0.8)
for i in range(9 + flag):
    for j in range(9 + flag):
        Detect.read_images('./cutted/{0}/{1}_{2}_{3}.jpg'.format(flag, flag, i, j))
        Detect.detect_objects()
        Detect.output_result()




# todo 将所有bbox坐标转化为全局坐标，再 1、根据IOU去重 2、合并同一物体不同部位的bbox


