from test_detector import detector

for i in range(2):
    flag = i + 1
    det_path = './det/{}/'.format(flag)
    # 清空bbox结果文件
    with open('./results/final_result{}.csv'.format(flag), 'w') as f:
        f.truncate()
        f.close()
    # 开始检测
    Detect = detector(det_path, confidence=0.8)
    for i in range(9 + flag):
        for j in range(9 + flag):
            Detect.read_images('./cutted/{0}/{1}_{2}_{3}.jpg'.format(flag, flag, i, j))
            Detect.detect_objects()
            Detect.output_result()
