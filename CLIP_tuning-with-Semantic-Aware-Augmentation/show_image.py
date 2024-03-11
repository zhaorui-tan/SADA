import numpy as np
import cv2
import os

#这里
root = '../clip_finetune/shared_space/ccnl/mm_data/Arch-Data-1114/arch_data' # 图片存的地址
dist_root = 'zerooutput'  # 输出截过图的地址
result_dist = 'vis_result.npy'       # 之前保存好的测试结果的地址
test_file_dist = '../clip_finetune/shared_space/ccnl/mm_data/Arch-Data-1114/test/test_1114_new_5test.txt' # 测试集数据描述文件

# 读取结果文件和对应的测试集数据描述文件
a = np.load(result_dist, allow_pickle=True).item()
with open(test_file_dist, 'r') as f:
    lines = f.readlines()
    file2text = {}
    for line in lines:
        file = line.split('#')[0]
        name = line.split(' ')[-1].strip()
        file2text[file] = name

# 将图片命转化为词
b = {file2text[k]:a[k] for k in a}

# 通过词对应的图片名们，生成对应的图片，并保存在制定文件夹内
for k in b:
    img_files = b[k]
    imglist = []
    dist_file = f'{dist_root}/{k}.jpg'

    for img_file in img_files:
        ifile = os.path.join(root, img_file)
        img = cv2.imread(ifile)
        img = cv2.resize(img, (256,256))
        imglist.append(img)
    h_imgs = cv2.hconcat(imglist)
    
    cv2.imwrite(dist_file, h_imgs)