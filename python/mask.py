import matlab
import matlab.engine               # import matlab引擎
import matlab
import matlab.engine
import cv2
import numpy as np
import os.path
import copy
import scipy.io

import matplotlib.pyplot as plt

img_path=r'/home/liuqg/wgj/diffu2/lzdata/wang0220/xueguantest60.png'
d = cv2.imread(img_path,0)
# 使用 OpenCV 读取图像文件（灰度模式）
d = d / 255.
# 对图像进行归一化处理，将像素值缩放到 0 到 1 之间。
img=d
img= img.astype(np.float32)
img=img.tolist()
# 将图像转换为 Python 列表格式。
img=matlab.double(img)
# 将图像数据转换为 MATLAB 可接受的类型 matlab.double。
engine = matlab.engine.start_matlab()  # 启动matlab engine
# 启动 MATLAB 引擎。
sensor_data111=engine.forward2(img)
# 调用 MATLAB 的 forward2 函数处理图像数据并返回处理后的结果 sensor_data111。
##
sensor_data111=np.array(sensor_data111)
# 将 MATLAB 返回的处理结果转换为 NumPy 数组。
plt.imshow(sensor_data111)
plt.savefig('aaa1.png')
plt.show()



mask=np.zeros((512,2075))
for i in range(64):
    mask[8*i,:]=1
# 创建一个 512x2075 的全零数组，并将每隔 8 行的数据设置为 1。
print(mask)

data=scipy.io.loadmat('512xueguansignal.mat')
yy=data['y']
yy=yy*mask
# 从 MAT 文件中加载数据，将数据与之前创建的 mask 数组相乘

plt.imshow(yy)
plt.savefig('aaa2.png')
plt.show()



aa=np.zeros((64,2075))
for j in range(64):
    aa[j,:]=yy[8*j,:]
# 创建一个 64x2075 的全零数组，并提取间隔为 8 的行数据。

plt.imshow(aa)
plt.savefig('aaa3.png')
plt.show()

#yy=yy.tolist()
#yy=matlab.double(yy)

aa=aa.tolist()
aa=matlab.double(aa)
# 调用 MATLAB 的 backward2 函数对处理后的数据进行逆处理。
recon=engine.backward2(aa)
recon=np.array(recon)
print(recon.shape)
print(recon)
recon=(recon-recon.min())/(recon.max()-recon.min())
# 将 MATLAB 返回的逆处理结果转换为 NumPy 数组，并进行归一化处理。
#cv2.imshow('image', recon)
cv2.imwrite('./wwresult/512xueguangsignalto64_2.png',255.*recon)
# 将逆处理后的图像保存为图像文件