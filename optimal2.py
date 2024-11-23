import cv2 as op
import numpy as np
from scipy import ndimage

im=op.imread("imageonline.jpg")
#用于处理的灰度目标图像
gray=im[:,:,0]
gray_x=gray.shape[0]
gray_y=gray.shape[1]
#全一矩阵
one=np.ones((gray_x,gray_y))

#最小二乘操作解出p表达式
b=gray-one
