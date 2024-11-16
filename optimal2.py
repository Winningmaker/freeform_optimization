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

# 将灰度图像缩放到50x50的大小
resized_gray = cv2.resize(gray, (50, 50), interpolation=cv2.INTER_AREA)

# 显示图像
cv2.imshow('Image', resized_gray)

# 等待按键事件
cv2.waitKey(0)

# 关闭所有窗口
cv2.destroyAllWindows()