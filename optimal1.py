import cv2 as op
import numpy as np
import sympy as sp


im=op.imread("imageonline.jpg")
#用于处理的灰度目标图像
gray=im[:,:,0]
gray_x=gray.shape[0]
gray_y=gray.shape[1]
#全一矩阵
one=np.ones((gray_x,gray_y))
#符号变量
symbols_list = [sp.symbols(f'x_{i}_{j}') for i in range(1, gray_x+1) for j in range(1, gray_y+1)]


#定出一个边界扩充补0的p
p_temp=sp.Matrix(gray_x+2,gray_y+2,lambda i,j:0)
for i, var in enumerate(symbols_list):
    row = i // gray_y + 1
    col = i % gray_y + 1
    p_temp[row, col] = var
#进行离散拉普拉斯运算，将每个点的值赋给laplacian_p
laplacian_p=sp.Matrix(gray_x,gray_y,lambda i,j:0)

for i in range(0, gray_x):
    for j in range(0, gray_y):
            # 应用拉普拉斯算子的离散近似公式,p[i,j]相当于p_temp的p_temp[i+1,j+1]
            laplacian_p[i, j] = (p_temp[i+1, j] + p_temp[i, j+1] + p_temp[i+2, j+1] + p_temp[i+1, j+2] - 4*p_temp[i+1, j+1]) / 1

#积分操作
gray_sym=sp.Matrix(gray)
one_sym=sp.Matrix(one)

result_p=sp.Matrix(gray_x,gray_y,lambda i,j:0)
result_p=laplacian_p+gray_sym-one_sym
# 对结果矩阵的每个元素进行平方
result_p = result_p.applyfunc(lambda x: x**2)

print(result_p[0,0])