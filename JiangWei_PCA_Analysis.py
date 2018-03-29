# -*- coding: utf-8 -*-
'''
降维技术：
  为何要降维：
      让数据集更加容易使用
      降低算法计算开销
      去除噪声
      使得结果易懂
降维算法：
    主成分分析PCA：坐标系转换
        PCA（n_components = None,  设置主成分数目，可以设置为0-1的数值，也可以设置为ml1
            copy = True,   设置在计算的时候是否使用输入的数据样本，输出为降维后的数据样本
            whiten = False）  将主成分向量值除以N倍的奇异值，True则移除部分信息
        PCA 属性：
        PCA方法：
    因子分析：隐变量
'''
#协方差矩阵
import numpy as np
X = [[2, 0, -1.4],
[2.2, 0.2, -1.5],
[2.4, 0.1, -1],
[1.9, 0, -1.2]]
print(np.cov(np.array(X).T))

#特征值与特征向量
w, v = np.linalg.eig(np.array([[1, -2], [2, -3]]))
print('特征值：{}\n特征向量：{}'.format(w,v))

#
a = [[-0.27, -0.3],
[1.23, 1.3],
[0.03, 0.4],
[-0.67, 0.6],
[-0.87, 0.6],
[0.63, 0.1],
[-0.67, -0.7],
[-0.87, -0.7],
[1.33, 1.3],
[0.13, -0.2]]
b = [[0.73251454], [0.68075138]]
np.dot(a,b)


#鸢尾花数据集的降维
%matplotlib inline
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

data = load_iris()
# 鸢尾花的种类
y = data.target
# 自变量
X = data.data
pca = PCA(n_components=2)
# 将PCA运行结果保存在reduced_X中
reduced_X = pca.fit_transform(X)
# 定义三种不同话的数组
red_x, red_y = [], []
blue_x, blue_y = [], []
green_x, green_y = [], []

for i in range(len(reduced_X)):
    if y[i] == 0:
        red_x.append(reduced_X[i][0])
        red_y.append(reduced_X[i][1])
    elif y[i] == 1:
        blue_x.append(reduced_X[i][0])
        blue_y.append(reduced_X[i][1])
    else:
        green_x.append(reduced_X[i][0])
        green_y.append(reduced_X[i][1])
# 用散点图展示分类后的数据
plt.scatter(red_x, red_y, c='r', marker='x')
plt.scatter(blue_x, blue_y, c='b', marker='D')
plt.scatter(green_x, green_y, c='g', marker='.')
plt.show()



