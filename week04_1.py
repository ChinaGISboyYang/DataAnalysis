# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
# 将条件逻辑表达为数组运算

x = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
y = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
z = np.array([True, False, True, True, False])

result = [
    (x if z else y)
    for x, y, z in zip(x, y, z)
]

# where 函数可以快速更改数组

# 数学和统计方法
# 方法和说明：
#            sum            对数组中全部或者轴向的元素求和
#            mean           算术平均数，零长度的数组mean为NaN
#            std, var       标准差和方差，自由度可调
#            min, max       最大值和最小值
#            argmin, argmax 最大和最小元素的索引
#            cumsum         所有元素的累计和
#            cumprod        所有元素的累计积
arr = np.random.randn(5, 4)
arr.mean()
np.
