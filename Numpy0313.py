# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 08:49:22 2018

@author: hasee_yang
"""

# 通用函数

from __future__ import division
from numpy.random import randn
import numpy as np

arr = np.arange(10)
arr1 = np.sqrt(arr)
arr2 = np.exp(arr)

x = randn(8)
y = randn(8)

print(arr)
print("........")
print(x)
print("........")
print(y)
print(arr2)
