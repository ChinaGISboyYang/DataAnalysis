# coding=utf-8

import numpy as np


def numpysum(n):
    a = np.arange(n) ** 2
    b = np.arange(n) ** 3

    c = a + b
    return c
import sys
from datetime import datetime
import numpy as np

size = 1000

start = datetime.now()
c = numpysum(n)
delate = datetime.now() - start
print "The last 2 elements of the sum", c[-2:]

# numpy 数组
a = arange()
a.dtype()


numpysum(3)