# _*_ coding: utf-8 _*_
'''
Created on Jan 24, 2018
numpy相关属性
@author: hudaqiang
'''

import numpy as np
from numpy import arange
arr = np.array([1,23,45,56])
print arr

ndarr = arange(24).reshape(2,3,4)
print ndarr

print ndarr.dtype #numpy中的元素类型 
print ndarr.shape #数组的形状
print ndarr.ndim  #数组的秩
print ndarr.size #数组中的元素个数

