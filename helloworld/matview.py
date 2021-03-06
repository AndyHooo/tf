# _*_ coding: utf-8 _*_
'''
Created on Jan 24, 2018

@author: hudaqiang
'''
import matplotlib.pyplot as plt
import numpy as np

#===============================================================================
# 画线
#===============================================================================
# x = np.linspace(-np.pi,np.pi,256,endpoint = True)
# y = np.sin(x)
# y_ = np.zeros_like(x)
# x_ = np.zeros_like(y)
# plt.plot(x,y)
# plt.plot(x,y_)
# plt.plot(x_,y)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('y=sin(x)')
# plt.legend()
# plt.show()

#===============================================================================
# 条形图
#===============================================================================
# plt.bar([1,4,7,10,13],[32,56,10,34,59],label = 'group one',color = 'g')
# plt.bar([2,5,8,11,14],[31,49,10,34,59],label = 'group two',color = 'r')
# plt.bar([3,6,9,12,15],[32,46,20,24,59],label = 'group three',color = 'b')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('bar')
# plt.legend()
# plt.show()

#===============================================================================
# 直方图
#===============================================================================
# population_ages = [22,55,62,45,21,22,34,42,42,4,99,102,110,120,121,122,130,111,115,112,80,75,65,54,44,43,42,48]
# bins = [0,10,20,30,40,50,60,70,80,90,100,110,120,130]
# plt.hist(population_ages, bins, histtype='bar', rwidth=0.8)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title(u'年龄分布')
# plt.legend()
# plt.show()

#===============================================================================
# 散点图
#===============================================================================
# x = [1,2,3,4,5,6,7,8]
# y = [5,2,4,2,1,4,5,2]
# plt.scatter(x,y, label='skitscat', color='r', s=25, marker=u'.')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title(u'散点图')
# plt.legend()
# plt.show()

#===============================================================================
# 堆叠图
#===============================================================================
# days = [1,2,3,4,5]
# sleeping = [7,8,6,11,7]
# eating =   [2,3,4,3,2]
# working =  [7,8,7,2,2]
# playing =  [8,5,7,8,13]
# 
# plt.plot([],[],color='m', label='Sleeping', linewidth=5)
# plt.plot([],[],color='c', label='Eating', linewidth=5)
# plt.plot([],[],color='r', label='Working', linewidth=5)
# plt.plot([],[],color='k', label='Playing', linewidth=5)
# 
# plt.stackplot(days, sleeping,eating,working,playing, colors=['m','c','r','k'])
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title(u'堆叠图')
# plt.legend()
# plt.show()


#===============================================================================
# 饼图
#===============================================================================
# slices = [7,2,2,13]
# activities = ['sleeping','eating','working','playing']
# cols = ['c','m','r','b']
# 
# plt.pie(slices,
#         labels=activities,
#         colors=cols,
#         startangle=90,
#         shadow= True,
#         explode=(0,0.1,0,0),
#         autopct='%1.1f%%')
# 
# plt.title(u'饼图')
# plt.show()

#===============================================================================
# 世界地图
#===============================================================================




