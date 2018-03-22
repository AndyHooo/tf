# _*_ coding: utf-8 _*_
'''
Created on 2018年2月9日
图像转置
@author: hudaqiang
'''
import matplotlib.image as mpimg  
import matplotlib.pyplot as plt  
import tensorflow as tf  
  
#加载图像  
filename = "imgs/happy.jpg"  
image = mpimg.imread(filename)  
  
#创建tensorflow变量  
x = tf.Variable(image,name='x')  
  
model = tf.initialize_all_variables()  
  
with tf.Session() as session:  
    x = tf.transpose(x, perm=[0,1,2])  
    session.run(model)  
    result = session.run(x)  
    
plt.imshow(result)  
plt.show()  

