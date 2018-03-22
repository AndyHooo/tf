# _*_ coding: utf-8 _*_
'''
Created on 2018年2月5日
数据流图
@author: hudaqiang
'''
import tensorflow as tf
cst1 = tf.constant(1,tf.float32)

with tf.Graph().as_default():
    tf.constant(2,tf.float32)
    sess = tf.Session()
