# _*_ coding: utf-8 _*_
'''
Created on Jan 22, 2018

@author: hudaqiang
'''
import  tensorflow as tf
#构建计算图
state = tf.Variable(0,name = "counter")
one = tf.constant(1)
new_state = tf.add(state, one, name = "new_state")
update = tf.assign(state,new_state)

#将变量初始化到计算图中
init_op = tf.initialize_all_variables()

#执行计算图
with tf.Session() as sess:
    sess.run(init_op)
    
    print sess.run(state)
    for _ in range(5):
        sess.run(update)
        print sess.run(state)
