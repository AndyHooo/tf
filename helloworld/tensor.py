# _*_ coding: utf-8 _*_
'''
Created on Jan 20, 2018
@author: hudaqiang
'''
import tensorflow as tf

#计算图:一系列的计算操作抽象为图中的节点
#节点：有0个或者多个tensor输入,一个tensor输出
#常量节点：如同tensorflow中的常量,0个输入一个输出
node1 = tf.constant(3.0,tf.float32)
node2 = tf.constant(4.0)
print(node1,node2)

#会话：评估一个节点，必须在一个会话Session中运行计算图，会话封装了Tensorflow运行时的状态和控制
sess = tf.Session()
print(sess.run([node1,node2]))

node3 = tf.add(node1, node2)
print node3
print(sess.run(node3))

#placeholder:计算图可以使用占位符placeholder参数化的从外部输入数据，placeholder的作用是在稍后提供一个值
#构造计算图
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
add_node = a + b
#运行计算图
print(sess.run(add_node,{a:1,b:2}))
print(sess.run(add_node,{a:[1,2],b:[3,4]}))

#变量Variable
#构造一个变量，需要提供类型和初始值：
W=tf.Variable([.3],tf.float32)  
b=tf.Variable([-.3],tf.float32)  
x=tf.placeholder(tf.float32)  
linear_model=W*x+b  

#常量节点在调用tf.constant时就被初始化，而变量在调用tf.Variable时并不初始化，必须显性的执行如下操作：
init = tf.global_variables_initializer()  
sess.run(init)
#意识到init对象是Tensorflow子图初始化所有全局变量的句柄是重要的，在调用sess.run(init)方法之前，所有变量都是未初始化的。


