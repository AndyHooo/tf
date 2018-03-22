# _*_ coding: utf-8 _*_
'''
Created on 2018年3月21日
多层感知机
@author: hudaqiang
'''
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

sess = tf.Session()
#构建数据流图
in_units = 784
hidden_units = 300
keep_prob = tf.placeholder(tf.float32) #保留比率

w1 = tf.Variable(tf.truncated_normal([in_units,hidden_units],  stddev = 0.1, dtype = tf.float32))
b1 = tf.Variable(tf.zeros([hidden_units]))

w2 = tf.Variable(tf.zeros([hidden_units,10]))
b2 = tf.Variable(tf.zeros([10]))

x = tf.placeholder(tf.float32, [None,784])
hidden_layer_out = tf.nn.relu(tf.add(tf.matmul(x,w1),b1))
hidden_layer_drop = tf.nn.dropout(hidden_layer_out, keep_prob)
y = tf.nn.softmax(tf.add(tf.matmul(hidden_layer_drop,w2),b2))

#定义损失函数和优化器
y_ = tf.placeholder(tf.float32, [None,10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices = [1]))
train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess.run(init)

#训练
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)
for i in range(3000):
    batch_x,batch_y = mnist.train.next_batch(100)
    sess.run(train_step,{x:batch_x,y_:batch_y,keep_prob:0.75})
    
#测试
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print(sess.run(accuracy,{x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0}))

