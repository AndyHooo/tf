# _*_ coding: utf-8 _*_
'''
Created on 2018年3月21日
自编码器
@author: hudaqiang
'''
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import sklearn.preprocessing as prep

def xavier_init(fan_in,fan_out,constant = 1):
    """
    xavier权重初始化器实现均匀分布
    Args:
        fan_in:输入节点数量
        fan_out:输出节点数量
    Return:
        返回一个标准均匀分布
    """
    
    low = - constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    
    return tf.random_uniform((fan_in,fan_out), minval = low, maxval = high, dtype = tf.float32)

def standard_scale(X_train,X_test):
    """
    将训练集和测试集进行标准化
    """
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train,X_test

def get_random_block_from_data(data,batch_size):
    start_index = np.random.randint(0,len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]

class AdditiveGaussianNoiseAutoEncoder(object):
    """
    加性高斯噪音自编码
    """
    def __init__(self,n_input,n_hidden,transfer_function = tf.nn.softplus,optimizer = tf.train.AdamOptimizer(),scale = 0.1):
        """
        构造器
        Args:
            n_input:输入变量数
            n_hidden:隐含层节点数量
            transfer_function:隐含层激活函数
            optimizer:优化器
            scale:高斯噪声系数
        """
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        network_weights = self._initialize_weights()
        self.weights = network_weights
        self.x = tf.placeholder(tf.float32, [None,self.n_input])
        self.hidden = self.transfer(tf.add(tf.matmul(self.x + scale * tf.random_normal((n_input,)),self.weights['w1']),self.weights['b1']))
        self.reconstruction = tf.add(tf.matmul(self.hidden,self.weights['w2']),self.weights['b2'])
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x),2.0))
        self.optimizer = optimizer.minimize(self.cost)
        init = tf.global_variables_initializer();
        self.sess = tf.Session()
        self.sess.run(init)
        
    def _initialize_weights(self):
        all_weight = {}
        all_weight['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden))
        all_weight['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype = tf.float32))
        all_weight['w2'] = tf.Variable(tf.zeros([self.n_hidden,self.n_input], dtype = tf.float32))
        all_weight['b2'] = tf.Variable(tf.zeros([self.n_input], dtype = tf.float32))
        return all_weight
    
    def partial_fit(self,X):
        cost,opt = self.sess.run((self.cost,self.optimizer),feed_dict = {self.x: X,self.scale: self.training_scale})
        return cost
    
    def calc_total_cost(self,X):
        return self.sess.run(self.cost,feed_dict = {self.x : X,self.scale:self.training_scale})
    
    def transform(self,X):
        return self.sess.run(self.hidden,feed_dict = {self.x : X,self.scale: self.training_scale})
    
    def generate(self,hidden = None):
        hidden = np.random.normal(size = self.weights['b1'])
        return self.sess.run(self.reconstruction,feed_dict={self.hidden:hidden})
    
    def reconstruct(self,X):
        """
        获取隐含层的权重
        """
        return self.sess.run(self.reconstruction,feed_dict = {self.x:X,self.scale:self.training_scale})
    
    def get_weights(self):
        return self.sess.run(self.weights['w1'])
    
    def get_biases(self):
        """
        获取隐含层的偏置系数
        """
        return self.sess.run(self.weights['b1'])
    
#读取数据
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)
#对数据进行标准化处理(标准化:让数据变成0均值，标准差为1的分布)
X_train,X_test = standard_scale(mnist.train.images, mnist.test.images)
n_samples = int(mnist.train._num_examples) #样本总数
training_epochs = 20
batch_size = 128
display_step = 1

autoencoder = AdditiveGaussianNoiseAutoEncoder(n_input = 784,n_hidden = 200,transfer_function=tf.nn.softplus,optimizer=tf.train.AdamOptimizer(learning_rate=0.001),scale=0.01)
for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(n_samples / batch_size)
    for i in range(total_batch):
        batch_xs = get_random_block_from_data(X_train, batch_size)
        cost = autoencoder.partial_fit(batch_xs)
        avg_cost += cost / n_samples * batch_size
    
    if epoch % display_step == 0:
        print("Epoch:",'%04d' % (epoch + 1),"cost=","{:.9f}".format(avg_cost))
        
print("Total cost:" + str(autoencoder.calc_total_cost(X_test)))

#######
#tf.random_normal:从正态分布输出随机值
#tf.random_uniform:从均匀分布中输出随机值
#tf.truncated_normal:截断的正态分布函数，生成的值遵循一个正态分布，但不会大于平均值2个标准差

#隐层的激活函数一般使用ReLU
#输出层一般还是使用sigmoid，因为它最接近概率输出分布
#######

print(xavier_init(768, 200, 1))
    
    
    