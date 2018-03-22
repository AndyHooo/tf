# _*_ coding: utf-8 _*_
'''
Created on 2018年3月22日
tf实现word2vec
@author: hudaqiang
'''
import collections
import os
import urllib
import zipfile

import tensorflow as tf
import numpy as np
from scipy import random

data_index = 0
url = 'http://mattmahoney.net/dc'
def maybe_download(filename,expected_bytes):
    if not os.path.exists(filename):
        filename,_ = urllib.urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified',filename)
    else:
        print('file download failed!')
        return None
    return filename  

def read_data(filename): 
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

def build_dataset(words):    
    count = [['UNK',-1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    
    dictionary = dict()
    for word,_ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(),dictionary.keys()))
    return data,count,dictionary,reverse_dictionary

def generate_batch(batch_size, num_skips, skip_window):
    """
    生成训练样本
    args:
        batch_size: 每一批训练样本的数量
        num_skips: 每个单词生成多少个样本，不能大于skip_window的两倍，batch_size必须是num_skip的整数倍，确保每个batch包含一个词汇对应的所有样本
        skip_window: 单词最远可以联系的距离
    """
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape = (batch_size),dtype = np.int32)
    labels = np.ndarray(shape = (batch_size,1),dtype = np.int32)
    span = 2 * skip_window + 1
    buffer = collections.deque(maxlen = span) #deque使用append添加变量时，只保留最后插入的那个变量
    
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    
    for i in range(batch_size // num_skips):
        target = skip_window
        target_to_aviod = [skip_window]
        for j in range(num_skips):
            while target in target_to_aviod:
                target = random.randint(0,span - 1)
            target_to_aviod.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips +j,0] = buffer[target]
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
    return batch,labels

filename = maybe_download('text8.zip', 31344016)

if filename == None:
    filename = 'data/text8.zip'
words = read_data(filename)

vocabulary_size = 50000 #取词频为top 50000的单词
data,count,dictionary,reverse_dictionary = build_dataset(words)
del words

print('generate batch samples...')
batch,labels = generate_batch(8, 2, 1)
for i in range(8):
    print(batch[i],labels[i])



    
        
    
