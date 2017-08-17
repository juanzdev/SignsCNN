from __future__ import print_function
import tensorflow as tf
import pandas as pd
import numpy as np
import time
from datetime import timedelta
import math
import dataset
import random
import math as math
import os
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes


num_channels = 1
img_size = 28
img_size_flat = img_size * img_size * num_channels
classes = ['A','G','V']
num_classes = len(classes)
batch_size = 254
validation_size = .2
train_path='../training_data'
test_path='../testing_data'
beta = 0.01
total_iterations = 0
alphaRelu=0.1
starter_learning_rate = 0.1

data = dataset.read_train_sets(train_path, img_size, classes, validation_size=validation_size)
test_images = dataset.read_test_set(test_path, img_size,classes)
xbatch_test = test_images.test._images
xbatch_test = xbatch_test.reshape(30, img_size_flat)


print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Test-set:\t\t{}".format(len(test_images.test.labels)))
print("- Validation-set:\t{}".format(len(data.valid.labels)))



def weight_variable(name,shape):
    return tf.get_variable(name,shape=shape,initializer=tf.contrib.layers.xavier_initializer())

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x,W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def _variable_on_cpu(name, shape, initializer):
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var

def _variable_with_weight_decay(name, shape, stddev, wd):
  var = _variable_on_cpu(name, shape,
                         tf.truncated_normal_initializer(stddev=stddev))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


session = tf.Session()
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,100000, 0.96, staircase=True)

y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])

W_conv1 = _variable_with_weight_decay("W_conv1",shape=[5,5,num_channels,32],stddev=0.1,wd=0.001)
b_conv1 = bias_variable([32])

_convlogit1= conv2d(x_image,W_conv1)+b_conv1
h_conv1 = tf.maximum(alphaRelu*_convlogit1,_convlogit1)
h_pool1 = max_pool_2x2(h_conv1)
norm1 = tf.nn.lrn(h_pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm1')

W_conv2 = _variable_with_weight_decay("W_conv2",shape=[5,5,32,64],stddev=0.1,wd=0.001)
b_conv2 = bias_variable([64])

_convlogit2= conv2d(norm1,W_conv2)+b_conv2
h_conv2 = tf.maximum(alphaRelu*_convlogit2,_convlogit2)
norm2 = tf.nn.lrn(h_conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm2')
h_pool2 = max_pool_2x2(norm2)

W_conv3 = _variable_with_weight_decay("W_conv3",shape=[5,5,64,128],stddev=0.1,wd=0.001)
b_conv3 = bias_variable([128])

_convlogit3= conv2d(h_pool2,W_conv3)+b_conv3
h_conv3 = tf.maximum(alphaRelu*_convlogit3,_convlogit3)
norm3 = tf.nn.lrn(h_conv3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm2')

W_fc1 = _variable_with_weight_decay("W_fc1",shape=[7*7*64,1024],stddev=0.1,wd=0.001)
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
_convlogit3= tf.matmul(h_pool2_flat,W_fc1)+b_fc1
h_fc1 = tf.maximum(alphaRelu*_convlogit3,_convlogit3)

keep_prob = tf.placeholder(tf.float32,name='keep_prob')
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = _variable_with_weight_decay("W_fc2",shape=[1024,num_classes],stddev=0.1,wd=0.001)
b_fc2 = bias_variable([num_classes])

y_conv = tf.matmul(h_fc1_drop,W_fc2)+b_fc2
y_conv = tf.identity(y_conv, name="y_conv")
y_conv_cls = tf.argmax(y_conv, dimension=1)
y_conv_cls = tf.identity(y_conv_cls, name="y_conv_cls")
regularizer = tf.nn.l2_loss(y_conv)
softmax_ = tf.nn.softmax(y_conv,name="softmax_tensor")
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_conv)+
    beta*tf.nn.l2_loss(W_conv1) +
    beta*tf.nn.l2_loss(W_conv2) +
    beta*tf.nn.l2_loss(W_fc1)+
    beta*tf.nn.l2_loss(W_fc2))

tf.add_to_collection('losses', cross_entropy)
total_loss= tf.add_n(tf.get_collection('losses'), name='total_loss')

#train_step = tf.train.GradientDescentOptimizer(0.01).minimize(total_loss)
train_step = tf.train.AdamOptimizer(1e-4).minimize(total_loss)
#train_step = tf.train.AdamOptimizer(1e-4).minimize(total_loss,global_step=global_step)
#train_step = tf.train.MomentumOptimizer(learning_rate,0.9).minimize(total_loss,global_step=global_step)

correct_prediction = tf.equal(y_conv_cls,y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
session.run(tf.initialize_all_variables())
train_batch_size = batch_size

def print_progress(epoch, feed_dict_train, feed_dict_validate,feed_dict_test, val_loss):
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    test_acc = session.run(accuracy, feed_dict=feed_dict_test)
    msg = "Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%}, Validation Loss: {3:.3f} , Test Accuracy: {4:>6.1%}"
    print(msg.format(epoch + 1, acc, val_acc, val_loss,test_acc))
    return acc,val_acc,test_acc

def optimize(num_iterations):
    global total_iterations
    best_val_loss = float("inf")
    for i in range(total_iterations,total_iterations + num_iterations):
        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(train_batch_size)
        x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(train_batch_size)
        x_batch = x_batch.reshape(train_batch_size, img_size_flat)
        x_valid_batch = x_valid_batch.reshape(train_batch_size, img_size_flat)
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch,
                           keep_prob:0.5}
        
        feed_dict_validate = {x: x_valid_batch,
                              y_true: y_valid_batch,
                              keep_prob:1}
                              
        feed_dict_test = {x: xbatch_test,
                              y_true: test_images.test._labels,
                              keep_prob:1}

        session.run(train_step, feed_dict=feed_dict_train)
        sft = session.run(softmax_, feed_dict=feed_dict_train)
        if i % 10 == 0:
            val_loss = session.run(cross_entropy, feed_dict=feed_dict_validate)
            epoch = int(i / int(data.train.num_examples/batch_size))
            train_acc,val_acc,test_acc = print_progress(epoch, feed_dict_train, feed_dict_validate,feed_dict_test,val_loss)
            if(val_acc > 0.9 and val_acc>0.9 and test_acc>0.8):
                print("saving snapshot...")
                saver = tf.train.Saver()
                saver.save(session, '../snapshots/snp_'+str(i)) 
        if i % 100 == 0:
            print("total iterations:")
            print(i)
            val_loss = session.run(cross_entropy, feed_dict=feed_dict_validate)
            epoch = int(i / int(data.train.num_examples/batch_size))
            train_acc,val_acc,test_acc = print_progress(epoch, feed_dict_train, feed_dict_validate,feed_dict_test,val_loss)
            if(val_acc > 0.9 and val_acc>0.9 and test_acc>0.8):
                print("saving snapshot...")
                saver = tf.train.Saver()
                saver.save(session, '../snapshots/snp_'+str(i)) 

    total_iterations += num_iterations
 
optimize(num_iterations=3000)