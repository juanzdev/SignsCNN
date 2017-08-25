import tensorflow as tf
import numpy as np
import math
import dataset
import os
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import model



num_channels = 1
img_size = 28
img_size_flat = img_size * img_size * num_channels
classes = ['G','V']
num_classes = len(classes)
batch_size = 254
train_batch_size = batch_size
validation_size = .2
train_path='../training_data'
test_path='../testing_data'
beta = 0.01

starter_learning_rate = 0.1



# model
with tf.variable_scope("convolutional"):
    x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
    keep_prob = tf.placeholder(tf.float32)
    y,y_conv,y_conv_cls,variables = model.convolutional(x, keep_prob)


# train
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)
regularizer = tf.nn.l2_loss(y_conv)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_conv))
    #REDUNDANTE
    #beta*tf.nn.l2_loss(variables[0]) +
    #beta*tf.nn.l2_loss(variables[2]) +
    #beta*tf.nn.l2_loss(variables[4]) +
    #beta*tf.nn.l2_loss(variables[6]) +
    #beta*tf.nn.l2_loss(variables[8]))

tf.add_to_collection('losses', cross_entropy)
total_loss= tf.add_n(tf.get_collection('losses'), name='total_loss')
train_step = tf.train.AdamOptimizer(1e-4).minimize(total_loss)


correct_prediction = tf.equal(y_conv_cls,y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

data = dataset.read_train_sets(train_path, img_size, classes, validation_size=validation_size)
test_images = dataset.read_test_set(test_path, img_size,classes)
xbatch_test = test_images.test._images
xbatch_test = xbatch_test.reshape(24, img_size_flat)

print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Test-set:\t\t{}".format(len(test_images.test.labels)))
print("- Validation-set:\t{}".format(len(data.valid.labels)))


saver = tf.train.Saver(variables)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(train_batch_size)
        x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(train_batch_size)
        x_batch = x_batch.reshape(train_batch_size, img_size_flat)
        x_valid_batch = x_valid_batch.reshape(train_batch_size, img_size_flat)

        val_loss = sess.run(cross_entropy, feed_dict={x: x_valid_batch, y_true: y_valid_batch, keep_prob: 1.0})
        training_loss = sess.run(cross_entropy, feed_dict={x: x_batch, y_true: y_true_batch, keep_prob: 1.0})

        train_accuracy = accuracy.eval(feed_dict={x: x_batch, y_true: y_true_batch, keep_prob: 1.0})
        val_accuracy = accuracy.eval(feed_dict={x: x_valid_batch, y_true: y_valid_batch, keep_prob: 1.0})
        test_accuracy = accuracy.eval(feed_dict={x: xbatch_test, y_true: test_images.test._labels, keep_prob: 1.0})

        msg = "step {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%}, Training Loss: {3:.3f},  Validation Loss: {4:.3f} , Test Accuracy: {5:>6.1%}"
        print(msg.format(i, train_accuracy, val_accuracy, training_loss,val_loss,test_accuracy))
        #print(train_accuracy)
        #print(val_accuracy)
        #print(test_accuracy)
        if(i>3000 and test_accuracy>=0.87):
            print("saving snapshot...")
            saver = tf.train.Saver()
            saver.save(sess, '../snapshots/snp_'+str(i)) 

        sess.run(train_step, feed_dict={x: x_batch, y_true: y_true_batch, keep_prob: 0.5})
