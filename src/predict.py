import tensorflow as tf
import os
import glob
import numpy as np
import cv2
from hand_cnn import HandCnn
import model
import numpy as np
import tensorflow as tf

num_channels = 1
img_size = 28
img_size_flat = img_size * img_size * num_channels

input_folder ="../testing_data/G"
input_data_set = [img for img in glob.glob(input_folder+"/"+"*jpg")]

def convolutional(input):
    return sess.run(y, feed_dict={x: input, keep_prob: 1.0}).flatten().tolist()


x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
sess = tf.Session()
# restore trained data
with tf.variable_scope("convolutional"):
    keep_prob = tf.placeholder("float")
    print(x)
    y,y_conv,y_conv_cls,variables = model.convolutional(x, keep_prob)
saver = tf.train.Saver(variables)
saver.restore(sess, "../snapshots/snp_8624")


for in_idx, img_path in enumerate(input_data_set):
	file_name = os.path.splitext(os.path.basename(img_path))[0]
	print(file_name)
	imggray = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
	equ = cv2.equalizeHist(imggray)
	equ_resize = cv2.resize(equ,(img_size,img_size))
	#predictions = cnn.predict_single(equ_resize)

	images = []
	print(equ_resize.shape)
	images.append(equ_resize)
	images = np.array(images)
	train_batch_size = 1
	img_size_flat = img_size * img_size * num_channels
	print(img_size_flat)
	x_batch = images;
	x_batch = x_batch.reshape(train_batch_size, img_size_flat)
	output2 = convolutional(x_batch)
	print(output2)
	print(output2)