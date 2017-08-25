import tensorflow as tf

num_channels = 1
img_size = 28
img_size_flat = img_size * img_size * num_channels
alphaRelu=0.1
classes = ['G','V']
num_classes = len(classes)
weight_decay_l2_reg = 0.1
# Multilayer Convolutional Network
def convolutional(x, keep_prob):

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
    
    # First Convolutional Layer
    x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])

    W_conv1 = _variable_with_weight_decay("W_conv1",shape=[5,5,num_channels,32],stddev=0.1,wd=weight_decay_l2_reg)
    b_conv1 = bias_variable([32])

    _convlogit1= conv2d(x_image,W_conv1)+b_conv1
    h_conv1 = tf.maximum(alphaRelu*_convlogit1,_convlogit1)
    norm1 = tf.nn.lrn(h_conv1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm1')
    h_pool1 = max_pool_2x2(norm1)

    W_conv2 = _variable_with_weight_decay("W_conv2",shape=[5,5,32,64],stddev=0.1,wd=weight_decay_l2_reg)
    b_conv2 = bias_variable([64])

    _convlogit2= conv2d(h_pool1,W_conv2)+b_conv2
    h_conv2 = tf.maximum(alphaRelu*_convlogit2,_convlogit2)
    norm2 = tf.nn.lrn(h_conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm2')
    h_pool2 = max_pool_2x2(norm2)

    W_conv3 = _variable_with_weight_decay("W_conv3",shape=[5,5,64,128],stddev=0.1,wd=weight_decay_l2_reg)
    b_conv3 = bias_variable([128])

    _convlogit3= conv2d(h_pool2,W_conv3)+b_conv3
    h_conv3 = tf.maximum(alphaRelu*_convlogit3,_convlogit3)
    #norm3 = tf.nn.lrn(h_conv3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm2')
    h_pool3 = max_pool_2x2(h_conv3)

    W_fc1 = _variable_with_weight_decay("W_fc1",shape=[4*4*128,1024],stddev=0.1,wd=weight_decay_l2_reg)
    b_fc1 = bias_variable([1024])

    h_pool3_flat = tf.reshape(h_pool3,[-1,4*4*128])
    _convlogit4= tf.matmul(h_pool3_flat,W_fc1)+b_fc1
    h_fc1 = tf.maximum(alphaRelu*_convlogit4,_convlogit4)

    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = _variable_with_weight_decay("W_fc2",shape=[1024,num_classes],stddev=0.1,wd=weight_decay_l2_reg)
    b_fc2 = bias_variable([num_classes])

    y_conv = tf.matmul(h_fc1_drop,W_fc2)+b_fc2
    y_conv = tf.identity(y_conv, name="y_conv")
    y_conv_cls = tf.argmax(y_conv, dimension=1)
    y_conv_cls = tf.identity(y_conv_cls, name="y_conv_cls")
    y = tf.nn.softmax(y_conv,name="softmax_tensor")

    return y,y_conv,y_conv_cls,[W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3,W_fc1, b_fc1, W_fc2, b_fc2]