# -*- coding: utf-8 -*-

import tensorflow as tf
import os
import input_data
from numpy import *
import Image
#from mnist_demo import * 
sess = tf.InteractiveSession()

# weight initialization
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    # print(tf.Variable(initial).eval())
    return tf.Variable(initial)

# convolution
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
# pooling
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')



def GetImage(dir_name,filelist):
	# value=zeros([1,width,height,1])
	width=28
	height=28
	value=zeros([1,width,height,1])
	value[0,0,0,0]=-1
	label=zeros([1,10])
	label[0,0]=-1
	for filename in filelist:
		#print(filename)
		img=array(Image.open(dir_name+'/'+filename).convert("L"))
		tmp_value=img.reshape(1,width,height,1)
		if(value[0,0,0,0]==-1):
			value=tmp_value
		else:
			value=concatenate((value,tmp_value))
			
		tmp_label=zeros([1,10])
		#index=int(filename.strip().split('/')[1][0])#read label from flename
		index=int(filename.strip()[0])#read label from flename, first char in a str
		#print "input:",index
		tmp_label[0,index]=1
		if(label[0,0]==-1):
			label=tmp_label
		else:
			label=concatenate((label,tmp_label))
	return array(value),array(label)




# Create the model
# placeholder
x = tf.placeholder("float", [None, 784])
y_ = tf.placeholder("float", [None, 10])
# variables
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x,W) + b)

# first convolutinal layer  h_conv1(0,28)
w_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# second convolutional layer
w_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# densely connected layer
w_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

# dropout
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# readout layer
w_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

#softmax layer
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)
# Create the model


#validation step
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))# used to get accuracy
accuracy=tf.cast(tf.argmax(y_conv, 1),tf.float32)# used to get calculated label


labels=[0,1,2,3,4,5,6,7,8,9]

dir_name="data/test/"
files = os.listdir(dir_name)
cnt=len(files)
#print(files)

test_images,test_labels=GetImage(dir_name,files)

test_set = input_data.DataSet(test_images, test_labels)

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, 'models/mnist_10.model')
    output=sess.run(accuracy,feed_dict={x:test_set.images,y:test_set.labels,keep_prob:1.0})
    for a in range(len(output)):
        print 'input_'+str(a)+':',labels[argmax(test_labels[a])]
        print 'output_'+str(a)+':',labels[int(output[a])]
    #print("output:",labels[res.argmax()])


'''

saver = tf.train.Saver()
with tf.Session() as sess:
  for i in range(1):
    files[i]=dir_name+"/"+files[i]
    print files[i]
    test_images1,test_labels1=GetImage([files[i]])
    test = input_data.DataSet(test_images1, test_labels1)
    #res=y_conv.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
    saver.restore(sess, 'models/mnist_10_100.model')
    output=sess.run(accuracy,feed_dict={x:test.images,y:test.labels,keep_prob:1.0})[0]
    print 'output:',int(output)
    #print("output:",labels[res.argmax()])
'''
