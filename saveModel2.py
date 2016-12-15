'''
Save and Restore a model given by the official doc using TensorFlow.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Sparrow
'''

# Import MINST data
import input_data
from mnist_demo import * 
#input_data.py has been rewriten to read extracted file from '/home/sparrow/caffe/data/mnist/'

mnist.train=input_data.Dataset(input_data.read_images_labels('train.txt'))
mnist.validation=input_data.Dataset(input_data.read_images_labels('validation.txt'))
mnist.test=input_data.Dataset(input_data.read_images_labels('test.txt'))

#mnist = input_data.read_data_sets("", one_hot=True)

#print shape(mnist.train.labels)

import tensorflow as tf

# Parameters
learning_rate = 0.001
batch_size = 100
display_step = 1

#save to current path;model and parameters
model_path = "models/mnist_"
#model_path2 = "mnist_2.model"

n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)


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

# Create the model
# tf Graph input
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
#y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)
pred = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)
# Create the model

'''
# Construct model in original file
#pred = constructNet(x, weights, biases)
'''
# Define loss and optimizer
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
cost = -tf.reduce_sum(y_*tf.log(pred))
optimizer = tf.train.AdagradOptimizer(0.01).minimize(cost)#learning_rate too low, hard to get better result

# Initializing the variables
init = tf.initialize_all_variables()

# 'Saver' op to save and restore all the variables
saver = tf.train.Saver()


'''
# Running first session
print "Starting 1st session..."
with tf.Session() as sess:
    # Initialize variables
    sess.run(init)
    # Training cycle
    for epoch in range(20):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)#55000/100
        # total_batch = 50 # batch_size*total_batch = mnist.train.num_examples
        # Loop over all batches
        for i in range(total_batch):
            batch= mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch[0],y_: batch[1],keep_prob:1.0})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print "Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost)
        #save model every 10 epoch
        if epoch>0 and (epoch+1) % 10 == 0:
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y_, 1))
            # Calculate accuracy
            #accuracy=tf.cast(tf.argmax(pred, 1),tf.float32)#used to get calculated labels
            #accuracy2= tf.cast(tf.argmax(pred, 1),tf.float32)#used to get calculated labels
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))#get accuracy
            test_batch= mnist.test.next_batch(1000)
            #print(test_batch[1])#right labels
            #print "Accuracy:", accuracy2.eval({x: test_batch[0], y_: test_batch[1],keep_prob:0.5 })
            print "Accuracy:", accuracy.eval({x: test_batch[0], y_: test_batch[1],keep_prob:0.5 })
            save_path = saver.save(sess, model_path+str(epoch+1)+'.model')
            print "Model saved in file: %s" % save_path

    print "First Optimization Finished!"
'''
'''
    # Test model, compare net output(pred) and input labels(y_) 
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y_, 1))
    # Calculate accuracy
    #accuracy=tf.cast(tf.argmax(pred, 1),tf.float32)#used to get calculated labels
    #accuracy2= tf.cast(tf.argmax(pred, 1),tf.float32)#used to get calculated labels
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))#get accuracy
    test_batch= mnist.test.next_batch(1000)
    #print(test_batch[1])#right labels
    #print "Accuracy:", accuracy2.eval({x: test_batch[0], y_: test_batch[1],keep_prob:0.5 })
    print "Accuracy:", accuracy.eval({x: test_batch[0], y_: test_batch[1],keep_prob:0.5 })

    # Save model weights to disk
    save_path = saver.save(sess, model_path)
    print "Model saved in file: %s" % save_path
'''


# Running a new session, continue to train
print "Starting 2nd session..."
with tf.Session() as sess:
    # Initialize variables
    sess.run(init)

    # Restore model weights from previously saved model
    load_path = saver.restore(sess, model_path+'10_250.model')
    print "Model restored from file: %s" % load_path

    # Resume training
    for epoch in range(100):
        avg_cost = 0.
        #total_batch = int(mnist.train.num_examples / batch_size)
        total_batch = 50
        # Loop over all batches
        for i in range(total_batch):
            batch = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch[0], y_: batch[1],keep_prob:0.5})#keep_prob change from 1 to 0.5,accuracy from 0.7- to 0.9+
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print "Epoch:", '%04d' % (epoch + 1), "cost=", \
                "{:.9f}".format(avg_cost)
        #save model every 10 epoch
        if epoch>0 and (epoch+1) % 10 == 0:
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y_, 1))
            # Calculate accuracy
            #accuracy=tf.cast(tf.argmax(pred, 1),tf.float32)#get calculated labels
            #accuracy2= tf.cast(tf.argmax(pred, 1),tf.float32)#get calculated labels
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))#get accuracy
            #test_batch= mnist.test.next_batch(1000)
            #print(test_batch[1])#right labels
            #print "Accuracy:", accuracy2.eval({x: test_batch[0], y_: test_batch[1],keep_prob:0.5 })
            print "Accuracy:", accuracy.eval({x: batch[0], y_:batch[1],keep_prob:0.5 })
            save_path = saver.save(sess, model_path+'10_'+str(epoch+251)+'.model')
            print "Model saved in file: %s" % save_path
    print "Second Optimization Finished!"

'''
    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y_, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    test_batch= mnist.test.next_batch(1000)
    print "Accuracy:", accuracy.eval({x: test_batch[0], y_: test_batch[1],keep_prob:0.5})

    # Save model weights to disk
    save_path2 = saver.save(sess, model_path2)
    print "Model saved in file: %s" % save_path2
'''

