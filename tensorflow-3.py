import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

'''
The MNIST data is split into three parts: 55,000 data points of training data (mnist.train),
 10,000 points of test data (mnist.test),
 and 5,000 points of validation data (mnist.validation).

As mentioned earlier, every MNIST data point has two parts: an image of a handwritten digit and a corresponding label.
We'll call the images "x" and the labels "y". Both the training set and test set contain images and their corresponding labels;
for example the training images are mnist.train.images and the training labels are mnist.train.labels.

Each image is 28 pixels by 28 pixels

A one-hot vector is a vector which is 0 in most dimensions, and 1 in a single dimension.

A softmax regression has two steps: first we add up the evidence of our input being in certain classes,
 and then we convert that evidence into probabilities.
'''

x = tf.placeholder (tf.float32, [None, 784]) #784 is the dimensional vector and None means that the dimension can be of any length.
w = tf.Variable (tf.zeros ([784, 10]))
b = tf.Variable (tf.zeros ([10]))
y = tf.nn.softmax (tf.matmul (x,w) + b)
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean (-tf.reduce_sum (y_ * tf.log (y), reduction_indices = [1]))


'''
w has a shape of [784, 10] because we want to multiply the 784-dimensional image vectors by it to produce 10-dimensional vectors of evidence for the difference classes.
 b has a shape of [10] so we can add it to the output.

To determine the loss of a model is called "cross-entropy."
'''

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
sess = tf.InteractiveSession ()
tf.global_variables_initializer ().run ()

for e in range (1000):
	batch_xs, batch_ys = mnist.train.next_batch (100)
	sess.run (train_step, feed_dict = {x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal (tf.argmax (y, 1), tf.argmax (y_, 1))	
accuracy = tf.reduce_mean (tf.cast (correct_prediction, tf.float32))

print (sess.run (accuracy, feed_dict = {x: mnist.test.images, y_: mnist.test.labels}))