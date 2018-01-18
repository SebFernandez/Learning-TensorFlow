from __future__ import print_function
import tensorflow as tf
 
#Constants
 
node1 = tf.constant (3.0, dtype = tf.float32)
node2 = tf.constant (4.0, dtype = tf.float32)
node3 = tf.add (node1,node2)
 
sess = tf.Session() #A session encapsulates the control and state of the TensorFlow runtime.
 
print ('-> Constants')
print ('Node3: ', node3)
print (sess.run (node3))
print ('')
 
#Placeholders -> A placeholder is a promise to provide a value later
 
a = tf.placeholder (tf.float32)
b = tf.placeholder (tf.float32)
 
adder_node = a + b
 
print ('-> Placeholders')
print (sess.run (adder_node, {a: 3, b: 4.5}))
print (sess.run (adder_node, {a: [1,3], b: [2,4]})) #Sums a[0] with b[0], a[1] with b[1]
print ('')
 
#Variables -> Variables allow us to add trainable parameters to a graph
 
w = tf.Variable ([-1.], dtype = tf.float32)
b = tf.Variable ([1.], dtype = tf.float32)
x = tf.placeholder (tf.float32)
 
linearModel = w * x + b
 
'''
It is important to realize init is a handle to the TensorFlow sub-graph that initializes all the global variables.
Until we call sess.run, the variables are uninitialized.
'''
 
init = tf.global_variables_initializer ()
sess.run (init)
 
print ('-> Variables')
print (sess.run (linearModel, {x: [1, 2, 3, 4]}))
print ('')
 
#Loss function -> A loss function measures how far apart the current model is from the provided data.
 
 
y = tf.placeholder (tf.float32)
squaredDeltas = tf.square (linearModel - y)
loss = tf.reduce_sum (squaredDeltas)
 
print ('-> Loss function')
print (sess.run (loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))