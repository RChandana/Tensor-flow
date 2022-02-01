#1

import tensorflow as tf

x = tf.constant([[1., 2., 3.],
                 [4., 5., 6.],
                 [7., 8., 9.]])

print(x)
print(x.shape)
print(x.dtype)
print(x+x)
print(5*x)
print(x @ tf.transpose(x))
print(tf.concat([x, x, x], axis=0))
print(tf.nn.softmax(x, axis=-1))
print(tf.reduce_sum(x))


#2

import tensorflow as tf

x1 = tf.constant([1,2,3,4])
x2 = tf.constant([5,6,7,8])

result = tf.multiply(x1, x2)

print(result)



#3

import tensorflow as tf

x1 = tf.constant([1,2,3,4])
x2 = tf.constant([5,6,7,8])

result = tf.multiply(x1, x2)

sess = tf.Session()

print(sess.run(result))

sess.close()
