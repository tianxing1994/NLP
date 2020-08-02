import tensorflow as tf


dense = tf.constant(value=[[1, 2, 2, 1],
                           [2, 3, 4, 5],
                           [1, 2, 2, 1]], dtype=tf.float32)

dense = tf.layers.dense(dense, units=5, kernel_initializer=tf.ones_initializer, bias_initializer=tf.ones_initializer)


sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(dense))
