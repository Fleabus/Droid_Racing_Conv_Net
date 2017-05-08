import tensorflow as tf

for i in range(10):
    with tf.name_scope("node_" + str(i)):
        tf.add(2, 4)

with tf.Session() as sess:
        writer = tf.summary.FileWriter('./logs/test_log', sess.graph)
