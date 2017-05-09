import numpy as np
import tensorflow as tf
from conv_net import ConvNet
from tensorflow.examples.tutorials.mnist import input_data
# Input data | 60,000 training samples | 10,000 testing samples
mnist = input_data.read_data_sets("temp_folder/", one_hot=True)

# The session to run
def train_neural_network():

    conv_net = ConvNet("Test_Name", [28, 28], [[5, 5, 1, 32], [5, 5, 32, 64], 7*7*64, 100, 10])
    hm_epochs = 10
    batch_size = 100
    # Starts the tensorflow session and initilizes all the variables
    with tf.Session() as sess:
        conv_net.setup_model(sess)
        # Create a summary writer, add the 'graph' to the event file.
        writer = tf.summary.FileWriter('./logs/test_log', sess.graph)
        # begin epochs
        for epoch in range(hm_epochs):
            epoch_loss = 0
            # Divide the dataset by the batch size
            for _ in range(int(mnist.train.num_examples / batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                c = conv_net.train(epoch_x, epoch_y)
                epoch_loss += c # adds the cost to the total loss of this epoch

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

            final_accuracy = 0
            for _ in range(int(mnist.test.num_examples / batch_size)):
                epoch_x, epoch_y = mnist.test.next_batch(batch_size) # Magically gets the next batch
                final_accuracy += conv_net.test(epoch_x, epoch_y)
            print("test accuracy %", final_accuracy)
# Starts training of the network

train_neural_network()
