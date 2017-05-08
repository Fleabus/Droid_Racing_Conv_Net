#=============================================================
#                                                            #
#   Convolutional Neural Network                             #
#   - Matthew Lee-Mattner                                    #
#                                                            #
#   construct with the following:                            #
#       - name (will be used to save network)                #
#       - image_shape (a two-tuple defining the image dimen) #
#       - shape (minimum two-tuple for input and output)     #
#       Optional:                                            #
#           - (float)learning_rate = 0.01                    #
#           - (bool)pooling = uses 2x2 max pooling           #
#                                                            #
#                                                            #
#=============================================================
'''
    Example - Two conv layers with three fully connected layers for mnist dataset:

        from conv_net import ConvNet
        conv_net = ConvNet("Test_Name", [28, 28], [[5, 5, 1, 32], [5, 5, 32, 64], 7*7*64, 100, 10])


'''
import tensorflow as tf
import numpy as np

class ConvNet:
    name = "Conv Net"
    sess = -1

    conv_shape = []
    fcl_shape = []
    image_shape = []
    learning_rate = 0.0;
    x = -1
    y = -1

    # Seperates the shape into the convolutional layers and fully connected layers
    # Checks that the shape is valid
    def __init__(self, name, image_shape, shape, pooling=True, learning_rate=0.01):
        self.TAG = "ConvNet- "
        # Seperate into the convolutional layers and the fully connected layers
        conv_shape = [conv for conv in shape if isinstance(conv, int) == False and len(conv) > 0]
        fcl_shape = shape[len(conv_shape):]

        #========  Check that the convolutional layers are valid  =======
        # Check that convolutional layers were set
        if(len(conv_shape) > 0):
            # Check that all convolutinal layers are of size 4
            if(all(len(x) == 4 for x in conv_shape) == False):
                errorList = [conv for conv in conv_shape if len(conv) != 4]
                raise Exception(self.TAG + "Invalid convolutional layer shape. Length of shape should equal 4: " + str(errorList))
        else:
            raise Exception(self.TAG + "No convolutional layers provided.")

        #======== Check that the fully connected layers are valid =======
        # If the length of shape is less than two, throw an exception
        if(len(fcl_shape) < 2):
            raise Exception(self.TAG + "Invalid shape of network " + str(len(shape)) + " was supplied. Requires at least two layers")
        # If any element of shape is not an int, throw an exception
        if(all(isinstance(x, int) for x in fcl_shape) == False):
            raise Exception(self.TAG + "Invalid value in network. Requires all values to be of type 'int'")
        # If any element is less than or equal to 0, throw an exception
        if(any(x <= 0 for x in fcl_shape)):
            raise Exception(self.TAG + "Invalid value in network. All elements are required to be greater than 0")
        # if all tests pass, assign shape
        self.conv_shape = conv_shape
        self.fcl_shape = fcl_shape
        self.learning_rate = learning_rate
        self.name = name
        self.image_shape = image_shape
        self.x = tf.placeholder('float', [None, image_shape[0] * image_shape[1]], "features") # placeholder for each image
        self.y = tf.placeholder('float', [None, fcl_shape[len(fcl_shape)-1]], "labels")

    #setup computation graph
    def setup_model(self, sess):
        self.sess = sess
        self.result = self.setup_conv_layers(self.x)
        self.prediction = self.setup_fully_connected_layers(self.result)
        with tf.name_scope("cost"):
            self.cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=self.prediction, labels=self.y))
        with tf.name_scope("backprop"):
            self.optimizer = tf.train.AdamOptimizer().minimize(self.cost) # Pass the cost into the backprop algorithm
        # Compare the predicted outcome against the expected outcome
        with tf.name_scope("accuracy"):
            self.correct = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.y, 1))
            # Use the comparison to generate the accuracy
            self.accuracy = tf.reduce_mean(tf.cast(self.correct, 'float'))
        sess.run(tf.global_variables_initializer())

    def train(self, features, labels):
        _, c = self.sess.run([self.optimizer, self.cost], feed_dict={self.x: features, self.y: labels}) # Runs the optimizer with current image
        return c

    def test(self, features, labels):
        result = self.accuracy.eval(feed_dict={self.x: features, self.y: labels})
        return result

    def run(self, feature, label):
        output = sess.run([self.prediction], feed_dict={self.x: feature, self.y: label});
        return output

    # setup conv layer graph
    def setup_conv_layers(self, image):
        prev_result = tf.reshape(image, [-1, self.image_shape[0], self.image_shape[1], 1])
        with tf.name_scope("convolutional_layers"):
            for i in range(len(self.conv_shape)):
                conv = self.conv_shape[i]
                with tf.name_scope("convolutional_layer_" + str(i)):
                    # Setup convolutional process
                    conv_layer = {
                        'weights': self.weight_variable(conv),
                        'bias': self.bias_variable([conv[3]])
                    }
                    with tf.name_scope("relu"):
                        # Feed this reshaped image into the relu activation function using a conv2d window
                        h_conv = tf.nn.relu(self.conv2d(prev_result, conv_layer['weights']) + conv_layer['bias'])
                    with tf.name_scope("pooling"):
                        p_conv = self.max_pool_2x2(h_conv)
                    prev_result = p_conv

        return prev_result


    # setup fully connected layer graph
    def setup_fully_connected_layers(self, inputs):
        with tf.name_scope("fully_connected_layers"):
            inputs = tf.reshape(inputs, [-1, self.fcl_shape[0]])
            prev_result = inputs
            for i in range(len(self.fcl_shape[1:])):
                layer_nodes = self.fcl_shape[i+1]
                with tf.name_scope("fully_connected_layer_" + str(i)):
                    layer = {
                        'weights': self.weight_variable([self.fcl_shape[i], layer_nodes]),
                        'bias': self.bias_variable([layer_nodes])
                    }
                    with tf.name_scope("matmul"):
                        mat_result = tf.matmul(prev_result, layer['weights'])
                    with tf.name_scope("relu"):
                        l_result = tf.nn.relu(mat_result) + layer['bias']
                    prev_result = l_result
        return prev_result

    #========== Helper Functions========
    # returns the result of the convolutional layers over an image
    def conv2d(self, x, W, padding='SAME'):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding)
    # returns the result of the pooling
    def max_pool_2x2(self, x, padding='SAME'):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=padding)
    # returns the weights given a shape
    def weight_variable(self, shape):
        with tf.name_scope("weights"):
            print(shape)
            initial = tf.truncated_normal(shape, stddev=0.1) # Outputs random values from a truncated normal distribution
            result = tf.Variable(initial) # Returns a tensorflow variable with the resulting truncated weight values
        return result
    #returns the biases given a shape
    def bias_variable(self, shape):
        with tf.name_scope("biases"):
            initial = tf.constant(0.1, shape=shape) # Creates the bias values based on a shape. 0.1 standard deviation
            result = tf.Variable(initial) # Returns a tensorflow variable with the resulting constant bias values
        return result
