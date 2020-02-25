from .BaseNN import *
from tensorflow.contrib import rnn
import tensorflow as tf


from tensorflow.python.ops import rnn, rnn_cell
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM#, CuDNNLSTM

class DNN(BaseNN):

    def rnn_model(self, data, num_hidden, num_labels):
        # define constants
        # unrolled through 28 time steps
        time_steps = 28
        # hidden LSTM units
        num_units = 128
        # rows of 28 pixels
        n_input = 28
        # learning rate for adam
        learning_rate = 0.001
        # mnist is meant to be classified in 10 classes(0-9).
        n_classes = 10
        # size of batch
        batch_size = 128


        weights = {
            # matrix from inputs (28) to hidden layer (128). shape is: (28, 128)
            'in': tf.Variable(tf.random_normal([n_input, num_hidden])),
            # matrix from hidden layer to output layer, shape is: (128, 10)
            'out': tf.Variable(tf.random_normal([num_hidden, n_classes]))
        }

        # Define bias vectors
        biases = {
            # bias for the input to hidden layer (128, )
            'in': tf.Variable(tf.constant(0.1, shape=[num_hidden, ])),
            # bias from the hidden to putput layer (10, )
            'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
        }

        data = tf.reshape(data, [-1, n_input])
        X_in = tf.matmul(data, weights['in']) + biases['in']
        X_in = tf.reshape(X_in, [-1, time_steps, num_hidden])

        # basic LSTM Cell
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_hidden, forget_bias=1.0, state_is_tuple=True)
        init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

        outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=init_state, time_major=False)

        outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))  # states is the last outputs
        prediction = tf.matmul(outputs[-1], weights['out']) + biases['out']

        return prediction

    def twolayer_rnn_model(self, data, num_hidden, num_labels):
        splitted_data = tf.unstack(data, axis=1)
        print('data shape is', data.shape)
        print('splitted data is', splitted_data)

        cell1 = tf.nn.rnn_cell.BasicLSTMCell(num_hidden, forget_bias=1.0, state_is_tuple=True)
        cell2 = tf.nn.rnn_cell.BasicLSTMCell(num_hidden, forget_bias=1.0, state_is_tuple=True, activation='relu')
        cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2], state_is_tuple=True)

        # print('splitted_data shape is', splitted_data)
        outputs, state = tf.nn.static_rnn(cell, splitted_data, dtype=tf.float32)
        output = outputs[-1]

        w_softmax = tf.Variable(tf.truncated_normal([num_hidden, num_labels]))
        b_softmax = tf.Variable(tf.random_normal([num_labels]))
        logit = tf.matmul(output, w_softmax) + b_softmax
        return logit

    def network(self, X):
        #
        # lstm_cell = rnn_cell.BasicLSTMCell(128, input_shape=(X.shape[2:]), activation='relu', return_sequences=True)
        #
        # outputs, states = rnn.rnn(lstm_cell, X, dtype=tf.float32)
        return self.rnn_model(X, 128, 10)
        # model = tf.sequential();
        # model.add(rnn.BasicLSTMCell(128, input_shape=(X.shape[2:]), activation='relu', return_sequences=True))
        # model.add(tf.nn.rnn_cell.DropoutWrapper(0.2))
        #
        # model.add(rnn.BasicLSTMCell(128, activation='relu'))
        # model.add(tf.nn.rnn_cell.DropoutWrapper(0.1))
        #
        # model.add(tf.nn.rnn_cell.BasicRNNCell(32, activation='relu'))
        # model.add(tf.nn.rnn_cell.DropoutWrapper(0.2))
        #
        # model.add(tf.nn.rnn_cell.BasicRNNCell(10, activation='softmax'))
        #
        # return model

        # # Convolutional Layer 1.
        # filter_size1 = 5  # Convolution filters are 5 x 5 pixels.
        # num_filters1 = 16  # There are 16 of these filters.
        #
        # # Convolutional Layer 2.
        # filter_size2 = 5  # Convolution filters are 5 x 5 pixels.
        # num_filters2 = 36  # There are 36 of these filters.
        #
        # # Convolutional Layer 3.
        # filter_size3 = 5
        # num_filters3 = 56
        #
        # # Fully-connected layer.
        # fc_size = 128  # Number of neurons in fully-connected layer.
        #
        # shape = [filter_size1, filter_size1, self.num_channels, num_filters1]
        # weights_conv1 = tf.get_variable("weights1", shape, initializer=tf.contrib.layers.xavier_initializer(seed=1))
        # biases_conv1 = tf.get_variable("biases1", num_filters1, initializer=tf.zeros_initializer())
        #
        # shape1 = [filter_size2, filter_size2, num_filters1, num_filters2]
        # weights_conv2 = tf.get_variable("weights2", shape1, initializer=tf.contrib.layers.xavier_initializer(seed=1))
        # biases_conv2 = tf.get_variable("biases2", num_filters2, initializer=tf.zeros_initializer())
        #
        # shape = [filter_size3, filter_size3, num_filters2, num_filters3]
        # weights_conv3 = tf.get_variable("weights3", shape, initializer=tf.contrib.layers.xavier_initializer(seed=1))
        # biases_conv3 = tf.get_variable("biases3", num_filters3, initializer=tf.zeros_initializer())
        #
        # layer_conv1 = self.new_conv_layer(weights_conv1, biases_conv1, input=X, use_pooling=True)
        #
        # layer_conv2 = self.new_conv_layer(weights_conv2, biases_conv2, input=layer_conv1, use_pooling=True)
        #
        # layer_conv3 = self.new_conv_layer(weights_conv3, biases_conv3, input=layer_conv2, use_pooling=True)
        #
        # layer_flat, num_features = self.flatten_layer(layer_conv3)
        #
        # weights4 = tf.get_variable("weights4", [num_features, fc_size], initializer=tf.contrib.layers.xavier_initializer(seed=1))
        # biases4 = tf.get_variable("biases4", fc_size, initializer=tf.zeros_initializer())
        #
        # layer_fc1 = self.new_fc_layer(weights4, biases4, input=layer_flat, use_relu=True)
        #
        # weights5 = tf.get_variable("weights5", [fc_size, self.num_classes], initializer=tf.contrib.layers.xavier_initializer(seed=1))
        # biases5 = tf.get_variable("biases5", self.num_classes, initializer=tf.zeros_initializer())
        #
        # layer_fc2 = self.new_fc_layer(weights5, biases5, input=layer_fc1, use_softmax=True)
        #
        # return layer_fc2

    def metrics(self, Y, Y_pred):
        self.y_true_cls = tf.argmax(Y, axis=1)
        self.y_pred_cls = tf.argmax(Y_pred, axis=1)
        self.layer_fc2 = tf.convert_to_tensor(self.layer_fc2, dtype=tf.float32)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.layer_fc2,
                                                                labels=Y)
        self.cost = tf.reduce_mean(cross_entropy)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        correct_prediction = tf.equal(self.y_pred_cls, self.y_true_cls)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # tf.compat.v1.summary.scalar('accurcay', self.accuracy)
        # tf.compat.v1.summary.scalar('cost', self.cost)
        # self.merged = tf.compat.v1.summary.merge_all()

