import tensorflow as tf
from data_loader import *
from abc import abstractmethod
from utils import create_summary_folders

class BaseNN:
    def __init__(self, train_images_dir, val_images_dir, test_images_dir, num_epochs, train_batch_size,
                 val_batch_size, test_batch_size, height_of_image, width_of_image, num_channels, 
                 num_classes, learning_rate, base_dir, max_to_keep, model_name):

        self.base_dir = base_dir
        self.model_name = model_name
        self.max_to_keep = int(max_to_keep)
        self.num_classes = int(num_classes)
        self.learning_rate = learning_rate
        self.num_epochs = int(num_epochs)
        self.height_of_image = int(height_of_image)
        self.width_of_image = int(width_of_image)
        self.num_channels = int(num_channels)
        self.data_loader = DataLoader(train_images_dir, val_images_dir, test_images_dir, train_batch_size, 
                val_batch_size, test_batch_size, height_of_image, width_of_image, num_channels, num_classes)

    def new_conv_layer(self, weights, biases,
                       input,  # The previous layer.
                       use_pooling=True):  # Use 2x2 max-pooling.

        layer = tf.nn.conv2d(input=input,
                             filter=weights,
                             strides=[1, 1, 1, 1],
                             padding='SAME')

        layer += biases

        if use_pooling:
            layer = tf.nn.max_pool(value=layer,
                                   ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='SAME')

        layer = tf.nn.relu(layer)

        return layer

    def flatten_layer(self, layer):
        layer_shape = layer.get_shape()
        num_features = layer_shape[1:4].num_elements()
        layer_flat = tf.reshape(layer, [-1, num_features])
        return layer_flat, num_features

    def new_fc_layer(self, weights, biases,
                     input,  # The previous layer.
                     use_relu=False, use_softmax=False):  # Use Rectified Linear Unit (ReLU)?

        layer = tf.matmul(input, weights) + biases

        # Use ReLU?
        if use_relu:
            layer = tf.nn.relu(layer)
        if use_softmax:
            layer = tf.nn.softmax(layer)

        return layer

    def create_network(self):
        self.x = tf.placeholder(tf.float32, shape=[None, self.height_of_image, self.width_of_image], name='x')
        self.x_image = tf.reshape(self.x, [-1, self.height_of_image, self.width_of_image, self.num_channels])
        self.y_true = tf.placeholder(tf.float32, shape=[None, self.num_classes], name='y_true')

        self.layer_fc2 = self.network(self.x)

        y_pred = tf.nn.softmax(self.layer_fc2)

        self.metrics(self.y_true, y_pred)

    def initialize_network(self):
        # tf.reset_default_graph()
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

        # global_step = tf.Variable(1, dtype=tf.int32, trainable=False, name="iter_number")
        self.checkpoint_path, self.logdir = create_summary_folders(self.base_dir, self.model_name)

        ########################################################################
        # logs for TensorBoard
        self.train_writer = tf.summary.FileWriter(self.logdir  + '/train', self.session.graph)  # visualize the graph
        self.val_writer = tf.summary.FileWriter(self.logdir  + '/val')  # visualize the graph

        # load / save checkpoints
        self.saver = tf.train.Saver(max_to_keep=self.max_to_keep)

        ckpt = tf.train.get_checkpoint_state(self.checkpoint_path)

        # resume training if a checkpoint exists
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.session, ckpt.model_checkpoint_path)
            print("Loaded parameters from {}".format(ckpt.model_checkpoint_path))

    def train_model(self, display_step, validation_step, checkpoint_step, summary_step):
        train_paths_len = len(self.data_loader.train_paths)
        number_of_iterations = int(train_paths_len / self.data_loader.train_batch_size)
        print('num', number_of_iterations, train_paths_len , self.data_loader.train_batch_size)
        val_batch_size = int(len(self.data_loader.val_paths) / self.data_loader.val_batch_size)
        # Create a new Summary object with your measure
        summary_train = tf.Summary()
        summary_val = tf.Summary()

        for i in range(self.num_epochs):
            train_accuracies = []
            val_accuracies = []
            train_costs = []
            val_costs = []
            # shuffle dataset
            self.data_loader.shuffle_dataset(self.data_loader.train_paths)
            for j in range(number_of_iterations):
                # print('j and #', j, number_of_iterations)
                x_batch, y_true_batch = self.data_loader.train_data_loader(j)
                feed_dict_train = {self.x: x_batch,
                                   self.y_true: y_true_batch}
                self.session.run(self.optimizer, feed_dict=feed_dict_train)
                # Calculate the accuracy on the training-set.
                acc, cost = self.session.run([self.accuracy, self.cost], feed_dict=feed_dict_train)
                # print('acc cost', acc, cost, j)
                train_accuracies.append(acc)
                train_costs.append(cost)

            if i % display_step == 0:
                train_acc = sum(train_accuracies) / len(train_accuracies)
                train_cost = sum(train_costs) / len(train_costs)
                # Message for printing.
                print('Accuracy on Train-Set: ', i + 1, train_acc)
                print('Cost on Train-Set: ', i + 1, train_cost)

            if i % validation_step == 0:
                self.data_loader.shuffle_dataset(self.data_loader.val_paths)
                number_of_epochs = int(len(self.data_loader.val_paths) / self.data_loader.val_batch_size)
                for j in range(number_of_epochs):
                    x_val_batch, y_true_val_batch = self.data_loader.val_data_loader(random.randint(0, val_batch_size))
                    feed_dict_val = {self.x: x_val_batch,
                                     self.y_true: y_true_val_batch}
                    acc, cost = self.session.run([self.accuracy, self.cost], feed_dict=feed_dict_val)
                    val_accuracies.append(acc)
                    val_costs.append(cost)
                    val_acc = sum(val_accuracies) / len(val_accuracies)
                    val_cost = sum(val_costs) / len(val_costs)
                    # print('val_acc cost', val_acc, val_cost)
                print('Accuracy on Validation-Set: ', val_acc)
                print('Cost on Validation-Set: ', val_cost)

            if i % checkpoint_step == 0:
                self.saver.save(self.session, os.path.join(self.checkpoint_path, 'checkpoint' + str(i)))
                print("Model saved!")

            if i % summary_step == 0:
                summary_train.value.add(tag="Accuracy", simple_value=train_acc)
                summary_train.value.add(tag="Loss", simple_value=train_cost)
                self.train_writer.add_summary(summary_train, i)
                summary_val.value.add(tag="Accuracy", simple_value=val_acc)
                summary_val.value.add(tag="Loss", simple_value=val_cost)
                self.val_writer.add_summary(summary_val, i)

    def test_model(self):
        session = self.session
        accuracies = []
        costs = []

        test_paths_len = len(self.data_loader.test_paths)
        number_of_epochs = int(test_paths_len / self.data_loader.test_batch_size)

        for i in range(number_of_epochs):
            x_batch, y_true_batch = self.data_loader.test_data_loader(i)
            feed_dict_test = {self.x: x_batch,
                              self.y_true: y_true_batch}
            acc, cost = session.run([self.accuracy, self.cost], feed_dict=feed_dict_test)
            accuracies.append(acc)
            costs.append(cost)

        print('Accuracy on Test-Set: ', sum(accuracies) / len(accuracies))
        print('Cost on Test-Set: ', sum(costs) / len(costs))

    @abstractmethod
    def network(self, X):
        raise NotImplementedError('subclasses must override network()!')

    @abstractmethod
    def metrics(self, Y, y_pred):
        raise NotImplementedError('subclasses must override metrics()!')
