from mlxtend.data import loadlocal_mnist
from sklearn.model_selection import train_test_split
from PIL import Image
import os
import tensorflow as tf

res = {}

def get_data():
        return res

def create_data():
        """
        loads dataset and then splits to train, validation and test sets
        :return: (X_train, y_train), (X_val, y_val), (X_test, y_test)
        """
        X_train, y_train = loadlocal_mnist(
                images_path='./data/source_data/train-images.idx3-ubyte',
                labels_path='./data/source_data/train-labels.idx1-ubyte')

        X_test, y_test = loadlocal_mnist(
                images_path='./data/source_data/t10k-images.idx3-ubyte',
                labels_path='./data/source_data/t10k-labels.idx1-ubyte')

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state = 1)

        global res

        res = {
                "train": (X_train, y_train),
                "val": (X_val, y_val),
                "test": (X_test, y_test)
        }

def create_directories(res):
        if os.path.basename(os.getcwd()) != 'data':
                os.chdir('data')
        for key in res:
                if not os.path.exists(key):
                        os.makedirs(key)
                        for i in range(10):
                                os.makedirs(str(key) + '/' + str(i))

                for index, img_arr in enumerate(res[key][0]):
                        im = Image.fromarray(img_arr.reshape(28,28), 'L')
                        im.save(str(key) + '/' + str(res[key][1][index]) + '/' + str(index) + '.png')

def create_summary_folders(base_dir, model_name):
        if not os.path.exists(base_dir):
                os.makedirs(base_dir)
        if not os.path.exists(base_dir + '/' + model_name):
                os.makedirs(base_dir + '/' + model_name)
        checkpoints_path, summaries_path = base_dir + '/' + model_name + '/checkpoints', base_dir + '/' + model_name + '/summaries'
        if not os.path.exists(checkpoints_path):
                os.makedirs(checkpoints_path)
        if not os.path.exists(summaries_path):
                os.makedirs(summaries_path)

        return checkpoints_path, summaries_path


def one_hot_matrix(labels, C=10):
        """
        Creates a matrix where the i-th row corresponds to the ith class number and the jth column
                         corresponds to the jth training example. So if example j had a label i. Then entry (i,j)
                         will be 1.

        Arguments:
        labels -- vector containing the labels
        C -- number of classes, the depth of the one hot dimension

        Returns:
        one_hot -- one hot matrix
        """
        C = tf.constant(C, name="C")

        one_hot_matrix = tf.one_hot(labels, C, axis=1)

        with tf.Session() as sess:
                one_hot = sess.run(one_hot_matrix)

        return one_hot