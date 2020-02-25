import tensorflow as tf
from models.DNN import *
from utils import create_data, get_data, create_directories
import argparse

def str2bool(value):
    return value.lower == 'true'

parser = argparse.ArgumentParser(description='')

parser.add_argument('--train_images_dir', dest='train_images_dir', default='../data/train', help='Training images data directory.')
parser.add_argument('--val_images_dir', dest='val_images_dir', default='../data/val', help='Validation images data directory.')
parser.add_argument('--test_images_dir', dest='test_images_dir', default='../data/test', help='Testing images data directory.')

parser.add_argument('--train', dest='train', type=str2bool, default=True, help='whether to train the network')
parser.add_argument('--num_epochs', dest='num_epochs', type=int, default=30, help='epochs to train')
parser.add_argument('--train_batch_size', dest='train_batch_size', type=int, default=100, help='number of elements in a training batch')
parser.add_argument('--val_batch_size', dest='val_batch_size', type=int, default=100, help='number of elements in a validation batch')
parser.add_argument('--test_batch_size', dest='test_batch_size', type=int, default=100, help='number of elements in a testing batch')

parser.add_argument('--height_of_image', dest='height_of_image', type=int, default=28, help='Height of the images.')
parser.add_argument('--width_of_image', dest='width_of_image', type=int, default=28, help='Width of the images.')
parser.add_argument('--num_channels', dest='num_channels', type=int, default=1, help='Number of the channels of the images.')
parser.add_argument('--num_classes', dest='num_classes', default=10, help='Number of classes.')

parser.add_argument('--learning_rate', dest='learning_rate', default=1e-4, help='Learning rate of the optimizer')

parser.add_argument('--display_step', dest='display_step', default=1, help='Number of steps we cycle through before displaying detailed progress.')
parser.add_argument('--validation_step', dest='validation_step', default=1, help='Number of steps we cycle through before validating the model.')

parser.add_argument('--base_dir', dest='base_dir', default='./results', help='Directory in which results will be stored.')
parser.add_argument('--checkpoint_step', dest='checkpoint_step', default=1, help='Number of steps we cycle through before saving checkpoint.')
parser.add_argument('--max_to_keep', dest='max_to_keep', default=5, help='Number of checkpoint files to keep.')

parser.add_argument('--summary_step', dest='summary_step', default=1, help='Number of steps we cycle through before saving summary.')

parser.add_argument('--model_name', dest='model_name', default='softmax_classifier', help='name of model')

FLAGS = parser.parse_args()

def main(argv=None):
    create_data()
    create_directories(get_data())
    model = DNN(
        train_images_dir=FLAGS.train_images_dir,
        val_images_dir=FLAGS.val_images_dir,
        test_images_dir=FLAGS.test_images_dir,
        num_epochs=FLAGS.num_epochs,
        train_batch_size=FLAGS.train_batch_size,
        val_batch_size=FLAGS.val_batch_size,
        test_batch_size=FLAGS.test_batch_size,
        height_of_image=FLAGS.height_of_image,
        width_of_image=FLAGS.width_of_image,
        num_channels=FLAGS.num_channels,
        num_classes=FLAGS.num_classes,
        learning_rate=FLAGS.learning_rate,
        base_dir=FLAGS.base_dir,
        max_to_keep=FLAGS.max_to_keep,
        model_name=FLAGS.model_name,
    )

    model.create_network()
    model.initialize_network()

    if FLAGS.train:
        model.train_model(FLAGS.display_step, FLAGS.validation_step, FLAGS.checkpoint_step, FLAGS.summary_step)
    else:
        model.test_model()


if __name__ == "__main__":
    tf.app.run()
