import os
import glob
import numpy as np
from skimage.io import imread
import re
from utils import one_hot_matrix
import random

class DataLoader:

    def __init__(self, train_images_dir, val_images_dir, test_images_dir, train_batch_size, val_batch_size, 
            test_batch_size, height_of_image, width_of_image, num_channels, num_classes):

        self.train_paths = glob.glob(os.path.join(train_images_dir, "**/*.png"), recursive=True)
        self.val_paths = glob.glob(os.path.join(val_images_dir, "**/*.png"), recursive=True)
        self.test_paths = glob.glob(os.path.join(test_images_dir, "**/*.png"), recursive=True)

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size

        self.height_of_image = int(height_of_image)
        self.width_of_image = int(width_of_image)
        self.num_channels = int(num_channels)
        self.num_classes = int(num_classes)

    def load_image(self, path):
        label = int(path.split('/')[-2])
        return imread(path), label

    def shuffle_dataset(self, file_paths):
        return random.shuffle(file_paths)


# """
# map(lambda path: global_list_trian.insert(int(path.split('/')[-1].split('.')[0], path), self.train_paths)
# """
    def batch_data_loader(self, batch_size, file_paths, index):
        res = np.array([])
        label = np.array([])
        remainder = 0
        index = index % (len(file_paths) // batch_size)
        low_r = index * batch_size
        high_r = (index + 1) * batch_size

        if low_r >= len(file_paths):
            remainder = low_r - len(file_paths)
            low_r = remainder
            high_r = min(len(file_paths), low_r + batch_size)
        elif high_r >= len(file_paths):
            remainder = high_r - len(file_paths)
            high_r = len(file_paths)

        # need to be optimized
        for i in range(low_r, high_r):
            r = re.compile(".*/" + str(i) + ".png")
            path = list(filter(r.match, file_paths))[0]
            img_arr, _label = self.load_image(path)
            res = np.append(res, img_arr)
            label = np.append(label, _label)
        #         if remainder > 0:
        #           for i in range(0, remainder):
        #             r = re.compile(".*/" + str(i) + ".png")
        #             path = list(filter(r.match, file_paths))[0]
        #             img_arr, _label = self.load_image(path)
        #             res = np.append(res, img_arr)
        #             label = np.append(label, _label)

        one_hot_matrix_label = one_hot_matrix(label)
        res = res.reshape((-1, self.height_of_image, self.width_of_image))

        return res, one_hot_matrix_label

    def train_data_loader(self, index):
        return self.batch_data_loader(self.train_batch_size, self.train_paths, index)

    def val_data_loader(self, index):
        return self.batch_data_loader(self.val_batch_size, self.val_paths, index)

    def test_data_loader(self, index):
        return self.batch_data_loader(self.test_batch_size, self.test_paths, index)


