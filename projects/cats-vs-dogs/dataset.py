import torch
import torch.nn as nn
from torch.utils.data import Dataset

import os
import cv2
import numpy as np
from PIL import Image

class DogCatDataset(Dataset):
    def __init__(self, data_path, mode):
        super(DogCatDataset, self).__init__()
        self.data_path = data_path
        self.mode = mode
        self.samples = []

        assert self.mode in ["train", "validate"]

        image_path = os.path.join(self.data_path, mode)
        
        dog_images = os.listdir(os.path.join(image_path, "dogs"))
        cat_images = os.listdir(os.path.join(image_path, "cats"))

        #self.images = dog_images + cat_images
        self.images = [os.path.join(image_path, "dogs", image) for image in dog_images] + \
            [os.path.join(image_path, "cats", image) for image in cat_images]
        self.label_map = {
            "cat": 0,
            "dog": 1,
        }
        # labels = ["dog"] * len(dog_images) + ["cat"] * len(cat_images)
        # self.labels = [self.label_map[label] for label in labels]


    def read_img(self, filename):
        img = Image.open(filename)
        # scale 0~255 to 0~1
        np_img = np.array(img, dtype=np.float32) / 255.
        np_img = cv2.resize(np_img, (200, 200))
        return np_img


    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = self.read_img(img_path)
        image = image.transpose(2, 0, 1)
        # if "cat" in img_path
        label = "cat" if "cat" in img_path else "dog"

        return image, self.label_map[label] # one-hot coding



class TestDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.images = ["{}.jpg".format(idx+1) for idx in range(25000)]
        #print(self.images)
        #test_images = os.listdir(self.data_path)
        #self.images = test_images


    def read_img(self, filename):
        img = Image.open(filename)
        # scale 0~255 to 0~1
        np_img = np.array(img, dtype=np.float32) / 255.
        np_img = cv2.resize(np_img, (200, 200))
        return np_img


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = self.read_img(os.path.join(self.data_path, image_path))
        image = image.transpose(2, 0, 1)

        return image, image_path


if __name__ == "__main__":
    # TEST CODE
    # train
    # dataset_path = "/mnt/A/qiyh/2021/Dogs-vs-Cats/TrainingData"
    # dog_cat_dataset = DogCatDataset(data_path=dataset_path, mode="train")
    # # print(dog_cat_dataset.size())

    # sample = dog_cat_dataset[88]

    # cv2.imwrite("dog.jpg", sample[0]*255.0)

    # print(sample)

    # test
    data_path = "/mnt/A/qiyh/2021/Dogs-vs-Cats/test1"
    test_dataset = TestDataset(data_path=data_path)
    print(len(test_dataset))

    item = test_dataset[100]
    print(item.shape)

