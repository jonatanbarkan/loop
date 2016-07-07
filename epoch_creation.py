import numpy as np
import cv2
from time import sleep


class EpochCreation(object):
    def __init__(self, input_size, sample_size):
        self.sample_size = sample_size  # number of pixels in a sample ROI
        self.input_size = input_size  # number of samples in a batch
        self.epoch = []

    def attach(self, ROI):
        # attach new ROI to input self.batch
        new = ROI.reshape([1, -1])  # flatten the ROI
        temp = (self.epoch, new)
        self.epoch = np.concatenate(temp)  # add last example to epoch
        # if self.epoch.shape[0] % 20 == 0:
        #     cuando = (self.epoch.shape[0] / 20)*20
        #     print('god help us all: ', cuando)
        print('ROI: ', ROI.shape)
        print('new: ', new.shape)
        print('epoch: ', self.epoch.shape)
        sleep(5)
