from __future__ import division
import numpy as np
import cv2
import random
import faces


class Cropper(object):

    def __init__(self, window_rows, window_cols, resolution, center1=0, center2=0):
        # window_rows is the rows step size we want the sliding window to have
        # window_cols is the columns step size we want the sliding window to have
        # center1 is the center of the box on the
        self.window_rows = window_rows
        self.original_rows = window_rows
        self.window_cols = window_cols
        self.original_cols = window_cols
        # self.center1 = center1 + (window_rows-1)/2
        # self.center2 = center2 + (window_cols-1)/2
        self.center1 = center1
        self.center2 = center2
        self.frame_rows = resolution[0]
        self.frame_cols = resolution[1]

    def move_rect_center_row(self, jump):
        self.center1 += jump

    def move_rect_center_col(self, jump):
        self.center2 += jump

    def random_walk(self):
        self.center2 = random.randint(0, self.frame_rows)
        self.center1 = random.randint(0, self.frame_cols)

    def random_size(self):
        # once it gets small it stays small because it is self referencing
        row_min = int(0.8*self.original_rows)
        row_max = int(1.2*self.original_rows)
        self.window_rows = random.randint(row_min, row_max)
        self.window_cols = np.round(1.14*self.window_rows)

    def find_rect_edges(self):
        # (center1, center2) is the pixel in the center of the rectangles view
        # up, down, left, right are the respective rows and columns location of the rectangles edges
        h = (self.window_rows-1)/2  # height step size from center to edge
        w = (self.window_cols-1)/2  # width step size from center to edge
        up = int(self.center1-h)
        down = int(self.center1+h)
        left = int(self.center2-w)
        right = int(self.center2+w)
        return up, down, left, right

    def crop(self, frame):
        # crop the frame
        up, down, left, right = self.find_rect_edges()
        img = frame[up:down, left:right]
        return img

    def display_loop(self, frame):
        for row_jump in range(frame.shape[0]):
            # self.move_rect_center_row(1)
            for col_jump in range(frame.shape[1]):
                # self.move_rect_center_col(1)
                # print(frame.shape)
                # if frame.shape == (self.window_rows - 1, self.window_cols - 1):
                self.display_rect(frame)

    def check_ROI_in_frame(self, ROI):

        # print ROI.shape[0], ROI.shape[1]
        # print self.window_rows
        # print self.window_cols

        if ROI.shape[0] == 0 & ROI.shape[1] == 0:
            return False
        elif (ROI.shape[0]+1) == self.window_rows:
            if (ROI.shape[1]+1) == self.window_cols:
                return True
        return False

    def save_to_dir(self, path, frame, size, frame_num):
        if frame.shape == (size[0] - 1, size[1] - 1):
            cv2.imwrite(path + 'cropSize_' + str(size) + 'frameNum_' + str(frame_num) + 'center_' + str(
                (self.center1, self.center2)) + '.png', frame)

    def display_rect(self, frame, color=(0, 255, 0)):
        up, down, left, right = self.find_rect_edges()
        cv2.rectangle(frame, (up, left), (down, right), color, 1)
        cv2.imshow('frame', frame)
