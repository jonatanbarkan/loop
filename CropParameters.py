import numpy as np
from SetRes import SetRes


class CropParameters(object):
    def __init__(self, cap, video_width=-1, rand=True):

        ret, frame = cap.read()
        if video_width == -1:
            self.video_width = cap.get(3)
        else:
            self.video_width = video_width
            frame = SetRes.downgrade(frame, video_width)
        self.display_size = frame.shape
        self.rand = rand

    def window_size(self, p_height=0.2, p_width_to_height=1.3):
        cropper_height = int(self.display_size[0]*p_height)
        cropper_width = int(cropper_height*p_width_to_height)
        return cropper_height, cropper_width

    def center(self, center1=0, center2=0):

        if self.rand:
            center1 = np.random.randint(self.display_size[0])
            center2 = np.random.randint(self.display_size[1])
            return center1, center2
        else:
            return center1, center2
