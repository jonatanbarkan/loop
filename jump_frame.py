import cv2


class Jump(object):
    @staticmethod
    def jump_to_frame(cap, frame_number):
        cap.set(1, frame_number)

    @staticmethod
    def jump_frame(cap, jump_size):
        jump = cap.get(1) + jump_size
        cap.set(1, jump)
        # print(cap.get(1))