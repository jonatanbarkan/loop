import cv2


class Blur(object):
    def __init__(self, kernelSize=(5, 5)):
        self.kernelSize = kernelSize

    def Blur(self, ROI, sigmaX=3):
        blur = cv2.GaussianBlur(ROI, self.kernelSize, sigmaX)
        return blur

    def implant(self, frame, up, down, left, right):
        ROI = frame[up:down, left:right]
        ROI = self.Blur(ROI)
        frame[up:down, left:right] = ROI
        return frame
