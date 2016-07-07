import numpy as np
import cv2
from PIL import Image
from cropper import Cropper
import os
import time

# image = 'C:/Users/Barkanuki/Desktop/block.jpg'
# img = cv2.imread(image)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# print(img.shape)
# print(img.dtype)
# new = img[1:300,1:300]
# cv2.imshow('new',new)

cropSize = (5,5)
crops = Cropper(cropSize[0],cropSize[1])
path = 'C:\\Users\\Goren\\Desktop\\jonatan\\S06\\'
cap = cv2.VideoCapture(path + 'S06E01.mkv')

ret, frame = cap.read()
# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
frameNum = 0

while True:
    ret, frame = cap.read()
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # print(frame.shape,frame.dtype)
    # print(gray.shape,gray.dtype)
    cv2.imshow('show', frame)
    frameNum += 1
    # resize(frame, frame, Size(640, 360), 0, 0, INTER_CUBIC);
    # if frameNum == 25:
    path = 'C:\\Users\\Goren\\PycharmProjects\\Loop\\croppings\\' + str(frameNum) + '\\'
    if not os.path.exists(path):
        os.makedirs(path)
    crops.save_to_dir(path, gray, cropSize, frameNum)
    print(frameNum, 'time = ', time.clock())
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(frameNum)
        break

cap.release()

cv2.destroyAllWindows()


