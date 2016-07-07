from __future__ import division
import numpy as np
import cv2
import epoch_creation as epoch
from cropper import Cropper
from CropParameters import CropParameters as param
from time import sleep
from SetRes import SetRes
from jump_frame import Jump
import faces
import blur
import autoencoder as AE
from scenes import Scenes


path = '../../shared/S06E02.mkv'
xml_detector_path = 'haarcascade_frontalface_default.xml'

cap = cv2.VideoCapture(path)

wanted_video_width = 320
crop_parameters = param(cap, wanted_video_width, rand=False)
# wanted_video_width = cap.get(3)
# crop_parameters = param(cap)

window_rows, window_cols = crop_parameters.window_size(0.35)
center1, center2 = crop_parameters.center()
ROI_size = (window_rows-1, window_cols-1, 3)

crops = Cropper(window_rows, window_cols, crop_parameters.display_size, center1, center2)
# detector = faces.FaceDetection(xml_detector_path) # initialize artificial face detector

fps = cap.get(5)
wanted_fps = 5

jump = int(fps/wanted_fps)
countframes = 0
numface = 0
# cap.set(1, 210)


# scenes = Scenes(path, fps)
# scene_list, scene_list_msec, scene_list_tc = scenes.jump_cuts()
# scene_number = 0
# first_frame, last_frame = scenes.num_frames_in_scene(scene_list, scene_number)
# input_size = ROI_size[0]*ROI_size[1]*ROI_size[2] # number of pixels in a ROI*3
# sample_size = last_frame-first_frame #  number of samples in a scene +-1

# all movie:
input_size = ROI_size[0]*ROI_size[1]*ROI_size[2] # number of pixels in a ROI*3
sample_size = int(cap.get(7)/jump) #  number of samples in a scene +-1

epoch = epoch.EpochCreation(input_size, sample_size)
while cap.get(1) < cap.get(7):

    ret, frame = cap.read() # read next frame

    Jump.jump_frame(cap, jump) # jump to the next frame (adjusted fps = cap.get(1)/jump)
    frame = SetRes.downgrade(frame, wanted_video_width) # manipulating the frame into requested resolution

    # set new center and size of rectangle

    # Initialize new ROI
    crops.random_walk()
    # crops.random_size()import numpy as np

    up, down, left, right = crops.find_rect_edges()
    ROI = frame[up:down, left:right]

    # create a ROI with the right dimensions
    while not crops.check_ROI_in_frame(ROI):
        up, down, left, right = crops.find_rect_edges()
        ROI = frame[up:down, left:right]
        crops.random_walk()
        # sleep(1)

    # create epoch for Autoencoder
    # while ret:
    epoch.attach(ROI)

    # display the ROI in a separate window
    cv2.imshow('ROI', ROI)
    cv2.rectangle(frame, (up, left), (down, right), (0, 255, 0), 1)


    # blurring the ROI or the entire frame
    # mask = blur.Blur((3, 3))
    # frame = mask.implant(frame, up, down, left, right)

    # FACE DETECTION haar cascade
    # faces = detector.detect(frame, scaleFactor=1.3, minNeighbors=4)
    # numface += np.minimum(1, len(faces))
    # for (x, y, w, h) in faces:
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # show frame
    cv2.imshow('frame', frame)

    # print the count of the number of frames up until now
    countframes += 1
    if countframes % 10 == 0:
        print('frame number:', countframes)

    # save to dir:
    # crops.save_to_dir(path, ROI, )

    # break out of loop by pressing q or wait till finish
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    sleep(0.05)

print(epoch.epoch.shape)
precent = np.round(100*numface/countframes, 1)
print('faces in clip = ' + str(precent) + '%')
cap.release()
cv2.destroyAllWindows()
