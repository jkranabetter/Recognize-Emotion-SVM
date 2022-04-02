import time
import numpy as np
import imageio
from skimage.feature import hog
import dlib
import cv2

'''
Script to get extraction times
Joshua K 2022
'''

def get_landmarks(image, rects):
    # this function have been copied from http://bit.ly/2cj7Fpq
    if len(rects) > 1:
        raise BaseException("TooManyFaces")
    if len(rects) == 0:
        raise BaseException("NoFaces")
    return np.matrix([[p.x, p.y] for p in predictor(image, rects[0]).parts()])

# read a test image
image = cv2.imread('temp.jpg',0)

predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# do the prediction and keep track of time
start_time = time.time()

image = np.uint8(image)
imageio.imwrite('temp.jpg', image)
image2 = cv2.imread('temp.jpg')
face_rects = [dlib.rectangle(left=1, top=1, right=47, bottom=47)]
face_landmarks = get_landmarks(image2, face_rects)

features, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                                        cells_per_block=(1, 1), visualize=True)

end_time = time.time()
total_time = end_time - start_time 

print('extraction done in %f seconds'% total_time)
