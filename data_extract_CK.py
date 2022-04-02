import numpy as np
import pandas as pd
import os
import dlib
import cv2
import imageio
from skimage.feature import hog

'''
Script to load csv data into npy for increased speed
Delete previous output folder before running again
Joshua K 2022
'''

def fetch_landmarks(image, rects, predictor):
        if len(rects) > 1:
            raise BaseException('too many faces')
        if len(rects) == 0:
            raise BaseException('no faces')
        return np.matrix([[p.x, p.y] for p in predictor(image, rects[0]).parts()])

def main():
    # initialization
    img_height = 48
    img_width = 48

    output_folder = "numpy_CK"
    get_landmarks = True
    get_hog = True
    omit_labels = [] # from 0-6

    # choose csv to extract
    data = pd.read_csv('data/CK_dataset.csv')

    # load Dlib predictor
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    # make folder
    os.makedirs(output_folder)

    for category in data['Usage'].unique():
        print('processing category: %s'% category)

        # create sub folder
        os.makedirs(output_folder + '/' + category)

        # get samples and labels of the actual category
        category_data = data[data['Usage'] == category]
        samples = category_data['pixels'].values
        labels = category_data['emotion'].values
        
        # lists to store data
        images = []
        landmarks = []
        labels_list = []
        hog_features = []

        for i in range(len(samples)):

            # omit select labels
            if labels[i] in omit_labels:
                continue

            # fetch image
            image = np.fromstring(samples[i], dtype=int, sep=" ").reshape((img_height, img_width))
            images.append(image)

            # get hog features
            if get_hog:
                features, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                                        cells_per_block=(1, 1), visualize=True)
                hog_features.append(features)
            
            # get facial landmarks
            if get_landmarks:
                image = np.uint8(image)
                imageio.imwrite('temp.jpg', image)
                image2 = cv2.imread('temp.jpg')
                face_rectangles = [dlib.rectangle(left=1, top=1, right=47, bottom=47)]
                face_landmarks = fetch_landmarks(image2, face_rectangles, predictor)
                landmarks.append(face_landmarks)   

            # append label         
            labels_list.append(labels[i])

        # save data to numpy array files (faster than csv)
        np.save(output_folder + '/' + category + '/images.npy', images)
        np.save(output_folder + '/' + category + '/labels.npy', labels_list)
        if get_hog:
            np.save(output_folder + '/' + category + '/hog_features.npy', hog_features)
        if get_landmarks:
            np.save(output_folder + '/' + category + '/landmarks.npy', landmarks)

if __name__ == '__main__':
    main()