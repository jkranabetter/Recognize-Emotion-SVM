import os
import cv2
import numpy as np
import csv
from sklearn.model_selection import train_test_split

'''
Script to store CK dataset in a csv file for consistancy with FER2013.
Joshua K 2022
'''

rootdir ='data\CK_Dataset'

emotion_dict = {'anger':0, 'contempt':1, 'disgust':2, 'fear':3, 'happy':4, 'sadness':5, 'surprise':6}

def main():
    X_pixels = []
    Y_emotions = []

    for subdir, dirs, files in os.walk(rootdir):
        
        # get keyword
        emotion = subdir.split('\\')[-1]
        
        # skip if not correct directory
        if emotion not in emotion_dict.keys():
            print('skipping %s folder'% emotion)
            continue

        print(emotion)

        # iterate through files in the correct directories
        print(len(files))
        for file in files:
            # read file
            current_path = subdir + '\\' + file
            image = cv2.imread(current_path,0)
            # print(image)
            # print(np.shape(image))
            pixels = ''
            for row in image:
                for pixel in row:
                    pixels += str(pixel) + ' '

            # change pixels list to np array
            pixels = np.array(pixels)
            
            X_pixels.append(pixels)
            Y_emotions.append(emotion)

    # split data
    X_train, X_test, Y_train, Y_test = train_test_split(X_pixels, Y_emotions, test_size=0.2, random_state=1, stratify=Y_emotions)

    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.25, random_state=1, stratify=Y_train) # 0.25 x 0.8 = 0.2

    # write data
    f = open('data/CK_dataset.csv', 'w', newline='')
    writer = csv.writer(f, )
    writer.writerow(['emotion', 'pixels', 'Usage'])

    # test
    for idx, val in enumerate(X_test):
        writer.writerow([emotion_dict[Y_test[idx]], X_test[idx], 'test'])
    #validation
    for idx, val in enumerate(X_val):
        writer.writerow([emotion_dict[Y_val[idx]], X_val[idx], 'validation'])
    # train
    for idx, val in enumerate(X_train):
        writer.writerow([emotion_dict[Y_train[idx]], X_train[idx], 'training'])

if __name__ == '__main__':
    main()
