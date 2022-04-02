from sklearn.svm import SVC
import time
import os
import _pickle as cPickle
import numpy as np

'''
Script to get prediction time for the SVM model
Joshua K 2022
'''

def main():
        model_path = "saved_model.bin"

        # load model
        if os.path.isfile(model_path):
                        with open(model_path, 'rb') as f:
                                model = cPickle.load(f)

        # get some sample data
        landmarks = np.load('fer2013_features\Training\landmarks.npy')
        hogs = np.load('fer2013_features\Training\hog_features.npy')
        datas = np.concatenate((landmarks[0], hogs[0]), axis=None)

        # do the prediction and keep track of time
        start_time = time.time()
        result = model.predict([datas])
        end_time = time.time()
        total_time = end_time - start_time 

        print(result)
        print('Prediction done in %f seconds'% total_time)

if __name__ == '__main__':
    main()

