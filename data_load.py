import numpy as np

'''
Script to fetch train/val/test data stored in numpy array files
Joshua K 2022
'''

def get_data(train=False, validation=False, test=False, hog_only = True):

    # define paths
    train_folder = 'numpy_fer2013\Training'
    validation_folder = 'numpy_fer2013\PublicTest'
    test_folder = 'numpy_fer2013\PrivateTest'

    # create dictionaries
    data_dict = dict()
    validation_dict = dict()
    test_dict = dict()

    # load train set
    if train:   
        if hog_only:
            data_dict['X'] = np.load(train_folder + '/hog_features.npy')
        else:
            data_dict['X'] = np.load(train_folder + '/landmarks.npy')
            data_dict['X'] = np.array([x.flatten() for x in data_dict['X']])
            data_dict['X'] = np.concatenate((data_dict['X'], np.load(train_folder + '/hog_features.npy')), axis=1)

        # load train labels
        data_dict['Y'] = np.load(train_folder + '/labels.npy')

    # load validation set 
    if validation:
        if hog_only:
            validation_dict['X'] = np.load(validation_folder + '/hog_features.npy')
        else:
            validation_dict['X'] = np.load(validation_folder + '/landmarks.npy')
            validation_dict['X'] = np.array([x.flatten() for x in validation_dict['X']])
            validation_dict['X'] = np.concatenate((validation_dict['X'], np.load(validation_folder + '/hog_features.npy')), axis=1)
        
        # load validation labels
        validation_dict['Y'] = np.load(validation_folder + '/labels.npy')

    # load test set
    if test:
        if hog_only:
            test_dict['X'] = np.load(test_folder + '/hog_features.npy')
        else:
            test_dict['X'] = np.load(test_folder + '/landmarks.npy')
            test_dict['X'] = np.array([x.flatten() for x in test_dict['X']])
            test_dict['X'] = np.concatenate((test_dict['X'], np.load(test_folder + '/hog_features.npy')), axis=1)
        
        # load test labels
        test_dict['Y'] = np.load(test_folder + '/labels.npy')
        # np.save(test_folder + "/lab.npy", test_dict['Y'])

    if test:
        return test_dict
    else: 
        return data_dict, validation_dict

