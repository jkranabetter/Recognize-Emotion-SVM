import time
import os
import _pickle as cPickle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from data_load import get_data 

'''
Script to train and test the support vector classifier.

To train, set 'train_model' to True, select features by setting 'hog_only', then run the script.
To test, set 'train_model' to False, keep features the same as training, and run the script.

Joshua K 2022
'''

def main():
        # use this to choose whether you want to train/evaluate the model
        TRAIN_MODEL = False # if false, evaluates model on test set
        # choose prediction data
        HOG_ONLY = False # if false, loads hog and landmarks

        model_path = "saved_model.bin"

        if TRAIN_MODEL:
                data, validation = get_data(train = True, validation = True, hog_only = HOG_ONLY)
        else:
                test = get_data(test=True, hog_only = HOG_ONLY)
        
        if TRAIN_MODEL:
            # training
            model = SVC(random_state=0, kernel='rbf', gamma='auto')
            print('num training samples: %d'% (len(data['Y'])))
            print('num validation samples: %d'% (len(validation['Y'])))
        
            # fit model
            start_time = time.time()
            model.fit(data['X'], data['Y'])
            train_time = time.time() - start_time
            print('training time = %f sec'% train_time)

            # save model
            with open(model_path, 'wb') as f:
                cPickle.dump(model, f)

            # validate model
            val_accuracy = eval_model(model, validation['X'], validation['Y'])
            print('validation accuracy = %f'% (val_accuracy*100))
            return val_accuracy
        else:
            # test model
            if os.path.isfile(model_path):
                with open(model_path, 'rb') as f:
                        model = cPickle.load(f)
            else:
                print( "file not found")
                exit()
            print('test samples: %d'% (len(test['Y'])))
            test_accuracy = eval_model(model, test['X'], test['Y'])
            print('test accuracy = %f'% (test_accuracy*100))
            return test_accuracy

# function to evaluate a model
def eval_model(model, X, Y):
        predicted_Y = model.predict(X)
        accuracy = accuracy_score(Y, predicted_Y)
        return accuracy

if __name__ == "__main__":
        main()
