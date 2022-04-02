## Expression recognition using a Support Vector Machine
Using a multi-class SVM classifier (Sklearn) to predict facial expressions. Uses dlib shape predictor for landmark detection points. Uses hog extractor from skimage. 

# Instructions

1. Install required packages in requirements.txt.

2. Download datasets, for FER-2013 (link below) put the 'fer2013' csv file in the data folder, then run 'data_extract_FER2013' to store numpy files. For the CV dataset, put the dataset in the data folder then use 'data_create_CK_CSV' to create csv file. Then run 'data_extract_CK' to store numpy files.

3. Get things moving in the 'train.py' file where further instructions are included.

## Classification Results :

|       Features       |     Dataset      |  7 emotions   |   6 emotions   |   5 emotions   |
|----------------------|------------------|---------------|----------------|----------------|
| HoG features         |    FER2013       |     27.4%     |      27.8%     |      30.9%     |
| HoG features         |       CK         |     25.4%     |      26.9%     |      29.2%     |
| Face landmarks + HOG |    FER2013       |     48.1%     |      48.6%     |      49.3%     |
| Face landmarks + HOG |       CK         |     89.3%     |      92.5%     |      93.0%     |


Performed on Intel i7 3.6Ghz CPU. Training takes ~151MB memory.
Training time is 1 - 2 minutes on FER2013 dataset. Training time is < 1 second on CK dataset.

FER2013 Dataset\
https://www.kaggle.com/datasets/msambare/fer2013\
(0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)\
7 emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral\
6 emotions: Angry, Fear, Happy, Sad, Surprise, Neutral\
5 emotions: Angry, Happy, Sad, Surprise, Neutral

CK Dataset\
(0=Angry, 1=Contempt, 2=Disgust, 3=Fear, 4=Happy, 5=Sadness, 6=Surprise)\
7 emotions: Angry, Contempt, Disgust, Fear, Happy, Sadness, Surprise\
6 emotions: Angry, Disgust, Fear, Happy, Sadness, Surprise\
5 emotions: Angry, Disgust, Happy, Sadness, Surprise

## Info
Code was written and tested on Python 3.9.10.
