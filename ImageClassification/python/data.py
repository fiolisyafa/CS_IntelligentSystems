import os
import numpy as np
from utils import env, format_image, label_img


# ! toggle save rendered data
def create_train_data():
    #get the images from the training directory
    directory = env('TRAIN_DIR')

    #format the images into numpy array
    training_data = []
    for img in os.listdir(directory):
        # ignore dot files
        if(img.startswith('.')): continue
        label = label_img(img)

        path = os.path.join(directory,img)
        img = format_image(path)

        training_data.append([np.array(img),np.array(label)])

    return np.array(training_data)

# ! toggle save rendered data
def process_test_data(directory):
    #get the images from the testing directory
    test_directory = directory

    #format the images into numpy array
    testing_data = []
    for img in os.listdir(test_directory):
        # ignore dot files
        if(img.startswith('.')): continue
        img_num = img.split('.')[0]

        path = os.path.join(test_directory,img)
        img = format_image(path)

        #unlike the training images,
        #the file name will not include the object label
        testing_data.append([np.array(img), img_num])
        
    return np.array(testing_data)