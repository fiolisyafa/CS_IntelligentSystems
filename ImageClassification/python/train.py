# Libraries
import os
import tflearn
import numpy as np
import tensorflow as tf

# Files
from model import network
from utils import env
from data import create_train_data

def train(model_name):
	#Get the lables from the .env file
	labels = env('LABELS')
	size = int(env('IMG_SIZE'))
	#create the train data (format the training images)
	train_data = create_train_data()

	tf.reset_default_graph()

	convnet = network(size, labels)

	model = tflearn.DNN(convnet, tensorboard_dir='log')

	#if the model already exists, load it so we are not training from scratch
	if os.path.exists('{}.meta'.format(model_name)):
	    model.load(model_name)
	    print('model loaded!')

	X = np.array([i[0] for i in train_data]).reshape(-1,size,size,1)
	Y = [i[1] for i in train_data]

	model.fit(
	    X,
	    Y,
	    n_epoch=50
	)

	#save the model in the models folder
	model.save('../models/' + model_name)
	print ("here")
	print(type(model))
	print("here")
	return model

# if __name__ == "__main__":
train('imagerecognition-{}-{}'.format(1e-3, '2conv-basic'))