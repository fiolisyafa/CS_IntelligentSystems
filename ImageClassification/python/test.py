from utils import env
from data import process_test_data
from model import network
import numpy as np
import tflearn

#have the model classify the test images and plot them with the labels
def test_debug(model_name):
	# Only import when called
	import matplotlib.pyplot as plt

	labels = env('LABELS')
	size = int(env('IMG_SIZE'))
	# if you need to create the data:
	test_data = process_test_data(env('TEST_DIR'))
	# if you already have some saved:
	# test_data = np.load('test_data.npy')

	convnet = network(size, labels)
	model = tflearn.DNN(convnet, tensorboard_dir='log')

	#load the model for testing
	model.load('../models/' + model_name)

	fig = plt.figure()

	#plot the test images along with the label as identified by the model
	for num,data in enumerate(test_data):
	    img_num = data[1]
	    img_data = data[0]
	    
	    y = fig.add_subplot(2,2,num+1)
	    orig = img_data
	    data = img_data.reshape(size,size,1)
	    model_out = model.predict([data])[0]
	    
	    print(model_out)
	    print(np.argmax(model_out))

	    str_label = labels[np.argmax(model_out)]
	        
	    y.imshow(orig,cmap='gray')
	    plt.title(str_label)
	    y.axes.get_xaxis().set_visible(False)
	    y.axes.get_yaxis().set_visible(False)

	plt.show()

#classify the image and returns the result as a text
def test(model_name):
	labels = env('LABELS')
	size = int(env('IMG_SIZE'))

	#format the images that are uploaded
	test_upload = process_test_data(env('TEST_UPLOAD'))

	convnet = network(size, labels)
	model = tflearn.DNN(convnet, tensorboard_dir='log')

	#load the model for testing
	model.load('models/' + model_name)

	for num,data in enumerate(test_upload):
	    img_num = data[1]
	    img_data = data[0]

	    orig = img_data
	    data = img_data.reshape(size,size,1)
	    model_out = model.predict([data])[0]
	    
	    print(model_out)
	    print(np.argmax(model_out))

	    str_label = labels[np.argmax(model_out)]
	        
	    #output as a text
	    print(str_label)


# test_debug('imagerecognition-{}-{}'.format(1e-3, '2conv-basic'))
test('imagerecognition-{}-{}'.format(1e-3, '2conv-basic'))