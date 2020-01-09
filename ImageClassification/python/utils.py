import cv2 #mainly used for rezising images

def env(field):
    import os
    from dotenv import load_dotenv
    load_dotenv()

    result = os.getenv(field)

    # Check if env is an array
    if(',' in result):
        result = result.split(',')

    return result

#instead of giving raw images through the neural network
#we will create grayscale images of equal size
def format_image(path):

    #format the images to the size set in .env
    size = int(env('IMG_SIZE'))

    # Change the image to grayscale
    img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)

    # Resize the image to desired resolution
    img = cv2.resize(img, (size,size))

    return img

def label_img(img):
    #get the labels from .env
    labels = env('LABELS')
    
    #the labels are in the format name.number.png.
    #Split the label to get just the name
    word_label = img.split('.')[0]

    #create a matrix
    result = [0] * len(labels)
    result[labels.index(word_label)] = 1

    return result