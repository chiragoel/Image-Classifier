from tensorflow.keras.models import load_model
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import Image
import os
import numpy as np
from skimage import io
import math
import glob
import argparse


HEIGHT,WIDTH = 224,224
BATCH_SIZE = 8

class_mapping = {0:'animals', 1:'buildings',2: 'landscapes'}
def show(image,name,c):
    fig = plt.figure()
    plt.imshow(image)
    plt.axis('off')
    plt.title(name)
    plt.savefig('./figures/'+str(c)+'.jpg',bbox_inches='tight')



def predict(test_dir,name):
    if name == 'vgg':
            model = load_model('./trained_models/vgg.h5')
    elif name == 'resnet':
        model = load_model('./trained_models/resnet.h5')
    elif name == 'InceptionResnet':
        model = load_model('./trained_models/inception_resnet.h5')
    else:
        model = load_model('./trained_models/mobilenet.h5')

    c = 0
    li = []
    if not os.path.isdir('./figures'):
        os.mkdir('./figures')
    for i in glob.glob(test_dir+'/*.jpg'):
        c+=1
        im = cv2.imread(i)
        im = cv2.resize(im,(224,224))
        li.append(im)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        img = im/255.0
        im = np.expand_dims(img,0)

        predictions = model.predict(im)
        name = class_mapping[int(predictions.argmax(axis = 1))]
        show(img,name,c)
        
        


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Neural Network for Image Classification')

    parser.add_argument('-m', '--model', type=str, default='InceptionResnet',
                        help='Name of the Model to be used, Modes Names are :vgg, resnet, InceptionResnet, mobilenet (default-InceptionResnet)')
    parser.add_argument('-test_dir',type=str, default='',
                        help='Testing data directory')

    

    args = vars(parser.parse_args())
    predict(args['test_dir'],args['model'])
        

