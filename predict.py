# Imports here
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import time
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
from PIL import Image
import matplotlib.gridspec as gridspec
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from collections import OrderedDict
from skimage import io, transform
import json
import os, random
import imageio
import argparse
#from train import *


parser = argparse.ArgumentParser(description='Image Classifier - Prediction Part')

parser.add_argument('--input', default='./flowers/test/1/image_06752.jpg', action="store", type = str, help='image path')
parser.add_argument('--checkpoint', default='./checkpoint.pth', action="store", type = str, help='checkpoint file path/name')
parser.add_argument('--top_k', default=5, dest="top_k", action="store", type=int, help='return top K most likely classes')
parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json', help='mapping the categories to real names')
parser.add_argument('--gpu', dest='gpu', action='store_true', default="gpu", help='activate the GPU during the prediction')

arg_parser = parser.parse_args()

image_path = arg_parser.input
model_path = arg_parser.checkpoint
topk = arg_parser.top_k
category_names = arg_parser.category_names
gpu = arg_parser.gpu


# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoints(filepath):
    checkpoint = torch.load(filepath)
    model= models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
        model.classifier=checkpoint['classifier']
        model.class_to_idx = checkpoint['class_to_idx']
        model.optimizer_dict= checkpoint['optimizer_dict']
        model.load_state_dict(checkpoint['state_dict'])
        lr=checkpoint['lr']
        epoch=checkpoint['epoch']
        
        return model
    # Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False
    
    
from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(1024, 1000)),
                              ('relu', nn.ReLU()),
                              ('drpot1', nn.Dropout(p=0.5)),                                  
                              ('fc2', nn.Linear(1000, 500)),
                              ('relu', nn.ReLU()),
                              ('drpot2', nn.Dropout(p=0.5)),
                              ('fc3', nn.Linear(500, 102)),
                              ('output', nn.LogSoftmax(dim=1)),
                              ]))

model.classifier = classifier
    
    if gpu == 'gpu':
        model.to('cuda')
    else:
        print("the model will be trained using gpu due to the performance")
        model.to('cuda')
    
    model.class_to_idx = checkpoint['class_to_idx']
    model.optimizer_dict = checkpoint['optimizer_dict']
    model.load_state_dict(checkpoint['state_dict'])

    return model

def process_image(image):
    pil_image = Image.open(image)
    size = 256
    width, height = pil_image.size
 
    shortest_side = min(width, height)

    pil_image = pil_image.resize((int((pil_image.width/shortest_side)*size), int((pil_image.height/shortest_side)*size)))

    img_loader = transforms.Compose([
       transforms.CenterCrop(224),
       transforms.ToTensor()])

    pil_image = img_loader(pil_image).float()

    np_image = np.array(pil_image)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np.transpose(np_image, (1, 2, 0)) - mean)/std
    np_image = np.transpose(np_image, (2, 0, 1))

    return np_image



from PIL import Image
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    
    model.eval()
    model = model.cuda()
    #image = Image.open(image_path)
    np_array = process_image(image_path)
    tensor = torch.from_numpy(np_array)
    inputs = Variable(tensor.float().cuda())
    inputs = inputs.unsqueeze(0)
    output = model.forward(inputs)
    
    ps = torch.exp(output).data.topk(topk)
    probabilities = ps[0].cpu()
    classes = ps[1].cpu()
    class_to_idx_nw = {model.class_to_idx[k]: k for k in model.class_to_idx}
    tag_classes = list()
    
    for label in classes.numpy()[0]:
        tag_classes.append(class_to_idx_nw[label])

    y_by_class = []
    for x in tag_classes:
        y_by_class.append(cat_to_name[x])
    
    #print ('y_by_class: ',y_by_class)
    
    return probabilities.numpy()[0], y_by_class

# TODO: Display an image along with the top 5 classes
img = mpimg.imread('flowers/test/10/image_07104.jpg')

f, axarr = plt.subplots(2,1)

axarr[0].imshow(img)
axarr[0].set_title('hard-leaved pocket orchid')

probs, classes = predict('flowers/test/10/image_07104.jpg', model)

y_pos = np.arange(len(classes))

axarr[1].barh(y_pos, probs, align='center', color='blue')
axarr[1].set_yticks(y_pos)
axarr[1].set_yticklabels(y_by_class) #classes
axarr[1].invert_yaxis()  # labels read top-to-bottom
_ = axarr[1].set_xlabel('Probs')




if __name__== "__main__":

    print ("start Prediction ...")
    model = load_checkpoint(model_path)
    probs, classes = predict(image_path, model, topk)
    Sanity_Checking(probs, classes)
    print ("end Prediction ...")