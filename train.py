#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 05:53:22 2018

@author: Deema
"""


# Imports
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

parser = argparse.ArgumentParser(description='Image Classifier - Training Part')
parser.add_argument('--data_dir', type=str, action="store", default="flowers", help='the directory of flower images')
parser.add_argument('--gpu', dest='gpu', action='store_true', default="gpu", help='activate the GPU during the training')
parser.add_argument('--save_dir', type=str,dest="save_dir", action="store", default="checkpoint.pth", help='directory to save checkpoints')
parser.add_argument('--arch', dest='arch', action="store", default="densenet121", type = str, help='model architecture')
parser.add_argument('--learning_rate', type=float, dest="learning_rate", action="store", default=0.001, help='learning rate')
parser.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=9000, help='number of hidden units')
parser.add_argument('--epochs', type=int, dest="epochs", action="store", default=5, help='number of epochs')
parser.add_argument('--dropout', type=float, dest = "dropout", action = "store", default = 0.5, help='dropout percentage')

arg_parser = parser.parse_args()

data_dir = arg_parser.data_dir
gpu = arg_parser.gpu
save_dir = arg_parser.save_dir
model_arch = arg_parser.arch
lr = arg_parser.learning_rate
hidden_units = arg_parser.hidden_units
epochs = arg_parser.epochs
dropout = arg_parser.dropout

image_datasets_train = None


def load_data():
    data_dir = 'flowers'

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'    

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                                                            
    valid_transforms = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


# TODO: Load the datasets with ImageFolder

    train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)


# TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_datasets, batch_size=32, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_datasets, batch_size=32)
    testloader = torch.utils.data.DataLoader(test_datasets, batch_size=32)
    
    return train_datasets, valid_datasets, test_datasets 
    model = models.densenet121(pretrained=True)
    model
def model_architecture(lr=0.001, hidden_units=9000):


    
    for param in model.parameters():
        param.requires_grad = False
    
    
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

        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
        
    return model, criterion, optimizer

     # accuracy on the validation set:
def validation(model, validloader, criterion):
    model.eval()
    accuracy = 0
    for images, labels in validloader:
        images = Variable(images.float().cpu(), volatile=True)
        labels = Variable(labels.long().cpu(), volatile=True)

        output = model.forward(images) 
        valid_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return valid_loss, accuracy   

def do_deep_learning(epochs=2):
    epochs = 2
    print_every = 40
    steps = 0

print("training started")

    # change to cuda
if torch.cuda.is_available() and in_arg.gpu == 'yes':
   model.cuda()
else:
   model.cpu()
        

   model.train()

   model.to('cpu')

for e in range(epochs):
    running_loss = 0
    for ii, (inputs, labels) in enumerate(trainloader):
        steps += 1
        
        inputs, labels = inputs.to('cpu'), labels.to('cpu')
        
        optimizer.zero_grad()
        
        # Forward and backward passes
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if steps % print_every == 0:
            print("Epoch: {}/{}... ".format(e+1, epochs),
                  "Loss: {:.4f}".format(running_loss/print_every))
            
            running_loss = 0
 
            model.to('cpu')
    for e in range(epochs):
        model.train() #

        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            inputs, labels = inputs.to('cpu'), labels.to('cpu')

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
  
            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()

           # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    valid_loss, accuracy = validation(model, validloader, criterion)

                print("Epoch: {}/{}... ".format(e+1, epochs),
                 "Training Loss: {:.4f}".format(running_loss/print_every),
                 "validation Loss: {:.3f}.. ".format(valid_loss/len(validloader)),
                 "validation Accuracy: {:.3f}".format(accuracy/len(validloader)))

                running_loss = 0
                model.train()
                
                
   #Do validation on the test set
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to('cpu'), labels.to('cpu')
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total =total+labels.size(0)
        correct += (predicted == labels).sum().item()
        
print('Accuracy: %d %%' % (100 * correct / total))
    
    # save checkpoint
def save_checkpoint(model):
    model.class_to_idx = train_datasets.class_to_idx
    checkpoint = {'arch': 'densenet121',
              'state_dict': model.state_dict(),
              'optimizer_dict':optimizer.state_dict(),
              'epoch':2,
              'class_to_idx':model.class_to_idx,
              'classifier':model.classifier}
    torch.save(checkpoint,'checkpoint.oth')    
    
# Call to main function to run the program
if __name__== "__main__":

    print ("start training ...")
    train_datasets, valid_datasets, test_datasets = load_data()
    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_datasets, batch_size=32, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_datasets, batch_size=32)
    testloader = torch.utils.data.DataLoader(test_datasets, batch_size=32)
    model, criterion, optimizer = model_architecture(lr, hidden_units)
    do_deep_learning(epochs)
    save_checkpoint(model)
    print ("end training ...")