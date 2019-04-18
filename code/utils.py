# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 10:56:54 2019

@author: Victor Zuanazzi
"""
import torch
import os
import numpy as np

def print_flags(FLAGS):
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))
    
    
def accuracy(predictions, targets):
  """
  Computes the prediction accuracy, i.e. the average of correct predictions
  of the network.
  Args:
    predictions: 2D one hot econded vector[batch_size, n_classes]
    targets: 2D vector with the class indices  [batch_size].
  Returns:
    accuracy: scalar float, the accuracy of predictions,
              i.e. the average correct predictions over the whole batch

  """
  #calculates the mean accuracy over all predictions:
  accuracy = (predictions.argmax(dim=1) == targets).type(torch.float).mean()
  #accuracy = accuracy/len(targets)

  return accuracy.item() #returns the number instead of the tensor.

def print_model_params(model):
    """prints model archtecture and numbers of parameters. 
    Cotersy from Karan Malhotra.
    Input: (class nn.Module) the model"""
    
    total = 0
    for name, p in model.named_parameters():
        total += np.prod(p.shape)
        print("{:24s} {:12s} requires_grad={}".format(name, str(list(p.shape)), p.requires_grad))
    print("\nTotal parameters: {}\n".format(total))

def check_dir(path): 
    if not os.path.exists(path):
        os.makedirs(path)
        
        
        
        