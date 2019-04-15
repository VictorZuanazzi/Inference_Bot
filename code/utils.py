# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 10:56:54 2019

@author: Victor Zuanazzi
"""
import torch

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

