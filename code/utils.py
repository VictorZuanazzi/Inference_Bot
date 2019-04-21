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
        

def plot_grad_flow(named_parameters, name):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    
    #private inport
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    
    plt.savefig("./grad flow/" + name)
        
        
        