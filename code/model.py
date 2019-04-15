# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 12:39:00 2019

The following models are implemented:
    BoW model
    Uni LSTM
    Bi LSTM
    Bi LSTM with max pooling
    classifier

The functionality to load word embeddings.

@author: Victor Zuanazzi
"""
import torch
import torch.nn as nn

class InferClassifier(nn.Module):
    
    def __init__(self, input_dim, n_classes):
        """initializes a 2 layer MLP for classification.
        There are no non-linearities in the original code, Katia instructed us 
        to use tanh instead"""
        
        super(InferClassifier, self).__init__()
        
        #dimensionalities
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.hidden_dim = 512
        
        #creates a MLP
        self.classifier = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.Tanh(), #not present in the original code.
                nn.Linear(self.hidden_dim, self.n_classes))
        
    def forward(self, x):
        """forward pass of the classifier
        I am not sure it is necessary to make this explicit."""
        
        return self.classifier(x)

#class MeanEncoder(nn.Module):
#    #I am not sure this should be an nn.Module
#    
#    def __init__(self, )
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    