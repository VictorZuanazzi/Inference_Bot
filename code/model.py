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
from encoder import MeanEncoder

class InferClassifier(nn.Module):
    
    def __init__(self, input_dim, n_classes, encoder, matrix_embeddings):
        """initializes a 2 layer MLP for classification.
        There are no non-linearities in the original code, Katia instructed us 
        to use tanh instead"""
        
        super(InferClassifier, self).__init__()
        
        #dimensionalities
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.hidden_dim = 512
        self.encoder = encoder
        
        #embedding
        self.embeddings = nn.Embedding.from_pretrained(matrix_embeddings)
        self.embeddings.requires_grad = False
        
        #creates a MLP
        self.classifier = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.Tanh(), #not present in the original code.
                nn.Linear(self.hidden_dim, self.n_classes))
        
    def forward(self, sentence1, sentence2):
        """forward pass of the classifier
        I am not sure it is necessary to make this explicit."""
        
        #unpacks tuples (sentence, lenght)
        sentence1 = sentence1[0]
        len1 = sentence1[1]
        sentence2 = sentence2[0]
        len2 = sentence2[1]
        
        
        #get the embeddings for the inputs
        u = self.embeddings(sentence1)
        v = self.embeddings(sentence2)
        
        #pass the data through the enconder
        u = self.encoder.forward(u, len1)
        v = self.encoder.forward(v, len2)
        
        #concatenate the data
        x = self.special_concatenation(u, v)
        
        #forward to the classifier
        return self.classifier(x)
    
    def special_concatenation(self, u, v):
        """concatentes vector u and v as specified in the paper
        """
    
        diff = u - v 
        diff = diff.abs()
        prod = u * v
    
        return torch.cat((u, v, diff, prod), dim=1)


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    