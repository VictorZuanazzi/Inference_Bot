# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 12:14:15 2019

@author: Victor Zuanazzi
"""

#load data

#scramble one dimention per time

#keep track fo the peformance drops]

#big imports
import torch
import numpy as np
import matplotlib as plt
import seaborn as sns

#local imports
from data_2 import load_data
from train import mini_batch_iterator
from utils import load_classifier


def accuracy(predictions, targets):
  """
  Computes the prediction accuracy, i.e. the average of correct predictions
  of the network over all classes and for each class
  Args:
    predictions: 2D one hot econded vector[batch_size, n_classes]
    targets: 1D vector with the class indices  [batch_size].
  Returns:
    accuracy: (torch.tensor(torch.float)) with the accuracy of each class and 
        the accuracy across classes. Vector in the format:
        [acc class 0, acc clas 1, acc clas 2, acc all]
  """
  
  accuracy = torch.zeros(4)
  class_pred = predictions.argmax(dim=1)
  
  #calculates the mean accuracy over all predictions:
  accuracy[3] = (class_pred == targets).type(torch.float).mean().item()
  
  #calculate the accuracy for each sepeated class:
  for i in range(3):
      accuracy[i] = ((class_pred == i) == (targets == i)).type(torch.float).mean().item()

  return accuracy

def no_spaces(string):
    """replaces spaces by underscores of a string"""    
    string = string.split()
    string = "_".join(string)
    return string

def plot_n_save(data, title='', xlabel='GloVe Dimension', ylabel='Importance'):
    #set nice background and good size.
    sns.set(rc={'figure.figsize':(8,5)})
    
    #makes the plot
    ax = sns.barplot(x = [i for i in range(len(data))], 
                          y = data)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    fig = ax.get_figure()
    
    #parse the title so to have no spaces
    title = no_spaces(title)
    
    #save image
    fig.savefig(title + ".png")
    
DEVICE = torch.device('cpu')

def main():
    
    start_messages= np.array(["Please be patient, this GloVe is heavy!",
                              "Loading GloVe...", 
                              "It fits like a GloVe (it just takes a while...",
                              "... ... .. ... ...", 
                              "While you wait, why don't you google if pinguins have knews?",
                              "While you wait... Why are there no mamals that are green?",
                              "Whiie you wait, have you wondered on what are the main drivers for the pay gap between men and women",
                              "While you wait, google tiny turtle eating a strawberry."])
                              
    print(f"{np.random.choice(start_messages, 1)}")
    
    #load data
    d_data, text_f, _ = load_data()    
    d_data["test"].examples = d_data["test"].examples[0:100]
    
    batch_size = 64
    
    encoder_type = 'maxlstm'
    
    #load model
    model = load_classifier(text_f.vocab.vectors, 
                            encoder_type = encoder_type)
    model.eval()
    
    #loads the batches
    batch_iters = mini_batch_iterator(d_data, batch_size)     
    train_baches = len(batch_iters["train"])
    dev_batches = len(batch_iters["dev"])
    test_batches = len(batch_iters["test"])
    
    
    gold_acc = 0
    #check model accuracy on the data set with the original embeddings
    for batch in batch_iters["test"]:
        x_pre = batch.premise
        x_hyp = batch.hypothesis
        y = batch.label
               
        #perform forward pass
        y_pred = model.forward(x_pre, x_hyp)
        
        #get metrics
        gold_acc += accuracy(y_pred, y)/test_batches
        
        #avoid memory issues
        y_pred.detach()
        
    #print eval results
    print(f"Accuracies of reference: {gold_acc}")  
    
    emb_dim = 300
    acc_dim = torch.zeros((emb_dim, 4))
    for d in range(emb_dim):
        
        #create embeddings with one dimension shuffled
        emb_matrix = text_f.vocab.vectors.clone()
        
        #get a random permutation across one of the dimensions
        rand_index = torch.randperm(text_f.vocab.vectors.shape[0])
        emb_matrix[:, d] =  text_f.vocab.vectors[rand_index, d]
            
        #load model with the scrumbled embeddings
        model = load_classifier(emb_matrix.no_grad, 
                                encoder_type = encoder_type)
        model.eval()
    
    
        for batch in batch_iters["test"]:
            x_pre = batch.premise
            x_hyp = batch.hypothesis
            y = batch.label
                   
            #perform forward pass
            y_pred = model.forward(x_pre, x_hyp)        
            
            #calculate accuracies
            acc_dim[d] += accuracy(y_pred, y)/test_batches
            
            #avoid memory issues
            y_pred.detach()
        
        print(f"Dimension {d} accuracies: {acc_dim[d]}")   
        
        #avoid memory issues
        emb_matrix = None
    
    dim_importance = gold_acc - acc_dim
    
    
    
    
        

if __name__ == '__main__':
    main()
