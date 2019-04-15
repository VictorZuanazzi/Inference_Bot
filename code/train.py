# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 12:40:22 2019

Train the model of model.py using epochs or steps.
Evaluate the model during training.
Use early stop when the model has reached a plateau

Run from the command line as: python train.py <model_type> <model_name> <checkpoint_path> <train_data_path>
@author: Victor Zuanazzi
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os

from utils import print_flags, accuracy
from data import get_snli, split_snli, vocab_from_snli
from model import InferClassifier

import torch
import torch.optim as optim

# Default constants
LEARNING_RATE_DEFAULT = 0.1
BATCH_SIZE_DEFAULT = 64
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'adam'
DATA_DIR_DEFAULT = './data/'
MODEL_TYPE_DEFAULT = 'base_line'
MODEL_NAME_DEFAULT = 'mean'
TRAIN_DIR_DEFAULT = './train/'
CHECKOUT_DIR_DEFAULT = './checkout/'
DEVICE_DEFAULT = 'cpu'
DEVICE = DEVICE_DEFAULT
DATA_PERCENTAGE_DEFAULT =.1

#set datatype to torch tensor
DTYPE = torch.FloatTensor

FLAGS = None

def mean_encoder(vocab, sentences, embedding_dim=300):
    """baseline: mean enocder"""
    
    num_sentences = len(sentences)
    mean_sentence = torch.zeros(num_sentences, embedding_dim)
    
    for s ,sentence in enumerate(sentences):
        num_words = len(sentence)
        for word in sentence.split():
            
            #get the embedding
#            vec = vocab.get(word, [])  
#            
#            if len(vec) > 0:
#                mean_sentence[s] += vec
#            else:
#                #ignores unknown words
#                num_words -= 1
            
            #possibly faster implementation
            mean_sentence[s] += vocab.get(word, torch.zeros(embedding_dim))
                
        #average the tensor
        mean_sentence[s] /= num_words
        
    return mean_sentence


def index_mini_batches(data_size, batch_size=64, replacement=False):
    """return lists with the indexes to perform mini batch.
    Input:
        data_size: (int), number of datapoints.
        batch_size: (int), number of examples per batch
        replacemente: (bool), define if the sampling is done with (replacement = True) 
            or without (replacement = False) replacement
    """
    
    #get the number of batches
    num_batches = data_size/batch_size
    if replacement:
        #some sentences may occur multiple times
        num_batches = int(np.ceil(num_batches))
    else:
        #sentences may be left out
        num_batches = int(np.floor(num_batches))
  
    indices = np.arange(data_size)
    rand_idx= np.random.choice(indices, 
                               size= (num_batches, batch_size),
                               replace=replacement)
    
    return rand_idx, num_batches

def special_concatenation(u, v):
    """concatentes vector u and v as specified in the paper
    """
    
    diff = u - v 
    diff = diff.abs()
    prod = u * v

    return torch.cat((u, v, diff, prod), dim=1).type(DTYPE).to(DEVICE)

def train():
    """train model on inference task"""

    ####################################
    #parameters that have to be in FLAGS
    
    print(f"torch.device {DEVICE}")
    lr = FLAGS.learning_rate
    opt_type = FLAGS.opt_type
    weight_decay = 0.99
    percentage_data = FLAGS.data_percentage
    path = FLAGS.train_data_path
    
    #choses the model 
    if FLAGS.model_name == 'mean':
        #executes baseline model
        encoder = mean_encoder
        
        model = InferClassifier(input_dim = 1200,
                                n_classes = 3).to(DEVICE)
    
    #fetch data
    print("fetching data")
    data = get_snli()
    
    #for later optimization plit_snli and vocab_from_snli can be merged into one
    #function. That will save a loop thwrough the whole data set.
    print("parsing data")
    train_premises, train_hypothesis, train_labels = split_snli(data["train"])
    dev_premises, dev_hypothesis, dev_labels = split_snli(data["dev"])
    
    
    #convert labels to torch.tensor
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    dev_labels = torch.tensor(dev_labels, dtype=torch.long)
    
    vocab = vocab_from_snli(data["train"])
    
    #encode data
    print("encoding shit")
    train_premises = encoder(vocab, train_premises)
    train_hypothesis = encoder(vocab, train_hypothesis)
    
    dev_premises = encoder(vocab, dev_premises)
    dev_hypothesis = encoder(vocab, dev_hypothesis)
    
    #store sizes
    train_size = int(len(train_premises)*percentage_data)
    dev_size = int(len(dev_premises)*percentage_data)
    
    #initialize metrics
    train_acc = np.zeros(0)
    train_loss = np.zeros(0)
    dev_acc = np.zeros(0)
    dev_loss = np.zeros(0)
  
    #create loss function
    loss_func = torch.nn.CrossEntropyLoss()
    
    #loads optimizer (change it to SGD for presenting the results)
    if opt_type == 'adam':
        optimizer_func = optim.Adam
    elif opt_type == 'SGD':
        optimizer_func = optim.SGD
    else:
        print(f"optimizer {opt_type} is not available, using Adam instead")
        optimizer_func = optim.Adam
        
    optimizer = optimizer_func(model.parameters(), 
                               lr = lr, 
                               weight_decay = weight_decay)
    
    v_lr = lr
    epoch = 0
    while v_lr > 1e-5: 
        idx, num_batches = index_mini_batches(train_size)
        print(f"epoch: {epoch}")
        
        train_acc = np.append(train_acc, 0.)
        train_loss = np.append(train_loss, 0.)
        dev_acc = np.append(dev_acc, 0.)
        dev_loss = np.append(dev_loss, 0.)
        
        #######################################################################
        #train
        
        for b in range(num_batches):
            x_pre = train_premises[idx[b]]
            x_hyp = train_hypothesis[idx[b]]
            y = train_labels[idx[b]].type(torch.LongTensor).to(DEVICE)
            
            #creates the [u,v,|u-v|,u*v]
            x = special_concatenation(x_pre, x_hyp)
            
            #set optimizer gradient to zero
            optimizer.zero_grad()
            
            #perform forward pass
            y_pred = model.forward(x)
            loss_t = loss_func(y_pred, y)
            
            #backward propagation
            loss_t.backward()
            optimizer.step()
            
            #get metrics
            train_loss[epoch] += loss_t.item()
            train_acc[epoch] += accuracy(y_pred, y)/num_batches
        
        #print train results 
        print(f"TRAIN acc: {train_acc[epoch]}, loss: {train_loss[epoch]}") 
        
        #######################################################################
        #evaluation
        idx, num_batches = index_mini_batches(dev_size)
        
        for b in range(num_batches):
            x_pre = dev_premises[idx[b]].requires_grad_(False)
            x_hyp = dev_hypothesis[idx[b]].requires_grad_(False)
            y = dev_labels[idx[b]].type(torch.LongTensor).requires_grad_(False).to(DEVICE)

            #creates the [u,v,|u-v|,u*v]
            x = special_concatenation(x_pre, x_hyp)
            
            #perform forward pass
            y_pred = model.forward(x)          
            loss_t = loss_func(y_pred, y)
            
            #get metrics
            dev_loss[epoch] += loss_t.item()
            dev_acc[epoch] += accuracy(y_pred, y)/num_batches
            
        #print eval results
        print(f"EVAL acc: {dev_acc[epoch]}, loss: {dev_loss[epoch]}")    
        
        #update learning rate
        if dev_acc[epoch] > dev_acc[epoch-1]:
            v_lr /= 5.0
            #I am not sure if that is the best way to do it.
            optimizer = optimizer_func(model.parameters(), 
                                      lr = v_lr,
                                      weight_decay=weight_decay)
            
            print(f"learning rate: {v_lr}")
        
        #increment epoch
        epoch += 1
        
    #finished training:
    print("saving results in folder...")
    np.save(path + "train_loss", train_loss)
    np.save(path + "train_acc", train_acc)
    np.save(path + "dev_less", dev_loss)
    np.save(path + "dev_acc", dev_acc)
      
    print("saving model in folder")
    torch.save(model.state_dict(), path + model.__class__.__name__ + ".pt")
        
            
    return train_acc, train_loss, dev_acc, dev_loss

def evaluate():
    """ëvaluate model on inference task"""
    pass

def test():
    """test model on inference task"""
    pass


def main():
    """
    Main function
    """
    
    #use GPUs if available
    global DEVICE
    if FLAGS.torch_device == 'cuda': 
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        DEVICE = torch.device('cpu')
    
    print(f"torch.device {DEVICE}")
    
    # Print all Flags to confirm parameter settings
    print_flags(FLAGS)
    
    #create directories if they don't exist
    if not os.path.exists(FLAGS.train_data_path):
        os.makedirs(FLAGS.train_data_path)
    if not os.path.exists(FLAGS.checkpoint_path):
        os.makedirs(FLAGS.checkpoint_path)
    
    
    print("Step 2 of project Take Over the World: Read Human.")
    
    train_acc, train_loss, dev_acc, dev_loss = train()
      
    print("Training finished successfully. \nNote to self, humanity is confusing.")
    print("Ask help to ELMo...")
    


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type = str, default = MODEL_TYPE_DEFAULT,
                          help='model type')
    parser.add_argument('--model_name', type = str, default = MODEL_NAME_DEFAULT,
                          help='model name: "mean",')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                            help='Frequency of evaluation on the dev set')
    parser.add_argument('--train_data_path', type = str, default = TRAIN_DIR_DEFAULT,
                          help='Directory for storing train data')
    parser.add_argument('--checkpoint_path', type = str, default = CHECKOUT_DIR_DEFAULT,
                          help='Directory of check point')
    parser.add_argument('--torch_device', type = str, default = DEVICE_DEFAULT,
                          help='torch devices types: "cuda" or "cpu"')
    parser.add_argument('--data_percentage', type = float, default = DATA_PERCENTAGE_DEFAULT,
                          help='percentage of the data considered for training')
    parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                          help='starting learning rate')
    parser.add_argument('--opt_type', type = str, default = OPTIMIZER_DEFAULT,
                          help='optimizers types: "adam" or "SGC"')
    FLAGS, unparsed = parser.parse_known_args()
    
    main()