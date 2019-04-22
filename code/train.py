# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 12:40:22 2019

Train the model of model.py using epochs or steps.
Evaluate the model during training.
Use early stop when the model has reached a plateau

Run from the command line as: python train.py <model_type> <model_name> <checkpoint_path> <train_data_path>
@author: Victor Zuanazzi
"""

#heavy duty shit
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#make debugging more digestible
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torchtext
from torchtext import data
import torch
import torch.optim as optim
import time
import argparse
import numpy as np

#local imports
from utils import print_flags, accuracy, print_model_params, check_dir, plot_grad_flow
from data_2 import get_snli, split_snli, vocab_from_snli, load_data
from model import InferClassifier
from encoder import MeanEncoder, UniLSTM, BiLSTM, MaxLSTM

# Default constants
LEARNING_RATE_DEFAULT = 0.001
BATCH_SIZE_DEFAULT = 64
MAX_EPOCHS_DEFAULT = 20
OPTIMIZER_DEFAULT = 'adam' #'SGD', 'adam'
DATA_DIR_DEFAULT = './data/'
MODEL_TYPE_DEFAULT = ''
MODEL_NAME_DEFAULT =  'unilstm' #'unilstm' #'maxlstm'#'bilstm'# #'mean'
TRAIN_DIR_DEFAULT = './train/'
CHECKOUT_DIR_DEFAULT = './checkout/'
DEVICE_DEFAULT = 'cpu' #'cuda'
DEVICE = torch.device(DEVICE_DEFAULT)
DATA_PERCENTAGE_DEFAULT =.000005
WEIGHT_DECAY_DEFAUT = 0.01
PLOT_GRAD_DEFAULT = True

#set datatype to torch tensor
DTYPE = torch.FloatTensor

FLAGS = None

def mini_batch_iterator(d_data, batch_size):
    """return lists with the indexes to perform mini batch.
    Input:
        data_size: (int), number of datapoints.
        batch_size: (int), number of examples per batch
        replacemente: (bool), define if the sampling is done with (replacement = True) 
            or without (replacement = False) replacement
    """
 
    batch_iter = {"train": None, "dev": None, "test": None}
    batch_iter["train"], batch_iter["dev"], batch_iter["test"] = data.BucketIterator.splits(
                                datasets = (d_data["train"], d_data["dev"], d_data["test"]),
                                batch_sizes = (batch_size, batch_size, batch_size),  
                                sort_key = None, 
                                device= DEVICE,
                                shuffle = True) 

    return batch_iter

def train(training_code = ''):
    """train model on inference task"""

    ####################################
    #parameters that have to be in FLAGS
    lr = FLAGS.learning_rate
    opt_type = FLAGS.opt_type
    weight_decay = FLAGS.weight_decay
    percentage_data = FLAGS.data_percentage
    path_finished = FLAGS.train_data_path
    path_checkpoint = FLAGS.checkpoint_path
    batch_size = FLAGS.batch_size
    max_epochs = FLAGS.max_epochs
    plot_grad = FLAGS.plot_grad
    
    
    #fetch data
    print("fetching data")
    #I am not sure if text_f and label_f are necessary
    d_data, text_f, label_f = load_data(percentage_vocab = percentage_data, 
                                        percentage_data = percentage_data)   
    
    #number of samples in each dataset
    train_size = len(d_data["train"])
    dev_size = len(d_data["dev"])
    test_size = len(d_data["test"])
    print(f"train size: {train_size}, dev size: {dev_size}, test size: {test_size}")
    
    #choses the encoder 
    print("chosing encoder")
    model_n = FLAGS.model_name.lower()
    if model_n == 'mean':
        #executes baseline model
        encoder = MeanEncoder      
        in_dim= 4*300
    elif model_n == 'unilstm':
        #Uni directional LSTM
        encoder = UniLSTM
        in_dim= 4*2048
    elif model_n == 'bilstm':
        encoder = BiLSTM
        in_dim= 4*2*2048
    elif model_n == 'maxlstm':
        encoder = MaxLSTM
        in_dim= 4*2*2048
        
    #loads the model
    model = InferClassifier(encoder = encoder(),
                            input_dim = in_dim,
                            n_classes = 3,
                            matrix_embeddings = text_f.vocab.vectors).to(DEVICE)
    
    
    #print paramters of the model
    print_model_params(model)
    
    #name the model
    #It could be called John, Amanda or Thomas, but that is hard to automate
    model_name = model.__class__.__name__ + "_"
    model_name += encoder.__class__.__name__ + "_"
    model_name += model_n + "_"
    model_name += training_code + "_"
    
    #initialize metrics
    train_acc = np.zeros(0)
    train_loss = np.zeros(0)
    dev_acc = np.zeros(0)
    dev_loss = np.zeros(0)
  
    #create loss function
    loss_func = torch.nn.CrossEntropyLoss()
    
    #loads optimizer (SGD should be used to replicate the results)
    if opt_type == 'adam':
        optimizer_func = optim.Adam
        v_lr = 10 #just set it to a hibh number bot enter the while loop
    elif opt_type == 'SGD':
        optimizer_func = optim.SGD
        v_lr = lr
    else:
        print(f"optimizer {opt_type} is not available, using Adam instead")
        optimizer_func = optim.Adam
        v_lr = 10 #just set it to a high number bot enter the while loop
        
    optimizer = optimizer_func(model.parameters(), 
                               lr = lr, 
                               weight_decay = weight_decay)
    
    #get the number of batches for each dataset
    batch_iters = mini_batch_iterator(d_data, batch_size)
    train_baches = len(batch_iters["train"])
    dev_batches = len(batch_iters["dev"])
    test_batches = len(batch_iters["test"])
    
    epoch = 0
    best_acc = 0.0 
    v_lr = lr
    while v_lr > 1e-5 and epoch <= max_epochs: 
        
        print(f"epoch: {epoch}")
        
        #get the batch iterator for the mini batches
        batch_iters = mini_batch_iterator(d_data, batch_size)
        
        train_acc = np.append(train_acc, 0.)
        train_loss = np.append(train_loss, 0.)
        dev_acc = np.append(dev_acc, 0.)
        dev_loss = np.append(dev_loss, 0.)
        
        #######################################################################
        #train
        model.train() 
        for batch in batch_iters["train"]:
            
            #loads the batch
            x_pre = batch.premise
            x_hyp = batch.hypothesis
            y = batch.label
            
            #set optimizer gradient to zero
            optimizer.zero_grad()
            
            #perform forward pass
            y_pred = model.forward(x_pre, x_hyp)         
            
            loss_t = loss_func(y_pred, y)
            
            #backward propagation
            loss_t.backward()
            
            #optimization step
            optimizer.step()
            
            #get metrics
            train_loss[epoch] += loss_t.item()/y.shape[0]
            train_acc[epoch] += accuracy(y_pred, y)/train_baches
        
        #print train results 
        print(f"TRAIN acc: {train_acc[epoch]}, loss: {train_loss[epoch]}") 
        
        if plot_grad:
            #visualize the gradient flow
            plot_grad_flow(model.named_parameters(), name=model_n +str(epoch))
        
        #######################################################################
        #evaluation
        model.eval()
        for batch in batch_iters["dev"]:
            x_pre = batch.premise
            x_hyp = batch.hypothesis
            y = batch.label.requires_grad_(False)
                   
            #perform forward pass
            y_pred = model.forward(x_pre, x_hyp)        
            
            loss_t = loss_func(y_pred, y)
            
            #get metrics
            dev_loss[epoch] += loss_t.item()/y.shape[0]
            dev_acc[epoch] += accuracy(y_pred, y)/dev_batches
            
            #avoid memory issues
            y_pred.detach()
            
        #print eval results
        print(f"EVAL acc: {dev_acc[epoch]}, loss: {dev_loss[epoch]}")   
        
        
        #update learning rate
        if (dev_acc[epoch] < best_acc) & (opt_type == 'SGD'):
            #decrease learning rate as in the paper
            v_lr /= 5.0
            print(f"learning rate: {v_lr}")
            
            for g in optimizer.param_groups:
                g['lr'] = v_lr       
        else:
            #updates the best accuracy
            best_acc = dev_acc[epoch]
        
        #overfit 
        if train_acc[epoch] == 1.0:
            print("!!!Warning!!! The train accuracy is too dam high!")
            break
        
        #increment epoch
        epoch += 1
        
        #save intermediary results
        np.save(path_checkpoint + "train_loss" + model_n + training_code, train_loss)
        np.save(path_checkpoint + "train_acc" + model_n + training_code, train_acc)
        np.save(path_checkpoint + "dev_loss" + model_n + training_code, dev_loss)
        np.save(path_checkpoint + "dev_acc" + model_n + training_code, dev_acc)
        torch.save(model.state_dict(), path_checkpoint + model_name + ".pt")
    
    #finished training:
    print("saving results in folder...")
    np.save(path_finished + "train_loss" + model_n + training_code, train_loss)
    np.save(path_finished + "train_acc" + model_n + training_code, train_acc)
    np.save(path_finished + "dev_loss" + model_n + training_code, dev_loss)
    np.save(path_finished + "dev_acc" + model_n + training_code, dev_acc)
      
    print("saving model in folder")
    torch.save(model.state_dict(), path_finished + model_name + ".pt")
    torch.save(model.encoder.state_dict(), path_finished + model_name + "enc.pt")
    
    ###########################################################################
    #final test on the model
    test_loss = 0
    test_acc = 0
    model.eval()
    for batch in batch_iters["test"]:
        x_pre = batch.premise
        x_hyp = batch.hypothesis
        y = batch.label
               
        #perform forward pass
        y_pred = model.forward(x_pre, x_hyp)        
        
        loss_t = loss_func(y_pred, y)
        
        #get metrics
        test_loss += loss_t.item()/y.shape[0]
        test_acc += accuracy(y_pred, y)/test_batches
        
        #avoid memory issues
        y_pred.detach()
        
    #print eval results
    print(f"TEST acc: {test_acc}, loss: {test_loss}")  
    
    #save test results
    np.save(path_finished + "test_loss" + model_n + training_code, test_loss)
    np.save(path_finished + "test_acc" + model_n + training_code, test_acc)
    
            
    return train_acc, train_loss, dev_acc, dev_loss

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
        
    # Print all Flags to confirm parameter settings
    print_flags(FLAGS)
    
    
    #create directories if they don't exist
    check_dir(FLAGS.train_data_path)
    check_dir(FLAGS.checkpoint_path)  
    
    print("Step 2 of project Take Over the World: Read Human.")
    #time the training time
    time_start = time.time()
    
    #does the actuall training
    train_acc, train_loss, dev_acc, dev_loss = train()
      
    time_end = time.time()
    print(f"Training took {time_end - time_start} seconds") 
    print("Training finished successfully. \nNote to self, humanity is confusing.")
    print("Ask help to ELMo...")
    
if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type = str, default = MODEL_TYPE_DEFAULT,
                          help='model type')
    parser.add_argument('--model_name', type = str, default = MODEL_NAME_DEFAULT,
                          help='model name: "mean", "unilstm", "bilstm" or "maxlstm"')
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
    parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                          help='size of mini batch')
    parser.add_argument('--max_epochs', type = int, default = MAX_EPOCHS_DEFAULT,
                          help='max number of epochs')
    parser.add_argument('--weight_decay', type = float, default = WEIGHT_DECAY_DEFAUT,
                          help='weight_decay for the optimizer')
    parser.add_argument('--plot_grad', type = bool, default = PLOT_GRAD_DEFAULT,
                        help= 'True to plot gradient flow, False not to.')
    FLAGS, unparsed = parser.parse_known_args()
    
    main()
    