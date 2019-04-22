# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 10:56:54 2019

@author: Victor Zuanazzi
"""
import torch
import os
import numpy as np

from model import InferClassifier
from encoder import MeanEncoder, UniLSTM, BiLSTM, MaxLSTM

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
    '''Plots the gradients flowing through different layers in the net during 
    training. Can be used for checking for possible gradient vanishing or 
    exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient 
    flow.
    
    Source: https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/10
    '''
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

def path_n_name(enc_name='maxlstm', path=None, m_name=None, e_name=None):
    """Centralized path and file names, returns the inputs that are not specified
    for regarding the enc_name.
    Input:
        enc_name: (str) name of the encoder to be loaded. The options are 'mean',
            'unilstm', 'bilstm' and 'maxlstm', the last one is executed if 
            no valid encoder name is given.
        path: (str) path to the encoder. If None is given, then the latest path
            is used.
        e_name: (str) the file name of the pre-trained encoder. If None is 
            given, the latest file is used.
    Output:
        path: (str) the path to the files.
        m_name: (str) the classifier name file name. 
        e_name (str) the encoder file name.
    """
    enc_name = enc_name.lower()
    
    if enc_name == 'mean':
        #latest path is used if other not specified.
        if not path:
            path = "./train/baseline/20190418/"
            
        #latest encoder is used if other not specified.
        if not e_name:
            e_name = "InferClassifier_mean_type_mean__enc.pt"
            
        #latest classifier used if None is given
        if not m_name:
            m_name = "InferClassifier_mean_type_mean__.pt"
            
    elif enc_name == 'unilstm':
        #latest path is used if other not specified.
        if not path:
            path = "./train/"
            
        #latest encoder is used if other not specified.
        if not e_name:
            e_name = "InferClassifier_type_unilstm__enc.pt"
            
        #latest classifier is used if other not specified.
        if not m_name:
            m_name = "InferClassifier_type_unilstm__.pt"
            
    elif enc_name == 'bilstm':
        #latest path is used if other not specified.
        if not path:
            path = "./train/"
            
        #latest encoder is used if other not specified.
        if not e_name:
            e_name = "InferClassifier_type_bilstm__enc.pt"
            
        #latest classifier is used if other not specified.
        if not m_name:
            m_name = "InferClassifier_type_bilstm__.pt"
    else:
        #latest path is used if other not specified.
        if not path:
            path = './train/MaxLSTM/20190421/'
            
        #latest encoder is used if other not specified.
        if not e_name:
            e_name = "InferClassifier_type_maxlstm__enc.pt"
        
        #latest classifier is used if other not specified.
        if not m_name:
            m_name = "InferClassifier_type_maxlstm__.pt"
            
    return path, m_name, e_name
       
def load_encoder(enc_name='maxlstm', path=None, e_name=None):
    """Loads the correct encoder.
    Input:
        enc_name: (str) name of the encoder to be loaded. The options are 'mean',
            'unilstm', 'bilstm' and 'maxlstm', the last one is executed if 
            no valid encoder name is given.
        path: (str) path to the encoder. If None is given, then the latest path
            is used.
        e_name: (str) the file name of the pre-trained encoder. If None is 
            given, the latest file is used.
    Output: 
        Pretrained encoder from encoder.py"""
        
    #lower case to avoid user issues.
    enc_name = enc_name.lower()
    
    #get path and file nime
    path, _, e_name = path_n_name(enc_name = enc_name,
                                       path = path,
                                       m_name = None,
                                       e_name = e_name)
    
    if enc_name == 'mean':
        #executes baseline model
        encoder = MeanEncoder()
            
    elif enc_name == 'unilstm':
        #Uni directional LSTM
        encoder = UniLSTM()
            
    elif enc_name == 'bilstm':
        #Bidirectional LSTM
        encoder = BiLSTM()
        
    else:
        #standard option is the MaxLSTM
        encoder = MaxLSTM()
        
    encoder.load_state_dict(torch.load(path+e_name))
    
    return encoder        
        

def load_classifier(embedding_matrix, encoder_type='maxlstm',  path=None, m_name=None, e_name=None):
    """Loads the pretrained classifier of class model.InferClassifier.
    Input:
        embedding_matrix: (torch.tensor) the embedding matrix used for training 
            the classifier.
        encoder_type: (str), the encoder, check utils.load_encoder for options.
        path: (srt), path to the classifier. It assumes that the classifier and
            the encoder are stored in the same folder.
        m_name: (str), file name of the pretrained model.
        e_name: (str), file name of the pretrained encoder.
    Output:
        Pretrained classifier of class model.InferClassifier
            """
    
    #Loads the encoder
    encoder = load_encoder(enc_name = encoder_type, 
                           path=path, 
                           e_name=e_name)
    
    #Loads the model class
    model = InferClassifier(input_dim= 4 * encoder.output_size,
                            n_classes=3, 
                            encoder = encoder, 
                            matrix_embeddings = embedding_matrix)
    
    #get path and file name
    path, m_name, _ = path_n_name(enc_name = encoder_type,
                                  path = path, 
                                  m_name = m_name, 
                                  e_name = e_name)
    
    #finally, loads the model
    model.load_state_dict(torch.load(path+m_name))
    
    return model