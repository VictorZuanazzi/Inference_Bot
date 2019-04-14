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
import torch
import torch.optim as optim
from utils import print_flags


# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'
DATA_DIR_DEFAULT = './data/'
MODEL_TYPE_DEFAULT = 'base_line'
MODEL_NAME_DEFAULT = 'mean'
TRAIN_DIR_DEFAULT = './train/'
CHECKOUT_DIR_DEFAULT = './checkout/'
DEVICE_DEFAULT = 'cpu'

#set datatype to torch tensor
dtype = torch.FloatTensor

FLAGS = None

#use GPUs if available
if FLAGS.torch_device == 'cuda':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device =torch.device('cpu')
    


def train():
    """train model on inference task"""
    pass

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
    # Print all Flags to confirm parameter settings
    print_flags()
    
    #create directories if they don't exist
    if not os.path.exists(FLAGS.train_data_path):
        os.makedirs(FLAGS.train_data_path)
    
    
    print("Step 2 of project Take Over the World:\nRead Human.")
    
    if FLAGS.model_name == 'mean':
        #executes baseline model
      
    print("Training finished successfully. \nNote to self, humanity is confusing.")

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type = float, default = MODEL_TYPE_DEFAULT,
                          help='model type')
    parser.add_argument('--model_name', type = int, default = MODEL_NAME_DEFAULT,
                          help='model name: "mean",')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                            help='Frequency of evaluation on the dev set')
    parser.add_argument('--train_data_path', type = str, default = TRAIN_DIR_DEFAULT,
                          help='Directory for storing train data')
    parser.add_argument('--checkpoint_path', type = str, default = CHECKOUT_DIR_DEFAULT,
                          help='Directory of check point')
    parser.add_argument('--torch_device', type = str, default = DEVICE_DEFAULT,
                          help='torch devices types: "cuda" or "cpu"')
    FLAGS, unparsed = parser.parse_known_args()
    
    main()