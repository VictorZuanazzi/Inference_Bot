# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 23:22:51 2019

@author: Victor Zuanazzi
"""

##test transfer learning

#import stuff
from __future__ import absolute_import, division, unicode_literals
import sys
import os
import torch
import logging

#local imports
from data_2 import get_embeddings
from encoder import MeanEncoder, UniLSTM, BiLSTM, MaxLSTM

# Set global variables
PATH_SENTEVAL = './SentEval'
PATH_TO_DATA = './SentEval/data'
PATH_TO_W2V = './data/glove_embedding/glove.840B.300d.txt',                               
MODEL_PATH = './train/InferClassifier_type_mean_.pt'
LEARNING_RATE_DEFAULT = 0.1
BATCH_SIZE_DEFAULT = 64
MAX_EPOCHS_DEFAULT = 2
OPTIMIZER_DEFAULT = 'adam'
DATA_DIR_DEFAULT = './data/'
MODEL_TYPE_DEFAULT = 'base_line'
MODEL_NAME_DEFAULT =  'unilstm' #'unilstm' #'maxlstm'#'bilstm'# #'mean'
TRAIN_DIR_DEFAULT = './train/'
CHECKOUT_DIR_DEFAULT = './checkout/'
DEVICE_DEFAULT = 'cpu'
DEVICE = DEVICE_DEFAULT
DATA_PERCENTAGE_DEFAULT =.001
WEIGHT_DECAY_DEFAUT = 0.0

# import senteval
sys.path.insert(0, PATH_SENTEVAL)
import senteval

#set device
DEVICE = torch.device('cpu')


def load_encoder(enc_name='mean', path="./train/"):
    
    enc_name = enc_name.lower()
    if enc_name == 'mean':
        #executes baseline model
        encoder = MeanEncoder()
        name = "InferClassifier_type_mean_enc.pt"
    elif enc_name == 'unilstm':
        #Uni directional LSTM
        encoder = UniLSTM()
        name = "InferClassifier_type_unilstm_enc.pt"
    elif enc_name == 'bilstm':
        encoder = BiLSTM()
        name = "InferClassifier_type_bilstm_enc.pt"
    elif enc_name == 'maxlstm':
        encoder = MaxLSTM()
        name = "InferClassifier_type_maxilstm_enc.pt"
        
    encoder.load_state_dict(torch.load(path+name))
    
    return encoder

def prepare(params, samples):
    params.infersent.build_vocab([' '.join(s) for s in samples], tokenize=False)

def batcher(params, batch):
    sentences = [' '.join(s) for s in batch]
    embeddings = params.infersent.encode(sentences, bsize=params.batch_size, tokenize=False)
    return embeddings

def main():
    global DEVICE
    if FLAGS.torch_device == 'cuda': 
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        DEVICE = torch.device('cpu')
    
    # Load InferSent model
    encoder = load_encoder(enc_name='mean')
    
    # define senteval params
    params_senteval = {'task_path': PATH_TO_DATA, 
                       'usepytorch': True, 
                       'kfold': 5}
    
    params_senteval['classifier'] = {'nhid': 0, 
                   'optim': 'rmsprop', 
                   'batch_size': 128,
                   'tenacity': 3,
                   'epoch_size': 2}

    params_senteval['infersent'] = encoder.to(DEVICE)

    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                      'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
                      'SICKEntailment', 'SICKRelatedness', 'STSBenchmark',
                      'Length', 'WordContent', 'Depth', 'TopConstituents',
                      'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
                      'OddManOut', 'CoordinationInversion']
    results = se.eval(transfer_tasks)
    print(results)


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type = str, default = MODEL_NAME_DEFAULT,
                          help='model name: "mean", "unilstm", "bilstm" or "maxlstm"')
    parser.add_argument('--train_data_path', type = str, default = TRAIN_DIR_DEFAULT,
                          help='Directory for storing train data')
    parser.add_argument('--checkpoint_path', type = str, default = CHECKOUT_DIR_DEFAULT,
                          help='Directory of check point')
    parser.add_argument('--torch_device', type = str, default = DEVICE_DEFAULT,
                          help='torch devices types: "cuda" or "cpu"')
    parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                          help='size of mini batch')
    FLAGS, unparsed = parser.parse_known_args()
    
    main()