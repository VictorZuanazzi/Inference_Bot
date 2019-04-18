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
import io
import torch
import logging
import argparse
import numpy as np
import time

#local imports
from data_2 import get_embeddings
from encoder import MeanEncoder, UniLSTM, BiLSTM, MaxLSTM

# Set global variables
PATH_SENTEVAL = './SentEval'
PATH_TO_DATA = './SentEval/data'
PATH_TO_W2V = './data/glove_embedding/glove.840B.300d.txt',                               
MODEL_PATH = './train/InferClassifier_type_mean_.pt'
PATH_TO_VEC = './data/glove_embedding/glove.840B.300d.txt'
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
INCLUDE_TASKS_DEFAUT = 'one'

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

# Create dictionary
def create_dictionary(sentences, threshold=0):
    """function that creates a dictionary, stollen form SentEval"""
    words = {}
    for s in sentences:
        for word in s:
            words[word] = words.get(word, 0) + 1

    if threshold > 0:
        newwords = {}
        for word in words:
            if words[word] >= threshold:
                newwords[word] = words[word]
        words = newwords
    words['<s>'] = 1e9 + 4
    words['</s>'] = 1e9 + 3
    words['<p>'] = 1e9 + 2

    sorted_words = sorted(words.items(), key=lambda x: -x[1])  # inverse sort
    id2word = []
    word2id = {}
    for i, (w, _) in enumerate(sorted_words):
        id2word.append(w)
        word2id[w] = i

    return id2word, word2id

# Get word vectors from vocabulary (glove, word2vec, fasttext ..)
def get_wordvec(path_to_vec, word2id):
    """function that returns the embeddings, stollen form SentEval"""
    word_vec = {}

    with io.open(path_to_vec, 'r', encoding='utf-8') as f:
        # if word2vec or fasttext file : skip first line "next(f)"
        for line in f:
            word, vec = line.split(' ', 1)
            if word in word2id:
                word_vec[word] = np.fromstring(vec, sep=' ')

    logging.info('Found {0} words with word vectors, out of \
        {1} words'.format(len(word_vec), len(word2id)))
    return word_vec


# SentEval prepare and batcher
def prepare(params, samples):
    _, params.word2id = create_dictionary(samples)
    params.word_vec = get_wordvec(PATH_TO_VEC, params.word2id)
    params.wvec_dim = 300
    return

def batcher(params, batch):
    batch = [sent if sent != [] else ['.'] for sent in batch]
    embeddings = []

    for sent in batch:
        sentvec = []
        for word in sent:
            if word in params.word_vec:
                sentvec.append(params.word_vec[word])
        if not sentvec:
            vec = np.zeros(params.wvec_dim)
            sentvec.append(vec)
        sentvec = np.mean(sentvec, 0)
        embeddings.append(sentvec)

    embeddings = np.vstack(embeddings)
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
    include_tasks = FLAGS.include_tasks.lower()
    if include_tasks == 'all':
        transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                          'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
                          'SICKEntailment', 'SICKRelatedness', 'STSBenchmark',
                          'Length', 'WordContent', 'Depth', 'TopConstituents',
                          'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
                          'OddManOut', 'CoordinationInversion']
    elif (include_tasks == 'few'):
        transfer_tasks = ['MR', 'CR', 'MPQA']
    else:
        transfer_tasks = ['MR']
    
    start = time.time()
    results = se.eval(transfer_tasks)
    print(results)
    print(f"Test took {time.time() - start} s")


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
    parser.add_argument('--include_tasks', type = str, default = INCLUDE_TASKS_DEFAUT,
                        help='decide how many tests to include: "one", "few", "all"')
    FLAGS, unparsed = parser.parse_known_args()
    
    main()