# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 11:47:56 2019

This file contains functionalities to:
    Load the data
    Pre-process the data (torchnlp.datasets)
    batch the data for training


@author: Victor Zuanazzi
"""

from torchnlp.word_to_vector import GloVe
from torchnlp.datasets.snli import snli_dataset
import os
import numpy as np
import torch

def get_embeddings(words, embedding_type='840B'):
    """get the word embeddings requested in words.
    Embeddings of unknown words are a tensor of zeros.
    Input:
        words: (str or list(str) or dict{str:_}), the words of interest.
        embedding_type (str), the Glove type of embeddings. Only GloVe is 
            supported for now.
    Returns:
        dict{str: torch.tensor} returs a dictionary that maps the words to 
            their embeddings.
    """
    
    #convert the input words into a dictionary
    if type(words) == str:
        words = {words: None}
    
    #loads glove vectors into memory
    glove_vectors = GloVe(embedding_type)
        
    #returns the dictiorary with the resquested words
    return {w: glove_vectors[w] for w in words}

def create_vocab(sentences, case_sensitive=True):
    """uses the sentences to create a dict vocabulary from sentences
    Input:
        sentences (iterable(str)), an itererable with the sentences in string format.
        case_senstive (bool), in case True, the words in the sentence will be 
            returned as is. If False, it will lower case all words in the 
            sentence.
    Output:
        dict(str: None) a dictionary with all the words in the sentences.
    """
    
    #in case sentences is one string
    if type(sentences) == str:
        sentences = [sentences]
    
    #puts all sentences into one giant sentence
    #words are lower cased to avoid 
    sentences = " ".join(sentences)
    if not case_sensitive:
        #all words are lower cased
        sentences = sentences.lower()
    
    #create a set with all individual words
    #the set makes sure duplicated words are dealt with efficiently.
    vocab = set(sentences.split())
    
    #vocab is converted into a dictionary
    vocab = {word: None for word in vocab}
    
    #auxiliar words
    vocab['<s>'] = None #start sentence
    vocab['</s>'] = None #end sentence
    vocab['<p>'] = None #padding
    vocab['<unk>'] = None #unknown word
            
    return vocab
            
    
def vocab_embeddings(sentences, case_sensitive=True, embedding_type='840B'):
    """vocabulary with embeddings from sentences
    Input:
        sentences: (iterable(str)) an iterable containing the sentences.
        case_sensitive: (bool) if False all words are returned lowered cased, 
            if True they remain as is.
        embedding_type: (str) only '840B' is supported."""
    
    #get a dictionary with all the words in the sentences
    vocab = create_vocab(sentences, case_sensitive=case_sensitive)
    
    #get a dictionary with the embeddings for all the words
    vocab = get_embeddings(vocab, embedding_type=embedding_type)
    
    return vocab

def get_snli(transitions=False):
    """feches the snli dataset.
    Input:
        transitions (bool), if False, the attributes 'hypothesis_transitions' 
            and 'premise_transitions' are droped from the dataset. If True they
            are kept untouched.
    Output:
        dictionary containing the datasets in the calss torchnlp.datasets.Dataset
            valid keys are 'train', 'dev', 'test'"""
    data={}
    
    #fetch data from snli
    data["train"], data["dev"], data["test"] = snli_dataset(train = True,
                                                            dev = True,
                                                            test = True)
    if transitions:
        return data
    
    #excludes the unecessary transitions
    #reduces the memory cost for processing the data later.
    for d_set in data.keys():
        for i in range(len(data[d_set])):
            data[d_set][i].pop('hypothesis_transitions')
            data[d_set][i].pop('premise_transitions')
            
    return data
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    