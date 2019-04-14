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
    """uses the sentences to create a vocabulary from sentences"""
    
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
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    