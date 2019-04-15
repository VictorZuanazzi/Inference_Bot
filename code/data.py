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

def filter_dash(data=False):
    return data

def get_snli():
    """feches the snli dataset.
    The attributes 'hypothesis_transitions' and 'premise_transitions' are 
        droped from the dataset. 
    The class '-' is also deleted from the data.
    Output:
        dictionary containing the datasets with keys 'train', 'dev', 'test'
    """
    
    data={}
    clean_data ={"train": [], "dev": [], "test": []}
    
    #fetch data from snli
    data["train"], data["dev"], data["test"] = snli_dataset(train = True,
                                                            dev = True,
                                                            test = True)
    
    for d_set in data.keys():
        
        len_data = len(data[d_set])   
        for i in range(len_data):
            
            #excludes class '-'
            if data[d_set][i]["label"] != "-":
                
                #excludes the unecessary transitions
                #reduces the memory cost for processing the data later.
                data[d_set][i].pop('hypothesis_transitions')
                data[d_set][i].pop('premise_transitions')   
                
                #stores data after cleaning.
                clean_data[d_set].append(data[d_set][i])
       
    return clean_data
    
    
def vocab_from_snli(data=False):
    """builds a vocabulary from snli data.
    Input: 
        data: (bool=False or torchnlp.datasets.Dataset), if False it returns the
            vocab for the training set of snli. If a dataset is given, it 
            returns the vocab for the dataset. 
    Output:
        Same output as from vocab_embeddings()
    """    
    if not data:
        data = get_snli()
        data.pop('dev')
        data.pop('test')
        data = data["train"]
    
    #build a list with all sentences
    all_sentences = []
    for d in data:
        all_sentences.append(d["premise"])
        all_sentences.append(d["hypothesis"])
    
    #returns the vocab dict for all sentences.
    return vocab_embeddings(all_sentences)
    
        
def split_snli(data=False):
    """split snli in three lists, premises, hypothesis and labels.
    Input: 
        data: (bool=False or torchnlp.datasets.Dataset), if False it returns the
            vocab for the training set of snli. If a dataset is given, it 
            returns the vocab for the dataset. 
    Output:
        the index are shared among the lists.
        premises: (list(str)), the premises in the snli dataset
        hypothesis: (list(str)), the hypothsis in the snli dataset
        labels: (list(str)), the labels in the snli dataset   
    """
    
    #get training data
    if not data:
        data = get_snli()
        data.pop('dev')
        data.pop('test')
        data = data["train"]
    
    premises = []
    hypothesis = []
    #class "-" was filtered out, but it is included in the dict 
    l_dict ={"entailment": 0, "neutral": 1, "contradiction": 2 , "-":1}
    labels =[]
    
    #create the lists.
    for d in data:
        premises.append(d["premise"])
        hypothesis.append(d["hypothesis"])
        labels.append(l_dict[d["label"]])
        
    return premises, hypothesis, labels


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    