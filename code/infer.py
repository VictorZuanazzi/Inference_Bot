# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 12:48:31 2019

Load trained model and iteract with the user.
Ask for a hypothesis. Ask for an Entailment. Classify the pair as an 
enteilement, contradiction or neutral (using the models)

@author: Victor Zuanazzi
"""

from data_2 import get_embeddings, load_data
from nltk import word_tokenize
import torch
import numpy as np

from model import InferClassifier
from encoder import MeanEncoder, UniLSTM, BiLSTM, MaxLSTM
from utils import load_encoder, path_n_name, load_classifier

def sentence_to_idx(sentence, w2i_dict):
    
    #tokenize the sentence
    sentence = word_tokenize(sentence)
    
    #get the index of each word
    indexes = torch.tensor([w2i_dict[word] for word in sentence])
    
    return indexes

def infer_bot(encoder_type='maxlstm'):
    #get embeddings:
    start_messages= np.array(["Please be patient, this GloVe is heavy!",
                              "Loading GloVe...", 
                              "It fits like a GloVe (it just takes a while...",
                              "... ... .. ... ...", 
                              "While you wait, why don't you google if pinguins have knews?",
                              "While you wait... Why are there no mamals that are green?",
                              "Whiie you wait, have you wondered on what are the main drivers for the pay gap between men and women",
                              "While you wait, google tiny turtle eating a strawberry."])
                              
    print(f"{np.random.choice(start_messages, 1)}")
    
    _, text_f, _ = load_data()
    
    #possible entailments
    entailment= {0: "entails",1: "contradicts", 2: "is neutral to"}
    
    ask_premise = np.array(["Give me a premise, please.", 
                   "Would you be so kind as to provide me of a premise?", 
                   "Premise, NOW!", "What is your premise my dear?",
                   "Tell me something.", "P R E M I S E !"])
    
    ask_hypothesis = np.array(["Give me a hypothesis, please.", 
                   "Would you be so kind as to provide me of a hypothesis?", 
                   "Hyposthesis, NOW!", "What is your hypothesis my dear?", 
                   "What do you want to know about the premise?", 
                   "Be careful with what you want to infer from the premise!",
                   "To hypothetize is a cognitive task...",
                   "What do you want to infer from that?", 
                   "H Y P O T H E S I S !"])
    
    
    model = load_classifier(text_f.vocab.vectors, 
                            encoder_type = encoder_type)
    
    one_more = True
    while one_more:
        
        premise= input(f"{np.random.choice(ask_premise, 1)} >>> ")
        premise = sentence_to_idx(premise, text_f.vocab.stoi)
        p_len = len(premise)
        
        
        hypothesis = input(f"{np.random.choice(ask_hypothesis, 1)} >>> ")
        hypothesis = sentence_to_idx(hypothesis, text_f.vocab.stoi)
        h_len = len(hypothesis)
        
        y_pred = model.forward((premise.expand(1,-1).transpose(0,1), p_len),
                           (hypothesis.expand(1,-1).transpose(0,1), h_len))
        
        print(f"\n The premise {entailment[y_pred.argmax().item()]} the hypothesis. \n")
        
        if input("Do you want to keep playing? [y/n] ") == 'n':
            one_more = False
    
    
def main(encoder_type='maxlstm'):
    #this main is totally useless!
    infer_bot(encoder_type='maxlstm')
    

if __name__ == '__main__':
    main()

