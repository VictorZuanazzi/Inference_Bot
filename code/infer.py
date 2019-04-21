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
from utils import load_encoder

def sentence_to_idx(sentence, w2i_dict):
    
    #tokenize the sentence
    sentence = word_tokenize(sentence)
    
    #get the index of each word
    indexes = torch.tensor([w2i_dict[word] for word in sentence])
    
    return indexes

def load_classifier(text_f):
    encoder = MaxLSTM()
    path = './train/MaxLSTM/20190421/'
    m_name = "InferClassifier_type_maxlstm__.pt"
    e_name = "InferClassifier_type_maxlstm__enc.pt"
    
    encoder.load_state_dict(torch.load(path+e_name))
    
    model = InferClassifier(input_dim= 4 * encoder.output_size,
                            n_classes=3, 
                            encoder = encoder, 
                            matrix_embeddings = text_f.vocab.vectors)
    
    model.load_state_dict(torch.load(path+m_name))
    
    return model

    
def main():
    
    #get embeddings:
    _, text_f, _ = load_data()
    
    #possible entailments
    entailment= {0: "entails",1: "contradicts", 2: "is neutral to"}
    ask_premise = np.array(["Give me a premise, please.", 
                   "Would you be so kind as to provide me of a premise?", 
                   "Premise, NOW!", "What is your premise my dear?",
                   "Tell me something."])
    
    ask_hypothesis = np.array(["Give me a hypothesis, please.", 
                   "Would you be so kind as to provide me of a hypothesis?", 
                   "Hyposthesis, NOW!", "What is your hypothesis my dear?", 
                   "What do you want to know about the premise?", 
                   "Be careful with what you want to infer from the premise!",
                   "To hypothetize is a cognitive task...",
                   "What do you want to infer from that?"])
    
    
    model = load_classifier(text_f)
    
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
        
        print(f"The premise {entailment[y_pred.argmax().item()]} the hypothesis")
        
        if input("Do you want to keep playing? [y/n] ") == 'n':
            one_more = False

    
if __name__ == '__main__':
    main()

