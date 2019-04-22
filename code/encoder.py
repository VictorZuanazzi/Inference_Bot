# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 16:43:20 2019

@author: Victor Zuanazzi
"""
#heavy duty libraries
import torch
import torch.nn as nn
import numpy as np

#import local libraries
from data_2 import get_embeddings



class MeanEncoder(nn.Module):

    def __init__(self, embedding_dim=300, hidden_dim=300, batch_size = 64):
        """initializes the mean encoder"""
        
        super(MeanEncoder, self).__init__()
        puns = {0: "Hey, I am your baseline. I am not good, I am not bad, I just average!",
                1: "I am not sure you get what I mean.",
                2: "Those sentences are so mean...",
                3: "Three words entered a bar, I don't know which one was first...",
                4: "I like ambiguity more than most people.",
                }
        print(puns[np.random.randint(len(puns))])
        #Those are only neeced to make the code modular
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.batch_first = False
        self.output_size = hidden_dim
        
    def forward(self, x, x_len):
        """ mean forward pass  """    

        out = torch.div(torch.sum(x, dim=0), 
                        x_len.view(-1, 1).to(torch.float))
        
        return out
        
    
def batch_to_sequence(x, len_x, batch_first):
    """helpful function to do the pack padding shit
    returns the pack_padded sequence, whatever that is.
    The data does NOT have to be sorted by sentence lenght, we do that for you!
    Input:
        x: (torch.tensor[max_len, batch, embedding_dim]) tensor containing the  
            padded data. It expects the embeddings of the words in the sequence 
            they happen in the sentence.  If batch_first == True, then the 
            max_len and batch dimensions are transposed.
        len_x: (torch.tensor[batch]) a tensor containing the length of each 
            sentence in x.
        batch_first: (bool), indicates whether batch or sentence lenght are 
            indexed in the first dimension of the tensor.
    Output:
        x: (torch pack padded sequence) a the pad packed sequence containing 
            the data. (The documentation is horrible, I don't know what a 
            pack padded sequence really is.)
        idx: (torch.tensor[batch]), the indexes used to sort x, this index in 
            necessary in sequence_to_batch.
        len_x: (torch.tensor[batch]) the sorted lenghs, also needed for 
            sequence_to_batch."""
    
    #sort data because pack_padded is too stupid to do it itself
    #you can answer this after figuring the sorting out:
    #https://stackoverflow.com/questions/49203019/how-to-use-pack-padded-sequence-with-multiple-variable-length-input-with-the-sam
    len_x, idx = len_x.sort(0, descending=True)
    x = x[:,idx]
          
    #remove paddings before feeding it to the LSTM
    x = torch.nn.utils.rnn.pack_padded_sequence(x, 
                                                len_x, 
                                                batch_first = batch_first)
    
    return x, len_x, idx

def sequence_to_batch(x, len_x, idx, output_size, batch_first, all_hidden = False):
    """helpful function for the pad packed shit.
    Input:
        x: (packed pad sequence) the ouptut of lstm  or pack_padded_sequence().
        len_x (torch.tensor[batch]), the sorted leghths that come out of 
            batch_to_sequence().
        idx: (torch.tenssor[batch]), the indexes used to sort len_x
        output_size: (int), the expected dimension of the output embeddings.
        batch_first: (bool), indicates whether batch or sentence lenght are 
            indexed in the first dimension of the tensor.
        all_hidden: (bool), if False returs the last relevant hidden state - it 
            ignores the hidden states produced by the padding. If True, returs
            all hidden states.
    Output:
        x: (torch.tensor[batch, embedding_dim]) tensor containing the  
            padded data.           
    """
    
    #re-introduce the paddings
    #doc pack_padded_sequence:
    #https://pytorch.org/docs/master/nn.html#torch.nn.utils.rnn.pack_padded_sequence
    x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, 
                                                  batch_first = batch_first)
    if all_hidden:
        return x
    
    #get the indexes of the last token (where the lstm should stop)
    longest_sentence = max(len_x)
    #subtracsts -1 to see what happens
    last_word = [i*longest_sentence + len_x[i]-1 for i in range(len(len_x))]
    
    #get the relevant hidden states
    x = x.view(-1, output_size)
    x = x[last_word,:]

    #unsort the batch!
    _, idx = idx.sort(0, descending=False)
    x = x[idx, :]
    
    return x

class UniLSTM(nn.Module):
    
    def __init__(self, embedding_dim=300, hidden_dim=2048, batch_size=64):
        
        super(UniLSTM, self).__init__()
        puns = {0: "Get yourself a direction and don't even look back.",
                1: "Sequence matters, average splatters",
                2: "Three words entered a bar, and the last one is more important than the first one.",
                }
        print(puns[np.random.randint(len(puns))])
        
        #start stuff
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.batch_first = False
        self.output_size = hidden_dim
        
        self.uni_lstm = nn.LSTM(input_size = self.embedding_dim, 
                                hidden_size = self.hidden_dim,
                                batch_first = self.batch_first)
        
    def forward(self, x, len_x):
        
        #convert batch into a packed_pad sequence
        x, len_x, idx = batch_to_sequence(x, len_x, self.batch_first)
        
        #run LSTM, 
        x, (last_hidden, _) = self.uni_lstm(x)
        
#        #out = last_hidden.view(-1, self.output_size)
#        longest_sentence = max(len_x)
#        last_word = [i*longest_sentence + len_x[i]-1 for i in range(len(len_x))]
#    
#        #get the relevant hidden states
#        x = x.view(-1, self.output_size)
#        out = x[last_word,:]
                
        #takes the pad_packed_sequence and gives you the embedding vectors
        out = sequence_to_batch(x, len_x, idx, self.output_size, self.batch_first)        
        return out

class BiLSTM(nn.Module):
    
    def __init__(self, embedding_dim=300, hidden_dim=2048, batch_size=64):
        
        super(BiLSTM, self).__init__()
        
        #puns for initializing the model
        puns = {0: "When crossing the road, it is important to look both sides.",
                1: "Is here and there, there and here?",
                2: "Three words entered a bar, which one was the first again?",
                }
        print(puns[np.random.randint(len(puns))])
        
        #start stuff
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.batch_first = False
        self.output_size = 2*hidden_dim
        
        self.bi_lstm = nn.LSTM(input_size = self.embedding_dim, 
                                hidden_size = self.hidden_dim,
                                batch_first = self.batch_first,
                                bidirectional = True)
        
    def forward(self, x, len_x):
        
        #convert batch into a packed_pad sequence
        x, len_x, idx = batch_to_sequence(x, len_x, self.batch_first)
        
        #run LSTM, 
        x, (last_hidden, _) = self.bi_lstm(x)
        
#        #out = last_hidden.view(-1, self.output_size)
#        longest_sentence = max(len_x)
#        last_word = [i*longest_sentence + len_x[i]-1 for i in range(len(len_x))]
#    
#        #get the relevant hidden states
#        x = x.view(-1, self.output_size)
#        out = x[last_word,:]
        
        #takes the pad_packed_sequence and gives you the embedding vectors
        out = sequence_to_batch(x, len_x, idx, self.output_size, self.batch_first)
        
        return out


class MaxLSTM(nn.Module):
    
    def __init__(self, embedding_dim=300, hidden_dim=2048, batch_size=64):
        
        super(MaxLSTM, self).__init__()

        puns = {0: "Lets make the Max out of those sentences.",
                1: "Let's play some pool...",
                2: "What a fabulous weather, it is pool time!",
                3: "Three words entered a bar, or was it just one very fat and tall word?",
                }
        print(puns[np.random.randint(len(puns))])
        
        #start stuff
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.batch_first = False
        self.output_size = 2*hidden_dim
        
        self.bi_lstm = nn.LSTM(input_size = self.embedding_dim, 
                                hidden_size = self.hidden_dim,
                                batch_first = self.batch_first,
                                bidirectional = True)
        
    def forward(self, x, len_x):
        
        #convert batch into a packed_pad sequence
        #x, len_x, idx = batch_to_sequence(x, len_x, self.batch_first)
        
        #run LSTM, 
        #we want all hidden states
        x, (_, _) = self.bi_lstm(x)
        
        #takes the pad_packed_sequence and gives you the embedding vectors
        #x = sequence_to_batch(x, len_x, idx, self.output_size, self.batch_first, True)
        
        #get the max value for each dimension
        #not perfect, but will do for now. The paddings can be retrieved in case
        #all other hidden states have negative values
        x, _ = torch.max(x, dim=0)
        
        return x





    
