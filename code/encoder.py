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
    
    def build_vocab(self):
        return get_embeddings()
        
    
def batch_to_sequence():
    pass

def sequence_to_batch():
    pass
    
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
        
        
        #sort data because pack_padded is too stupid to do it itself
        #you can answer this after figuring the sorting out:
        #https://stackoverflow.com/questions/49203019/how-to-use-pack-padded-sequence-with-multiple-variable-length-input-with-the-sam
        #helpful

        len_x, idx = len_x.sort(0, descending=True)
        x = x[:,idx]
              
        #remove paddings before feeding it to the LSTM
        x = torch.nn.utils.rnn.pack_padded_sequence(x, 
                                                    len_x, 
                                                    batch_first = self.batch_first)
        
        #run LSTM, 
        x, (_, _) = self.uni_lstm(x)
        
        #re-introduce the paddings
        #doc pack_padded_sequence:
        #https://pytorch.org/docs/master/nn.html#torch.nn.utils.rnn.pack_padded_sequence
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, 
                                                      batch_first = self.batch_first)
        
        
        #get the indexes of the last token (where the lstm should stop)
        longest_sentence = max(len_x)
        last_word = [i*longest_sentence + len_x[i] for i in range(len(len_x))]
        
        #get the relevant hidden states
        x = x.view(-1, self.hidden_dim)
        x = x[last_word,:]
        
#        print(f"2 shapes x= {x.shape}")
        #unsort the batch!
        _, idx = idx.sort(0, descending=False)
        x = x[idx, :]
        
        
        
        

#*******************this part is probably outdated.
#        print(f"3 shapes x= {x.shape}")
#        #reshape the data ito the correct output dimension
        #last_hidden = last_hidden.view(-1, self.hidden_dim)
#        print(f"3 shape x: {x.shape}")
#        
        return x

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
        
        #run LSTM, 
        #we are just interested in the last hidden state
        _, (last_hidden, _) = self.bi_lstm(x)
        
        #concatenate both directons of the hidden dimention
        last_hidden = last_hidden.view(1, 2, -1, self.hidden_dim)
        last_hidden = torch.cat((last_hidden[:,0], last_hidden[:,1]), dim=2).view(-1, 2*self.hidden_dim)
        
        return last_hidden


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
        #run LSTM, 
        #we want all hidden states
        x, (_, _) = self.bi_lstm(x)
        
        #get the max value for each dimension
        x, _ = torch.max(x, dim=0)
        
        return x





    
