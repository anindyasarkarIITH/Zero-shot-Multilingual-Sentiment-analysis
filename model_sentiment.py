
# coding: utf-8

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"
torch.set_default_tensor_type('torch.cuda.FloatTensor')
import pandas as pd
import itertools
import more_itertools
import numpy as np
import pickle

## Make the the multiple attention with word vectors.
def attention_lc(rnn_outputs, att_weights,seq_length_tensor):  #[4,3,4] [4,3,1] FloatTensor[2.,3.,1.,1.]
    attn_vectors = None
    norm_attn_weights = F.softmax(att_weights,dim=1)
    
    for i in range(rnn_outputs.size(0)):
        h_i = rnn_outputs[i]
        a_i = norm_attn_weights[i]
        
        ###Apply masked attention
        b_i = a_i.clone()
        masked_attn = int(att_weights.size(1)) - int(seq_length_tensor[i])
        if (masked_attn > 0):
            
            b_i[int(seq_length_tensor[i]):int(att_weights.size(1))] = torch.zeros([masked_attn,1])
            
            b_i[:int(seq_length_tensor[i])] = F.softmax(a_i[:int(seq_length_tensor[i])].clone(),dim=0)
            
        
        
        a_i = b_i
        h_i = a_i.to(device) * h_i.to(device)
        
        sent_vec_sample = h_i.unsqueeze(0)

        if(attn_vectors is None):
            attn_vectors = sent_vec_sample
            
        else:
            
            attn_vectors = torch.cat((attn_vectors,sent_vec_sample),0)
        
    attn_vectors = torch.sum(attn_vectors,dim =1).unsqueeze(1)
    #print (attn_vectors)
    return attn_vectors


## The word RNN model for generating a sentence vector
class WordRNN(nn.Module):
    def __init__(self, embedsize, batch_size, hid_size):
        super(WordRNN, self).__init__()
        self.batch_size = batch_size
        self.embedsize = embedsize
        self.hid_size = hid_size
        ## Word Encoder
        self.wordRNN = torch.nn.GRU(embedsize, hid_size, bidirectional=True)
        ## Word Attention
        self.wordattn = nn.Linear(2*hid_size, 2*hid_size)
        self.attn_combine = nn.Linear(2*hid_size, 1, bias=False)

    def forward(self,inp,sorted_seq_length,sorted_idx,hidden_size,max_length_word,batch_size,seq_length_tensor):
        
        out_state, hid_state = self.wordRNN(inp.to(device))   #hid_state [2,64,100]  
        # unpack
        unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(out_state, batch_first=True)
        #print (unpacked.size()) #b_s,time,feature_dim
    
        ## add zero tensor at 0 word length position
        if (unpacked.size()[1] == 1):
           unpacked = torch.zeros(batch_size,max_length_word,2 * hidden_size)
        
        elif (unpacked.size()[1] < max_length_word):
           padd_tensor =  torch.zeros(batch_size,max_length_word-unpacked.size()[1],2 *hidden_size)
           #print (padd_tensor.size())
           #print (unpacked.size())
           unpacked = torch.cat((unpacked.to(device), padd_tensor.to(device)), 1)
           for i,length in enumerate(sorted_seq_length.tolist()):
              if (length == 1):
                 unpacked[i] = torch.zeros(max_length_word,2 * hidden_size)
                
        else :
           pass
    
        #print (unpacked.size()) #b_s,time,feature_dimx
        # unsort the batch sample to reverse the above sorting process
        sort_unpacked = torch.zeros(unpacked.size())
        for i,idx in enumerate(sorted_idx.tolist()): 
           sort_unpacked[idx] = unpacked[i]
        unpacked = sort_unpacked         ### [4,3,4]
        #print (unpacked)
        word_annotation = self.wordattn(unpacked.to(device)) ##[4,3,4]
        #print (word_annotation)
        #attn = F.softmax(self.attn_combine(word_annotation),dim=1)
        attn = self.attn_combine(word_annotation)
        #print (attn)
        sent = attention_lc(unpacked,attn,seq_length_tensor)
        #print (sent)
        return sent

## The HAN model
class SentenceRNN(nn.Module):
    def __init__(self,embedsize, batch_size, hid_size,c):
        super(SentenceRNN, self).__init__()
        self.batch_size = batch_size
        self.embedsize = embedsize
        self.hid_size = hid_size
        self.cls = c

        ## Sentence Encoder
        self.sentRNN = torch.nn.GRU(2*hid_size, hid_size, bidirectional=True)
        ## Sentence Attention
        self.sentattn = nn.Linear(2*hid_size, 2*hid_size)
        self.attn_combine = nn.Linear(2*hid_size, 1,bias=False)
        ## Mapping feature to cls level 
        self.doc_linear = nn.Linear((2*hid_size)+4096, c)

    
    def forward(self,review_feature,inp,sorted_seq_length,sorted_idx,hidden_size,max_length_word,batch_size,sent_length_tensor):
        #print (inp.size())
        out_state, hid_state = self.sentRNN(inp)
        #print (out_state.size())
        sent_annotation = self.sentattn(out_state)
        #attn = F.softmax(self.attn_combine(sent_annotation),dim=1)
        attn = self.attn_combine(sent_annotation)
        
        
        doc = attention_lc(out_state,attn,sent_length_tensor)
        ##### concat sentiment neuron feature at review level
        
        doc = torch.cat((doc,review_feature),dim = 2)
        d = self.doc_linear(doc)
        
        cls = F.log_softmax(d.view(-1,self.cls),dim=1)
        
        return cls, hid_state
        


