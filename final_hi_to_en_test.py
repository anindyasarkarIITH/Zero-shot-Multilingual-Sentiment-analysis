import torch
import torch.nn as nn
from model import WordRNN,SentenceRNN
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import os
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from torch.backends import cudnn
import numpy as np
import nltk
# nltk.download('punkt')
from nltk import tokenize
from pathlib import Path
from sklearn.externals import joblib
import gc 
import pdb
from sentiment_neuron.encoder import Model
from nltk import tokenize
from pathlib import Path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from srv_en_to_hindi_nn import english_hindi_net
from srv_en_to_hindi_nn import en_hi_dataset

def get_final_bert_tensor(bert_review_tensor, max_sents):
		shape = bert_review_tensor.size()
		a = max_sents - shape[0]
		b = shape[1]
		c = shape[2]
		return torch.cat((bert_review_tensor, torch.zeros(a, b, c)), 0)

def get_review_tensor(review_tensor, sent_tensor, max_words):
		sent_shape = sent_tensor.size()
		a = sent_shape[0]
		b = max_words - sent_shape[1]
		c = sent_shape[2]
		sent_tensor = torch.cat((sent_tensor, torch.zeros(a, b, c)), 1)
		return torch.cat((review_tensor, sent_tensor), 0)

word_net = WordRNN(768, 1, 100)
word_net.load_state_dict(torch.load('./weights/word_net_imdb_10_0.9735515364916774.pth'))
word_net.eval()
word_net.cuda()

## create instance of sentence level network
sent_net = SentenceRNN(768, 1, 100,2)
sent_net.load_state_dict(torch.load('./weights/sent_net_imdb_10_0.9735515364916774.pth'))
word_net.eval()
sent_net.cuda()

input_size = 200
h1 = 2*input_size
h2 = int((3*input_size)/2)
eh_net = english_hindi_net(input_size,h1,h2)
eh_net.load_state_dict(torch.load('./weights/transform_model_21_tensor(165.2781, grad_fn=<ThAddBackward>).pth'))
word_net.eval()
word_net.cuda()


def get_vects(sentences):
	no_of_sents = 1
	max_words = 512
	vects = []
	for sentence in sentences:
		hi_vecs = en_hi_dataset.get_bert_vectors(sentence,no_of_sents,max_words)
		hi_vector, sorted_seq_length, sorted_idx, sent_length_tensor = en_hi_dataset.get_han_sent_vects(hi_vecs)
		vects.append(eh_net(hi_vector))
	return vects,sorted_seq_length, sorted_idx, sent_length_tensor

#bs*no_of_sent*(2*hidsize)
badla = './hindi_revs/manikarnika.txt'
fp = open(badla,'r')
badla = fp.readlines()
fp.close()
badla = [sentence[:-1] for sentence in badla]

manikarnika = './hindi_revs/manikarnika.txt'
fp = open(manikarnika,'r')
manikarninka = fp.readlines()
fp.close()
manikarninka = [sentence[:-1] for sentence in manikarninka]

sent_vect_,sorted_seq_length, sorted_idx, sent_length_tensor = get_vects(badla)
sent_vect = torch.cat(sent_vect_, dim = 1)
#pdb.set_trace()
y_pred,_ = sent_net(sent_vect,sorted_seq_length,sorted_idx,100,100,1,sent_length_tensor)
print (y_pred)








