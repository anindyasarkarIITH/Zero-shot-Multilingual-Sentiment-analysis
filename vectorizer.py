from torch.utils.data import Dataset
import pandas as pd
import os
import torch
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type('torch.cuda.FloatTensor')
#device = "cuda"

class imdb_dataset(Dataset):
	df = pd.Series(list(range(0, 25000))).sample(frac=1,random_state=1432)
	i2file = dict(zip(df.tolist()[:12500], os.listdir('./aclImdb/aclImdb/test/pos')))
	i2file.update(dict(zip(df.tolist()[12500:], os.listdir('./aclImdb/aclImdb/test/neg'))))
	i2class = dict(zip(df.tolist()[:12500], ['pos']*12500))
	i2class.update(dict(zip(df.tolist()[12500:], ['neg']*12500)))
	path = Path('./aclImdb/aclImdb/test')
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
	bert_model = BertModel.from_pretrained('bert-base-uncased')
	bert_model.eval()
	#sn_model = Model()
	def __init__(self):
		pass
	def __len__(self):
		return 25000

	def __getitem__(self,idx):
		path = imdb_dataset.path / imdb_dataset.i2class[idx] / imdb_dataset.i2file[idx]
		review = open(str(path.absolute()),'r',encoding='utf-8').readlines()
		#sn_model = Model()
		#sn_vect = torch.tensor(imdb_dataset.sn_model.transform(review))
                
		max_words = 100
		tokenized_text_list = tokenize.sent_tokenize(review[0])
		if len(tokenized_text_list) >25:
			no_of_sents = 25
			tokenized_text_list = tokenized_text_list[:25]
		else:
			no_of_sents = len(tokenized_text_list)
		word_list = []
		level_list = []
		#print (tokenized_text_list)
		for sent_index,tokenized_text in enumerate(tokenized_text_list):
			# word_dict[mode][cls][file_index][sent_index] = {}

			## IFF Sentence is too big to handle by bert model
			if (len(tokenized_text) > 100):
				tokenized_text = tokenized_text[:100]

			# Convert token to vocabulary indices
			tokenized_text = imdb_dataset.tokenizer.tokenize(tokenized_text)
			indexed_tokens = imdb_dataset.tokenizer.convert_tokens_to_ids(tokenized_text)
			segments_ids = [0 for i in range(len(indexed_tokens))]
			if len(tokenized_text) != 0:
				word_list.append(len(tokenized_text))
			else:
				word_list.append(1)

			# Convert inputs to PyTorch tensors
			tokens_tensor = torch.tensor([indexed_tokens])
			segments_tensors = torch.tensor([segments_ids])

			# Predict hidden states features for each layer
			encoded_layers, _ = imdb_dataset.bert_model(tokens_tensor, segments_tensors, output_all_encoded_layers=False)
			encoded_layers = encoded_layers.data.to('cpu')

			if sent_index==0:
				bert_review_tensor = encoded_layers
				sent_shape = bert_review_tensor.size()
				a = sent_shape[0]
				b = max_words - sent_shape[1]
				c = sent_shape[2]
				bert_review_tensor = torch.cat((bert_review_tensor.to(device), torch.zeros(a, b, c).to(device)), 1)
			else:
				bert_review_tensor = imdb_dataset.get_review_tensor(bert_review_tensor,encoded_layers,max_words)
			#****************DO NOT DELETE, MIGHT BE USEFUL FOR WORD LEVEL SENTIMENT NEURON VECTORS*********************
			# We have a hidden states for each of the 12 layers in model bert-base-uncased
			## creating dictionary for word features
			# pdb.set_trace()
			# split_encoded_feature = torch.split(encoded_layers, 1, dim=1)
			#
			# for index,(raw_text,encoded_feature) in enumerate(zip(tokenized_text,split_encoded_feature)):
			# 	key = str(raw_text)+"_"+str(index)
			# 	value = torch.squeeze(encoded_feature)
			# 	word_dict[mode][cls][file_index][sent_index][key] = value #torch tensoor size[786]
			# ****************DO NOT DELETE, MIGHT BE USEFUL FOR WORD LEVEL SENTIMENT NEURON VECTORS*********************
			del tokens_tensor
			del segments_tensors
			# del split_encoded_feature
			del encoded_layers
			#del value
			del _
			gc.collect()
		if imdb_dataset.i2class[idx]=='pos':
			#y_label = torch.tensor([0,1])
			level_list.append(1)
		elif imdb_dataset.i2class[idx]=='neg':
			#y_label = torch.tensor([1,0])
			level_list.append(0)
		max_sents = 25
		word_list = word_list + [1]*(max_sents-no_of_sents)
		bert_review_tensor = imdb_dataset.get_final_bert_tensor(bert_review_tensor,max_sents)
		y_label = torch.LongTensor(level_list[:]).squeeze()
		gc.collect()
		#return (sn_vect,bert_review_tensor,no_of_sents,word_list,y_label)#
		return (bert_review_tensor,no_of_sents,word_list,y_label)#

	@staticmethod
	def get_final_bert_tensor(bert_review_tensor,max_sents):
		shape = bert_review_tensor.size()
		a = max_sents-shape[0]
		b = shape[1]
		c = shape[2]
		return torch.cat((bert_review_tensor,torch.zeros(a,b,c)),0)
	@staticmethod
	def get_review_tensor(review_tensor,sent_tensor,max_words):
		sent_shape = sent_tensor.size()
		a = sent_shape[0]
		b = max_words - sent_shape[1]
		c = sent_shape[2]
		sent_tensor = torch.cat((sent_tensor.to(device),torch.zeros(a,b,c).to(device)),1)
		return torch.cat((review_tensor,sent_tensor),0)

