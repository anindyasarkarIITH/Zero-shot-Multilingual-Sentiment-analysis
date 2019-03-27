import torch
import torch.nn as nn
from model import WordRNN
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

feature_dim = 768
batch_size = 32
hidden_size = 100
word_net = WordRNN(768, 1, 100)
word_net.load_state_dict(torch.load('./weights/word_net_imdb_7_0.9801136363636364.pth'))
word_net.eval()
word_net.cuda()


class english_hindi_net(nn.Module) : 
	def __init__(self, input_size, h1, h2) : 
		super(english_hindi_net, self).__init__()
		self.fc1 = nn.Linear(input_size , h1)
		self.fc2 = nn.Linear(h1 , h2)
		self.fc3 = nn.Linear(h2 , input_size)
		self.relu = nn.ReLU()

	def forward(self, x) : 
		out = self.fc1(x)
		out = self.relu(out)
		out = self.fc2(out)
		out = self.relu(out)
		out = self.fc3(out)
		return out

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


class en_hi_dataset(Dataset):
	df = pd.read_excel('./final_series_en_to_hindi.xlsx')
	df[0] = df[0].apply(lambda x: str(x))
	tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
	bert_model = BertModel.from_pretrained('bert-base-multilingual-uncased')
	bert_model.eval()


	def __len__(self):
		return 81266

	def __getitem__(self,idx):
		reviews = en_hi_dataset.df.iloc[idx][0].split('\n')
		if len(reviews)==2:
			en_review,hi_review = reviews
		else:
			en_review = reviews[0]
			hi_review = reviews[0]
		max_words = 512
		no_of_sents = 1
		
		en_vecs = en_hi_dataset.get_bert_vectors(en_review,no_of_sents,max_words)
		hi_vecs = en_hi_dataset.get_bert_vectors(hi_review,no_of_sents,max_words)

	# for i_batch, sample_batched in enumerate(data_loader):
		#review_feature = sample_batched[0]


		en_vector = en_hi_dataset.get_han_sent_vects(en_vecs)
		hi_vector = en_hi_dataset.get_han_sent_vects(hi_vecs)

		return (en_vector,hi_vector)


	@staticmethod
	def get_han_sent_vects(sample_batched):
	#for i_batch, sample_batched in enumerate(data_loader):
		#review_feature = sample_batched[0]
		#pdb.set_trace()
		seq_lengths = [[values] for values in sample_batched[2]]
		#print ("!!!!!!!!!!!!", seq_lengths)
		#print (len(seq_lengths))
		batch_in = sample_batched[0].unsqueeze(0)
		#print (batch_in.size())
		sent_lengths = [sample_batched[1]]
		#print (len(sent_lengths))
		#batch_level = sample_batched[3]
		sent_vect = None
		
	
		for sent_index,seq_length in enumerate(seq_lengths):

    			#get the batch data --> A sentence from all the reviwes in a batch
    			batch_word_in = batch_in.split(1,dim=1)
    
    			batch_in_word =  batch_word_in[0].squeeze(dim=1) #(batch_size,max_length_word,feature_dim)
    
    			# create sorted batch based on word length
    			seq_length_tensor = torch.FloatTensor(seq_length)
    			sent_length_tensor = torch.FloatTensor(sent_lengths)   

    			sorted_seq_length, sorted_idx = seq_length_tensor.sort(descending=True)
    
    			batch_in_word_sorted = torch.zeros(batch_in_word.size())
    
    			for i,idx in enumerate(sorted_idx.tolist()):
        			batch_in_word_sorted[i] = batch_in_word[idx]
        
    			# pack it
    			pack = torch.nn.utils.rnn.pack_padded_sequence(batch_in_word_sorted, sorted_seq_length.tolist(), batch_first=True)
    
    
    			out = word_net(pack,sorted_seq_length,sorted_idx,hidden_size,max_length_word,1,seq_length_tensor.to(device)).detach() #[B_S,1,h_s*2]
		return out, sorted_seq_length, sorted_idx, sent_length_tensor

	@staticmethod
	def get_bert_vectors(review,no_of_sents,max_words):
		tokenized_text_list = tokenize.sent_tokenize(review)
		if len(tokenized_text_list) >no_of_sents:
			tokenized_text_list = tokenized_text_list[:no_of_sents]
		else:
			no_of_sents = len(tokenized_text_list)
		word_list = []
		for sent_index,tokenized_text in enumerate(tokenized_text_list):
			# word_dict[mode][cls][file_index][sent_index] = {}

			## IFF Sentence is too big to handle by bert model
			if (len(tokenized_text) > max_words):
				tokenized_text = tokenized_text[:max_words]

			# Convert token to vocabulary indices
			tokenized_text = en_hi_dataset.tokenizer.tokenize(tokenized_text)
			indexed_tokens = en_hi_dataset.tokenizer.convert_tokens_to_ids(tokenized_text)
			segments_ids = [0 for i in range(len(indexed_tokens))]
			if len(tokenized_text) != 0:
				word_list.append(len(tokenized_text))
			else:
				word_list.append(1)

			# Convert inputs to PyTorch tensors
			tokens_tensor = torch.tensor([indexed_tokens])#.to(device)
			segments_tensors = torch.tensor([segments_ids])#.to(device)

			# Predict hidden states features for each layer
			encoded_layers, _ = en_hi_dataset.bert_model(tokens_tensor, segments_tensors, output_all_encoded_layers=False)
			encoded_layers = encoded_layers.data

			if sent_index==0:
				bert_review_tensor = encoded_layers
				sent_shape = bert_review_tensor.size()
				a = sent_shape[0]
				b = max_words - sent_shape[1]
				c = sent_shape[2]
				bert_review_tensor = torch.cat((bert_review_tensor, torch.zeros(a, b, c)), 1)
			else:
				bert_review_tensor = get_review_tensor(bert_review_tensor,encoded_layers,max_words)

			del tokens_tensor
			del segments_tensors
			# del split_encoded_feature
			del encoded_layers
			#del value
			del _
			gc.collect()
		max_sents = no_of_sents
		word_list = word_list + [1]*(max_sents-no_of_sents)
		bert_review_tensor = get_final_bert_tensor(bert_review_tensor,max_sents)
		return (bert_review_tensor,no_of_sents,word_list)##return bert_review_tensor


num_epochs = 25
max_length_word = 100
max_num_sent = 25

input_size = 200
h1 = 2*input_size
h2 = int((3*input_size)/2)

'''
learning_rate = 0.0001
model = english_hindi_net(input_size, h1, h2)
model.cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
dataset = en_hi_dataset()
data_loader = DataLoader(dataset,batch_size,shuffle=True)
for epoch in range(num_epochs) : 
	batch_loss = 0
	for i_batch, sample_batched in enumerate(data_loader):
				y = sample_batched[0]
				#print ("@@@@",y.size())
				x = sample_batched[1]

				outputs = model(x)
				loss = criterion(outputs , y)
				batch_loss += loss
				print ("loss %f after %d epochs" %(loss.data[0],epoch+1))
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				print (epoch)
	torch.save(model.state_dict(), './weights/transform_model_'+str(epoch)+str('_')+str(batch_loss)+'.pth')                     
'''		
