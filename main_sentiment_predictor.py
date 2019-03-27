
import torch
from vectorizer_sentiment import imdb_dataset
from torch.utils.data import DataLoader
import pdb

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from model_sentiment import WordRNN
from model_sentiment import SentenceRNN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type('torch.cuda.FloatTensor')
#device = "cpu"

batch_size = 32
num_epochs = 15
max_length_word = 100
max_num_sent = 25
hidden_size = 100
#n_layers =1
feature_dim = 768
cls = 2
#****sent_net_imdb_11_0.9764324583866837.pth  word_net_imdb_11_0.9764324583866837.pth
## create instance of word level network
#sent_net_imdb_10_0.992157490396927.pth  word_net_imdb_10_0.992157490396927.pth
word_net = WordRNN(feature_dim, batch_size, hidden_size)
word_net.load_state_dict(torch.load('../Sentiment_codebase2/weights/word_net_imdb_10_0.992157490396927.pth'))
word_net.cuda()
word_net.eval()
## create instance of sentence level network
sent_net = SentenceRNN(feature_dim, batch_size, hidden_size,cls)
sent_net.load_state_dict(torch.load('../Sentiment_codebase2/weights/sent_net_imdb_10_0.992157490396927.pth'))
sent_net.cuda()
sent_net.eval()

torch.backends.cudnn.benchmark=True

##Hyperparameters for network architecture
# learning_rate = 1e-3
# momentum = 0.9

##List parameters for optimizer
params = list(word_net.parameters()) + list(sent_net.parameters())

#create optimizer 
#sent_optimizer = torch.optim.SGD(params, lr=learning_rate, momentum= momentum)
# sent_optimizer = torch.optim.Adam(params, lr=learning_rate)

# create loss function
# criterion = nn.NLLLoss()



dataset = imdb_dataset()
data_loader = DataLoader(dataset,batch_size,shuffle=True)  #,num_workers=4

fp = open('predictions.txt','w+')
batch_count = 0
total_correct = 0
nume = 0
denom = 0
for num_epoch in range(num_epochs):
	loss_epoch = []
	acc_epoch = []

	for i_batch, sample_batched in enumerate(data_loader):
		batch_count += 1
		print("****************************Batch count : ",batch_count)
		review_feature = sample_batched[0]
		seq_lengths = [values.tolist() for values in sample_batched[3]]
		print (len(seq_lengths))
		batch_in = sample_batched[1]
		print (batch_in.size(0))
		sent_lengths = sample_batched[2].tolist()
		print (len(sent_lengths))
		batch_level = sample_batched[4]
		sent_vect = None
		if (batch_in.size(0) != batch_size):
			break
	
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
    
    
    			out = word_net(pack,sorted_seq_length,sorted_idx,hidden_size,max_length_word,batch_size,seq_length_tensor.to(device)) #[B_S,1,h_s*2]
    
    			#print (out.size())
    			if (sent_vect is None):
        			sent_vect = out
    			else:
        			sent_vect = torch.cat((sent_vect,out),1)
    			#print (sent_vect.size()) ## sent_vect [B_S,num_sent,hidden_size *2] 
		
		y_pred,_ = sent_net(review_feature, sent_vect,sorted_seq_length,sorted_idx,hidden_size,max_length_word,batch_size,sent_length_tensor)
		y_true = batch_level.to(device)
		

		max_index = y_pred.max(dim = 1)[1]
		correct = (max_index == y_true).sum()
		total_correct += float(correct)
		fp.write(str(float(correct))+'\n')
		nume = nume + float(correct)
		denom = denom + batch_size
		print('*************** accuracy till now : ',nume/denom)
		acc = float(correct)/batch_size

		
acc = total_correct/25000
fp.write('Total accuracy : ')
fp.write(str(acc))
fp.close()
print('Final accuracy : ',acc)
