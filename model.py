import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
use_cuda = torch.cuda.is_available() 

def softmax(inputs):
	x = inputs - torch.max(inputs, 1, keepdim=True)[0]
	exp_x = torch.exp(x)
	exp_x_sum = torch.sum(exp_x, 1, keepdim=True)
	return exp_x / exp_x_sum

def mask_softmax(score, weight):
	scores_rescale = score - torch.max(torch.abs(weight) * score, 1, keepdim=True)[0]
	scores_exp = torch.exp(scores_rescale)
	scores_sum = torch.sum(scores_exp * torch.abs(weight), 1, keepdim=True)
	return (scores_exp / scores_sum) * torch.abs(weight)

def dimensional_attentive_combine(a, b, theta, bias, diagonal=True):	
	if diagonal:
		a_trans = a * theta + bias
		b_trans = b * theta + bias
	else:
		a_trans = torch.matmul(a, theta) + bias
		b_trans = torch.matmul(b, theta) + bias		
	exp_a = torch.exp(a_trans)
	exp_b = torch.exp(b_trans)
	# exp_a = torch.sigmoid(a_trans)
	# exp_b = torch.sigmoid(b_trans)		
	sum_exp_a_b = exp_a + exp_b
	w_a = exp_a / sum_exp_a_b
	w_b = exp_b / sum_exp_a_b				
	combined = w_a * a + w_b * b
	return combined	

class NBOW(nn.Module):
	def __init__(self, embed_dim, n_word, bound, pretrain=False):
		super(NBOW, self).__init__()	
		self.embed_dim = embed_dim
		self.n_word = n_word	
		self.word_embedding = nn.Embedding(self.n_word+1, self.embed_dim, padding_idx=self.n_word)	
		if pretrain:
			self.word_embedding.weight = nn.Parameter(torch.from_numpy(np.load('pretrain_embedding/word_embedding.npy')))
		else:
			self.word_embedding.weight = nn.Parameter(torch.from_numpy(np.asarray(np.random.RandomState(122).uniform(low=-bound, high=bound, size=(self.n_word+1, self.embed_dim)), dtype=np.float32)))

	def forward(self, *args):
		ed = self.word_embedding(args[0])
		ed = torch.sum(ed, 1)	
		return ed

class APCNN(nn.Module):
	def __init__(self, embed_dim, n_word, bound, pretrain=False):
		super(APCNN, self).__init__()	
		self.embed_dim = embed_dim
		self.n_word = n_word	
		self.word_embedding = nn.Embedding(self.n_word+1, self.embed_dim, padding_idx=self.n_word)	
		if pretrain:
			self.word_embedding.weight = nn.Parameter(torch.from_numpy(np.load('pretrain_embedding/word_embedding.npy')))
		else:
			self.word_embedding.weight = nn.Parameter(torch.from_numpy(np.asarray(np.random.RandomState(122).uniform(low=-bound, high=bound, size=(self.n_word+1, self.embed_dim)), dtype=np.float32)))
		self.conv_layer = nn.Conv1d(embed_dim, embed_dim, 1)
		# self.conv_layer = nn.DataParallel(self.conv_layer)
		self.conv_layer.weight = nn.Parameter(torch.from_numpy(np.asarray(np.random.RandomState(130).uniform(low=-bound, high=bound, size=(embed_dim, embed_dim, 1)), dtype=np.float32)))
		self.conv_layer.bias = nn.Parameter(torch.from_numpy(np.asarray(np.zeros(embed_dim), dtype=np.float32)))

	def forward(self, inputs):
		inputs = self.word_embedding(inputs)
		inputs = torch.transpose(inputs, 1, 2)
		conv = self.conv_layer(inputs)
		activated = F.tanh(conv)
		pooled = torch.sum(activated, 2)
		# if self.diagonal:
		# 	e_theta = e * self.theta + self.bias
		# else:
		# 	e_theta = torch.matmul(e, self.theta) + self.bias
		# print activated.size()
		# print e_theta.size()
		# scores = torch.matmul(activated, torch.unsqueeze(e_theta, 2))
		# weight = softmax(torch.squeeze(scores, 2))
		# pooled = torch.sum(activated*torch.unsqueeze(weight, 2), 1)
		return pooled

class GRU_encoder(nn.Module):
	def __init__(self, embed_dim, n_word, bound, pretrain=False):
		super(GRU_encoder, self).__init__()	
		self.embed_dim = embed_dim
		self.n_word = n_word	
		self.gru = nn.GRU(embed_dim, embed_dim/2, bidirectional=True)
		self.gru = nn.DataParallel(self.gru, dim=1)						
		self.word_embedding = nn.Embedding(self.n_word+1, self.embed_dim, padding_idx=self.n_word)	
		if pretrain:
			self.word_embedding.weight = nn.Parameter(torch.from_numpy(np.load('pretrain_embedding/word_embedding.npy')))
		else:
			self.word_embedding.weight = nn.Parameter(torch.from_numpy(np.asarray(np.random.RandomState(122).uniform(low=-bound, high=bound, size=(self.n_word+1, self.embed_dim)), dtype=np.float32)))
		self.h0 = nn.Parameter(torch.from_numpy(np.asarray(np.random.uniform(low=-bound, high=bound, size=(2, 1, embed_dim/2)), dtype=np.float32)))

	def forward(self, ed):
		h0 = self.h0.repeat(1, ed.size()[0], 1)		
		ed = self.word_embedding(ed)	
		out_ed, hn = self.gru(torch.transpose(ed, 0, 1), h0)		
		out_ed = torch.sum(out_ed, 0)										
		return out_ed

class ProjE(nn.Module):
	def __init__(self, embed_dim, n_entity, n_relation, dropout=0.5, reg_weight=1e-5):
		super(ProjE, self).__init__()
		self.embed_dim = embed_dim
		self.n_entity = n_entity
		self.n_relation = n_relation
		self.dropout = dropout
		self.reg_weight = reg_weight
		bound = 6./np.sqrt(self.n_entity)
		self.entity_embedding = nn.Embedding(self.n_entity, self.embed_dim)
		self.entity_embedding.weight = nn.Parameter(torch.from_numpy(
			np.asarray(np.random.RandomState(123).uniform(
				low=-bound, high=bound, size=(self.n_entity, self.embed_dim)), dtype=np.float32)))	
		self.relation_embedding = nn.Embedding(self.n_relation, self.embed_dim)
		self.relation_embedding.weight = nn.Parameter(torch.from_numpy(
			np.asarray(np.random.RandomState(456).uniform(
				low=-bound, high=bound, size=(self.n_relation, self.embed_dim)), dtype=np.float32)))
		self.weight_t = nn.Parameter(torch.from_numpy(np.asarray(np.random.RandomState(501).uniform(low=-bound, high=bound, size=(self.embed_dim)), dtype=np.float32)))
		self.weight_tr = nn.Parameter(torch.from_numpy(np.asarray(np.random.RandomState(502).uniform(low=-bound, high=bound, size=(self.embed_dim)), dtype=np.float32)))
		self.weight_bias_t = nn.Parameter(torch.from_numpy(np.asarray(np.zeros(self.embed_dim), dtype=np.float32)))		
		self.weight_h = nn.Parameter(torch.from_numpy(np.asarray(np.random.RandomState(501).uniform(low=-bound, high=bound, size=(self.embed_dim)), dtype=np.float32)))
		self.weight_hr = nn.Parameter(torch.from_numpy(np.asarray(np.random.RandomState(502).uniform(low=-bound, high=bound, size=(self.embed_dim)), dtype=np.float32)))
		self.weight_bias_h = nn.Parameter(torch.from_numpy(np.asarray(np.zeros(self.embed_dim), dtype=np.float32)))	
		self.zero = Variable(torch.zeros((1)), requires_grad=False)	
		if use_cuda:
			self.zero = self.zero.cuda()

	def forward(self, triple, neg_sample_h, neg_sample_t):
		h = self.entity_embedding(triple[:,0])
		t = self.entity_embedding(triple[:,1])		
		r = self.relation_embedding(triple[:,2])

		tr = self.weight_t * t + self.weight_tr * r
		tr = F.dropout(F.tanh(tr + self.weight_bias_t), self.dropout, training=True)
		scores_trh = torch.matmul(tr, torch.t(self.entity_embedding.weight))
		log_prob_trh = torch.log(torch.clamp(mask_softmax(scores_trh, neg_sample_h), 1e-10, 1.0))
		y_trh = torch.max(neg_sample_h, self.zero)
		loss_trh = -torch.sum(log_prob_trh * y_trh/torch.sum(y_trh, 1, keepdim=True))		

		hr = self.weight_h * h + self.weight_hr * r
		hr = F.dropout(F.tanh(hr + self.weight_bias_h), self.dropout, training=True)
		scores_hrt = torch.matmul(hr, torch.t(self.entity_embedding.weight))
		log_prob_hrt = torch.log(torch.clamp(mask_softmax(scores_hrt, neg_sample_t), 1e-10, 1.0))
		y_hrt = torch.max(neg_sample_t, self.zero)
		loss_hrt = -torch.sum(log_prob_hrt * y_hrt/torch.sum(y_hrt, 1, keepdim=True))		

		regularizer_loss = torch.sum(torch.abs(
			self.entity_embedding.weight)) + torch.sum(torch.abs(
			self.relation_embedding.weight)) + torch.sum(torch.abs(
			self.weight_t)) + torch.sum(torch.abs(
			self.weight_tr)) + torch.sum(torch.abs(
			self.weight_h)) + torch.sum(torch.abs(
			self.weight_hr))

		return loss_trh + loss_hrt + self.reg_weight * regularizer_loss

	def pred(self, htr):
		h = self.entity_embedding(htr[:,0])
		t = self.entity_embedding(htr[:,1])		
		r = self.relation_embedding(htr[:,2])				
		tr = self.weight_t * t + self.weight_tr * r
		tr = F.tanh(tr + self.weight_bias_t)
		hr = self.weight_h * h + self.weight_hr * r
		hr = F.tanh(hr + self.weight_bias_h)		
		scores_h = torch.matmul(tr, torch.t(self.entity_embedding.weight))		
		scores_t = torch.matmul(hr, torch.t(self.entity_embedding.weight))
		return torch.topk(scores_h, self.n_entity)[1], torch.topk(scores_t, self.n_entity)[1]

class ProjE_R(nn.Module):
	def __init__(self, embed_dim, n_entity, n_relation, dropout=0.5, reg_weight=1e-5):
		super(ProjE_R, self).__init__()
		self.embed_dim = embed_dim
		self.n_entity = n_entity
		self.n_relation = n_relation
		self.dropout = dropout
		self.reg_weight = reg_weight
		bound = 6./np.sqrt(self.n_entity)
		self.entity_embedding = nn.Embedding(self.n_entity, self.embed_dim)
		self.entity_embedding.weight = nn.Parameter(torch.from_numpy(
			np.asarray(np.random.RandomState(123).uniform(
				low=-bound, high=bound, size=(self.n_entity, self.embed_dim)), dtype=np.float32)))	
		self.relation_embedding = nn.Embedding(self.n_relation, self.embed_dim)
		self.relation_embedding.weight = nn.Parameter(torch.from_numpy(
			np.asarray(np.random.RandomState(456).uniform(
				low=-bound, high=bound, size=(self.n_relation, self.embed_dim)), dtype=np.float32)))
		self.weight_h = nn.Parameter(torch.from_numpy(np.asarray(np.random.RandomState(501).uniform(low=-bound, high=bound, size=(self.embed_dim)), dtype=np.float32)))
		self.weight_t = nn.Parameter(torch.from_numpy(np.asarray(np.random.RandomState(502).uniform(low=-bound, high=bound, size=(self.embed_dim)), dtype=np.float32)))
		self.weight_bias = nn.Parameter(torch.from_numpy(np.asarray(np.zeros(self.embed_dim), dtype=np.float32)))		
		self.zero = Variable(torch.zeros((1)), requires_grad=False)	
		if use_cuda:
			self.zero = self.zero.cuda()

	def forward(self, triple, neg_sample_r):
		h = self.entity_embedding(triple[:,0])
		t = self.entity_embedding(triple[:,1])		

		ht = self.weight_h * h + self.weight_t * t
		ht = F.dropout(F.tanh(ht + self.weight_bias), self.dropout, training=True)
		scores_htr = torch.matmul(ht, torch.t(self.relation_embedding.weight))
		log_prob_htr = torch.log(torch.clamp(mask_softmax(scores_htr, neg_sample_r), 1e-10, 1.0))
		y_htr = torch.max(neg_sample_r, self.zero)
		loss_htr = -torch.sum(log_prob_htr * y_htr/torch.sum(y_htr, 1, keepdim=True))				

		regularizer_loss = torch.sum(torch.abs(
			self.entity_embedding.weight)) + torch.sum(torch.abs(
			self.relation_embedding.weight)) + torch.sum(torch.abs(
			self.weight_h)) + torch.sum(torch.abs(
			self.weight_t))

		return loss_htr + self.reg_weight * regularizer_loss

	def pred(self, htr):
		h = self.entity_embedding(htr[:,0])
		t = self.entity_embedding(htr[:,1])						
		ht = self.weight_h * h + self.weight_t * t
		ht = F.tanh(ht + self.weight_bias)		
		scores_r = torch.matmul(ht, torch.t(self.relation_embedding.weight))		
		return torch.topk(scores_r, self.n_relation)[1]

class TAProjE(nn.Module):

	def __init__(self, embed_dim, n_entity, n_relation, n_word, dropout=0.5, reg_weight=1e-5, encoder_type='nbow', combine_methods='gate', diagonal=False, pretrain=False):
		super(TAProjE, self).__init__()
		self.embed_dim = embed_dim
		self.n_entity = n_entity
		self.n_relation = n_relation
		self.n_word = n_word		
		self.dropout = dropout
		self.reg_weight = reg_weight
		self.encoder_type = encoder_type
		self.combine_methods = combine_methods
		self.diagonal = diagonal
		bound = 6./np.sqrt(self.n_entity)
		self.entity_embedding = nn.Embedding(self.n_entity, self.embed_dim)
		self.relation_embedding = nn.Embedding(self.n_relation, self.embed_dim)		
		if combine_methods=='gate':
			self.gate = nn.Embedding(self.n_entity, self.embed_dim)	
			self.gate.weight = nn.Parameter(torch.from_numpy(
				np.asarray(np.random.RandomState(234).uniform(
					low=-0.1, high=0.1, size=(self.n_entity, self.embed_dim)), dtype=np.float32)))			
		if combine_methods=='dimensional_attentive':
			if diagonal:
				self.theta = nn.Parameter(torch.from_numpy(np.asarray(np.random.RandomState(250).uniform(low=-bound, high=bound, size=(self.embed_dim)), dtype=np.float32)))
			else:
				self.theta = nn.Parameter(torch.from_numpy(np.asarray(np.random.RandomState(250).uniform(low=-bound, high=bound, size=(self.embed_dim, self.embed_dim)), dtype=np.float32)))
			self.bias = nn.Parameter(torch.from_numpy(np.zeros(shape=(self.embed_dim), dtype=np.float32)))

		if pretrain:
			self.entity_embedding.weight = nn.Parameter(torch.from_numpy(np.load('pretrain_embedding/entity_embedding.npy')))			
			self.relation_embedding.weight = nn.Parameter(torch.from_numpy(np.load('pretrain_embedding/relation_embedding.npy')))							
			self.weight_t = nn.Parameter(torch.from_numpy(np.load('pretrain_embedding/weight_t.npy')))
			self.weight_tr = nn.Parameter(torch.from_numpy(np.load('pretrain_embedding/weight_tr.npy')))
			self.weight_bias_t = nn.Parameter(torch.from_numpy(np.load('pretrain_embedding/weight_bias_t.npy')))		
			self.weight_h = nn.Parameter(torch.from_numpy(np.load('pretrain_embedding/weight_h.npy')))
			self.weight_hr = nn.Parameter(torch.from_numpy(np.load('pretrain_embedding/weight_hr.npy')))
			self.weight_bias_h = nn.Parameter(torch.from_numpy(np.load('pretrain_embedding/weight_bias_h.npy')))	
		else:			
			self.entity_embedding.weight = nn.Parameter(torch.from_numpy(
				np.asarray(np.random.RandomState(123).uniform(
					low=-bound, high=bound, size=(self.n_entity, self.embed_dim)), dtype=np.float32)))	
			self.relation_embedding.weight = nn.Parameter(torch.from_numpy(
				np.asarray(np.random.RandomState(456).uniform(
					low=-bound, high=bound, size=(self.n_relation, self.embed_dim)), dtype=np.float32)))						
			self.weight_t = nn.Parameter(torch.from_numpy(np.asarray(np.random.RandomState(501).uniform(low=-bound, high=bound, size=(self.embed_dim)), dtype=np.float32)))
			self.weight_tr = nn.Parameter(torch.from_numpy(np.asarray(np.random.RandomState(502).uniform(low=-bound, high=bound, size=(self.embed_dim)), dtype=np.float32)))
			self.weight_bias_t = nn.Parameter(torch.from_numpy(np.asarray(np.zeros(self.embed_dim), dtype=np.float32)))		
			self.weight_h = nn.Parameter(torch.from_numpy(np.asarray(np.random.RandomState(501).uniform(low=-bound, high=bound, size=(self.embed_dim)), dtype=np.float32)))
			self.weight_hr = nn.Parameter(torch.from_numpy(np.asarray(np.random.RandomState(502).uniform(low=-bound, high=bound, size=(self.embed_dim)), dtype=np.float32)))
			self.weight_bias_h = nn.Parameter(torch.from_numpy(np.asarray(np.zeros(self.embed_dim), dtype=np.float32)))	
		self.zero = Variable(torch.zeros((1)), requires_grad=False)	
		if use_cuda:
			self.zero = self.zero.cuda()
		if encoder_type=='gru':
			self.encoder = GRU_encoder(self.embed_dim, self.n_word, bound, pretrain)
		elif encoder_type=='apcnn':
			self.encoder = APCNN(self.embed_dim, self.n_word, bound, pretrain)
		else:
			self.encoder = NBOW(self.embed_dim, self.n_word, bound, pretrain)

	def des_weght(self, des_list):
		if self.combine_methods=='gate':
			des_w = 1.0 - torch.sigmoid(self.gate.weight)

		if self.combine_methods=='dimensional_attentive': 
			a = self.entity_embedding.weight
			b = self.encoder(des_list)
			if self.diagonal:
				a_trans = a * self.theta + self.bias
				b_trans = b * self.theta + self.bias
			else:
				a_trans = torch.matmul(a, self.theta) + self.bias
				b_trans = torch.matmul(b, self.theta) + self.bias		
			exp_a = torch.exp(a_trans)
			exp_b = torch.exp(b_trans)	
			sum_exp_a_b = exp_a + exp_b
			des_w = exp_b / sum_exp_a_b	
		return des_w	

	def combined_ed(self, des_list):
		des_embedding = self.encoder(des_list)
		if self.combine_methods=='gate':
			gate_e = torch.sigmoid(self.gate.weight)
			ed = gate_e * self.entity_embedding.weight + (1.0-gate_e) * des_embedding
		elif self.combine_methods=='dimensional_attentive':
			ed = dimensional_attentive_combine(self.entity_embedding.weight, des_embedding, self.theta, self.bias, self.diagonal)
		return ed

	def forward(self, triple, hd, td, neg_sample_h, neg_sample_t, des_list):
		h = self.entity_embedding(triple[:,0])
		t = self.entity_embedding(triple[:,1])		
		r = self.relation_embedding(triple[:,2])

		hd = self.encoder(hd)
		td = self.encoder(td)

		if self.combine_methods=='gate':
			gate_h = torch.sigmoid(self.gate(triple[:,0]))
			gate_t = torch.sigmoid(self.gate(triple[:,1]))
			h = gate_h * h + (1.0-gate_h) * hd
			t = gate_t * t + (1.0-gate_t) * td
			w_reg = torch.sum(torch.abs(self.gate.weight))						
		elif self.combine_methods=='dimensional_attentive':
			h = dimensional_attentive_combine(h, hd, self.theta, self.bias, self.diagonal)
			t = dimensional_attentive_combine(t, td, self.theta, self.bias, self.diagonal)
			w_reg = torch.sum(torch.abs(self.theta))			

		ed = self.combined_ed(des_list)
			
		tr = self.weight_t * t + self.weight_tr * r
		tr = F.dropout(F.tanh(tr + self.weight_bias_t), self.dropout, training=True)
		scores_trh = torch.matmul(tr, torch.t(ed))
		log_prob_trh = torch.log(torch.clamp(mask_softmax(scores_trh, neg_sample_h), 1e-10, 1.0))
		y_trh = torch.max(neg_sample_h, self.zero)
		loss_trh = -torch.sum(log_prob_trh * y_trh/torch.sum(y_trh, 1, keepdim=True))		

		hr = self.weight_h * h + self.weight_hr * r
		hr = F.dropout(F.tanh(hr + self.weight_bias_h), self.dropout, training=True)
		scores_hrt = torch.matmul(hr, torch.t(ed))
		log_prob_hrt = torch.log(torch.clamp(mask_softmax(scores_hrt, neg_sample_t), 1e-10, 1.0))
		y_hrt = torch.max(neg_sample_t, self.zero)
		loss_hrt = -torch.sum(log_prob_hrt * y_hrt/torch.sum(y_hrt, 1, keepdim=True))		

		regularizer_loss = torch.sum(torch.abs(
			self.entity_embedding.weight)) + torch.sum(torch.abs(
			self.relation_embedding.weight)) + torch.sum(torch.abs(
			self.encoder.word_embedding.weight)) + w_reg + torch.sum(torch.abs(
			self.weight_t)) + torch.sum(torch.abs(
			self.weight_tr)) + torch.sum(torch.abs(
			self.weight_h)) + torch.sum(torch.abs(
			self.weight_hr))

		return loss_trh + loss_hrt + self.reg_weight * regularizer_loss	

	def pred(self, htr, hd, td, ed):		
		h = self.entity_embedding(htr[:,0])
		t = self.entity_embedding(htr[:,1])	
		r = self.relation_embedding(htr[:,2])
			
		hd = self.encoder(hd)
		td = self.encoder(td)	
			
		if self.combine_methods=='gate':
			gate_h = torch.sigmoid(self.gate(htr[:,0]))
			gate_t = torch.sigmoid(self.gate(htr[:,1]))
			h = gate_h * h + (1.0-gate_h) * hd
			t = gate_t * t + (1.0-gate_t) * td			
		elif self.combine_methods=='dimensional_attentive':
			h = dimensional_attentive_combine(h, hd, self.theta, self.bias, self.diagonal)
			t = dimensional_attentive_combine(t, td, self.theta, self.bias, self.diagonal)			
					
		tr = self.weight_t * t + self.weight_tr * r
		tr = F.tanh(tr + self.weight_bias_t)
		hr = self.weight_h * h + self.weight_hr * r
		hr = F.tanh(hr + self.weight_bias_h)		
		scores_h = torch.matmul(tr, torch.t(ed))		
		scores_t = torch.matmul(hr, torch.t(ed))
		return torch.topk(scores_h, self.n_entity)[1], torch.topk(scores_t, self.n_entity)[1]

class TAProjE_R(nn.Module):

	def __init__(self, embed_dim, n_entity, n_relation, n_word, dropout=0.5, reg_weight=1e-5, encoder_type='nbow', combine_methods='gate', diagonal=False, pretrain=False):
		super(TAProjE_R, self).__init__()
		self.embed_dim = embed_dim
		self.n_entity = n_entity
		self.n_relation = n_relation
		self.n_word = n_word	
		self.dropout = dropout
		self.reg_weight = reg_weight
		self.encoder_type = encoder_type
		self.combine_methods = combine_methods
		self.diagonal = diagonal
		bound = 6./np.sqrt(self.n_entity)
		self.entity_embedding = nn.Embedding(self.n_entity, self.embed_dim)
		self.relation_embedding = nn.Embedding(self.n_relation, self.embed_dim)		
		if combine_methods=='gate':
			self.gate = nn.Embedding(self.n_entity, self.embed_dim)	
			self.gate.weight = nn.Parameter(torch.from_numpy(
				np.asarray(np.random.RandomState(234).uniform(
					low=-0.1, high=0.1, size=(self.n_entity, self.embed_dim)), dtype=np.float32)))			
		if combine_methods=='dimensional_attentive':
			if diagonal:
				self.theta = nn.Parameter(torch.from_numpy(np.asarray(np.random.RandomState(250).uniform(low=-bound, high=bound, size=(self.embed_dim)), dtype=np.float32)))
			else:
				self.theta = nn.Parameter(torch.from_numpy(np.asarray(np.random.RandomState(250).uniform(low=-bound, high=bound, size=(self.embed_dim, self.embed_dim)), dtype=np.float32)))
			self.bias = nn.Parameter(torch.from_numpy(np.zeros(shape=(self.embed_dim), dtype=np.float32)))

		if pretrain:
			self.entity_embedding.weight = nn.Parameter(torch.from_numpy(np.load('pretrain_embedding/entity_embedding.npy')))			
			self.relation_embedding.weight = nn.Parameter(torch.from_numpy(np.load('pretrain_embedding/relation_embedding.npy')))							
			self.weight_h = nn.Parameter(torch.from_numpy(np.load('pretrain_embedding/weight_h.npy')))
			self.weight_t = nn.Parameter(torch.from_numpy(np.load('pretrain_embedding/weight_t.npy')))
			self.weight_bias = nn.Parameter(torch.from_numpy(np.load('pretrain_embedding/weight_bias.npy')))		
	
		else:			
			self.entity_embedding.weight = nn.Parameter(torch.from_numpy(
				np.asarray(np.random.RandomState(123).uniform(
					low=-bound, high=bound, size=(self.n_entity, self.embed_dim)), dtype=np.float32)))	
			self.relation_embedding.weight = nn.Parameter(torch.from_numpy(
				np.asarray(np.random.RandomState(456).uniform(
					low=-bound, high=bound, size=(self.n_relation, self.embed_dim)), dtype=np.float32)))						
			self.weight_h = nn.Parameter(torch.from_numpy(np.asarray(np.random.RandomState(501).uniform(low=-bound, high=bound, size=(self.embed_dim)), dtype=np.float32)))
			self.weight_t = nn.Parameter(torch.from_numpy(np.asarray(np.random.RandomState(502).uniform(low=-bound, high=bound, size=(self.embed_dim)), dtype=np.float32)))
			self.weight_bias = nn.Parameter(torch.from_numpy(np.asarray(np.zeros(self.embed_dim), dtype=np.float32)))		
		self.zero = Variable(torch.zeros((1)), requires_grad=False)	
		if use_cuda:
			self.zero = self.zero.cuda()
		if encoder_type=='gru':
			self.encoder = GRU_encoder(self.embed_dim, self.n_word, bound, pretrain)
		elif encoder_type=='apcnn':
			self.encoder = APCNN(self.embed_dim, self.n_word, bound, pretrain)
		else:
			self.encoder = NBOW(self.embed_dim, self.n_word, bound, pretrain)

	def forward(self, triple, hd, td, neg_sample_r):
		h = self.entity_embedding(triple[:,0])
		t = self.entity_embedding(triple[:,1])		
		
		hd = self.encoder(hd)
		td = self.encoder(td)

		if self.combine_methods=='gate':
			gate_h = torch.sigmoid(self.gate(triple[:,0]))
			gate_t = torch.sigmoid(self.gate(triple[:,1]))
			h = gate_h * h + (1.0-gate_h) * hd
			t = gate_t * t + (1.0-gate_t) * td
			w_reg = torch.sum(torch.abs(self.gate.weight))	

		elif self.combine_methods=='dimensional_attentive':
			h = dimensional_attentive_combine(h, hd, self.theta, self.bias, self.diagonal)
			t = dimensional_attentive_combine(t, td, self.theta, self.bias, self.diagonal)
			w_reg = torch.sum(torch.abs(self.theta))			
			
		ht = self.weight_h * h + self.weight_t * t
		ht = F.dropout(F.tanh(ht + self.weight_bias), self.dropout, training=True)
		scores_htr = torch.matmul(ht, torch.t(self.relation_embedding.weight))
		log_prob_htr = torch.log(torch.clamp(mask_softmax(scores_htr, neg_sample_r), 1e-10, 1.0))
		y_htr = torch.max(neg_sample_r, self.zero)
		loss_htr = -torch.sum(log_prob_htr * y_htr/torch.sum(y_htr, 1, keepdim=True))			

		regularizer_loss = torch.sum(torch.abs(
			self.entity_embedding.weight)) + torch.sum(torch.abs(
			self.relation_embedding.weight)) + torch.sum(torch.abs(
			self.encoder.word_embedding.weight)) + w_reg + torch.sum(torch.abs(
			self.weight_h)) + torch.sum(torch.abs(
			self.weight_t))

		return loss_htr + self.reg_weight * regularizer_loss	

	def pred(self, htr, hd, td):		
		h = self.entity_embedding(htr[:,0])
		t = self.entity_embedding(htr[:,1])	
		
		hd = self.encoder(hd)
		td = self.encoder(td)	
			
		if self.combine_methods=='gate':
			gate_h = torch.sigmoid(self.gate(htr[:,0]))
			gate_t = torch.sigmoid(self.gate(htr[:,1]))
			h = gate_h * h + (1.0-gate_h) * hd
			t = gate_t * t + (1.0-gate_t) * td			
		elif self.combine_methods=='dimensional_attentive':
			h = dimensional_attentive_combine(h, hd, self.theta, self.bias, self.diagonal)
			t = dimensional_attentive_combine(t, td, self.theta, self.bias, self.diagonal)			
					
		ht = self.weight_h * h + self.weight_t * t
		ht = F.tanh(ht + self.weight_bias)		
		scores_r = torch.matmul(ht, torch.t(self.relation_embedding.weight))		
		return torch.topk(scores_r, self.n_relation)[1]