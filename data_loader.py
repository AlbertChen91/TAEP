import math
import os.path
import numpy as np
from torch.utils.data import Dataset

class DataLoader():

	def __init__(self, data_dir, subset=1.0):
		self.data_dir = data_dir
		self.subset = subset

		def load_dict(file):
			with open(os.path.join(data_dir, file), 'r') as f:
				n_dict = len(f.readlines())			
			with open(os.path.join(data_dir, file), 'r') as f:
				dict2id = {x.strip().split('\t')[0]: int(x.strip().split('\t')[1]) for x in f.readlines()}
				id2dict = {v: k for k, v in dict2id.items()}
			return n_dict, dict2id, id2dict	

		self.n_entity, self.entity2id, self.id2entity = load_dict('entity2id.txt')	
		print "number of entities: %d" % self.n_entity

		self.n_relation, self.relation2id, self.id2relation = load_dict('relation2id.txt')	
		print "number of relations: %d" % self.n_relation

		# self.n_word, self.word2id, self.id2word = load_dict('word2id.txt')
		# print "number of words: %d" % self.n_word	

		def load_triple(file_path):
			with open(file_path, 'r') as f_triple:
				triples = []
				for line_ in f_triple:
					triple = line_.strip().split('\t')
					triples.append([self.entity2id[triple[0]], self.entity2id[triple[1]], self.relation2id[triple[2]]])
				return np.asarray(triples,dtype=np.int_)

		def random_subset(triple, rate=0.5, seed=500):
			n_triple = triple.shape[0]
			selected_index = np.random.RandomState(seed).choice(range(n_triple), int(n_triple*rate), replace=False)
			return triple[selected_index]

		def gen_hr_t(triple_data):
			hr_t = dict()
			for h, t, r in triple_data:
				if h not in hr_t:
					hr_t[h] = dict()
				if r not in hr_t[h]:
					hr_t[h][r] = set()
				hr_t[h][r].add(t)
			return hr_t

		def gen_tr_h(triple_data):
			tr_h = dict()
			for h, t, r in triple_data:
				if t not in tr_h:
					tr_h[t] = dict()
				if r not in tr_h[t]:
					tr_h[t][r] = set()
				tr_h[t][r].add(h)
			return tr_h

		def gen_ht_r(triple_data):
			ht_r = dict()
			for h, t, r in triple_data:
				if h not in ht_r:
					ht_r[h] = dict()
				if t not in ht_r[h]:
					ht_r[h][t] = set()
				ht_r[h][t].add(r)
			return ht_r

		self.train_triple = random_subset(load_triple(os.path.join(data_dir, 'train.txt')), self.subset)
		print "number of training triples: %d" % self.train_triple.shape[0]

		self.valid_triple = load_triple(os.path.join(data_dir, 'valid.txt'))
		print "number of validation triples: %d" % self.valid_triple.shape[0]

		self.test_triple = load_triple(os.path.join(data_dir, 'test.txt'))
		print "number of testing triples: %d" % self.test_triple.shape[0]

		self.train_hr_t = gen_hr_t(self.train_triple)
		self.train_tr_h = gen_tr_h(self.train_triple)
		self.train_ht_r = gen_ht_r(self.train_triple)				

		all_triples = np.concatenate([self.train_triple, self.valid_triple, self.test_triple], axis=0)
		self.hr_t = gen_hr_t(all_triples)
		self.tr_h = gen_tr_h(all_triples)
		self.ht_r = gen_ht_r(all_triples)

	def get_description_list(self, max_len=0):
		maxl = 0
		with open(os.path.join(self.data_dir, 'entityWords.txt'), 'r') as f:
			entity2description = {}
			for line_ in f:
				line = line_.strip().split('\t')
				entity2description[line[0]] = [self.word2id[x] for x in line[2].split(' ')]
				if len(entity2description[line[0]])>maxl:
					maxl = len(entity2description[line[0]])
			print maxl
			if max_len==0:
				max_len = max([len(x) for x in entity2description.values()])
			for key,value in entity2description.items():
				word_list = entity2description[key]
				if len(word_list)>max_len:
					word_list = word_list[:max_len]
				word_list.extend([self.n_word]*(max_len-len(value)))
				entity2description[key] = word_list	
			description_list = np.asarray([entity2description[self.id2entity[i]] for i in range(self.n_entity)], dtype=np.int_)	
		return description_list	

	def get_pretrained_word_vec(self, dim=200, max_len=60):
		entity_embedding = np.load('pretrain_embedding/entity_embedding.npy')
		description_list = self.get_description_list(max_len)
		word_entities = {}
		for i in range(0, self.n_entity):
			for j in range(0, max_len):
				if description_list[i, j]!= self.n_word:
					if description_list[i, j] not in word_entities:
						word_entities[description_list[i, j]] = []
					word_entities[description_list[i, j]].append(i)
		word_vec = [np.mean(entity_embedding[word_entities[x]], axis=0) for x in range(0, self.n_word)]
		word_vec.append(np.zeros(dim, dtype=np.float32))
		np.save('pretrain_embedding/word_embedding.npy', np.asarray(word_vec, dtype=np.float32))	


class ProjE_Dataset(Dataset):

	def __init__(self, data_loader, dataset='train', neg_weight=0.1):

		self.data_loader = data_loader
		self.dataset = dataset
		self.neg_weight = neg_weight

	def __len__(self):
		if self.dataset == 'train':
			return self.data_loader.train_triple.shape[0]
		elif self.dataset == 'valid':
			return self.data_loader.valid_triple.shape[0]
		elif self.dataset == 'test':
			return self.data_loader.test_triple.shape[0]
		else:
			raise Exception("Invalid dataset type, option('train', 'valid', 'test')")

	def __getitem__(self, idx):
		if self.dataset == 'valid':
			pos_triple = self.data_loader.valid_triple[idx]
			return pos_triple
		
		elif self.dataset == 'test':
			pos_triple = self.data_loader.test_triple[idx]
			return pos_triple			
		
		elif self.dataset == 'train':
			pos_triple = self.data_loader.train_triple[idx]	
			neg_sample_h = [1. if x in self.data_loader.train_tr_h[pos_triple[1]][pos_triple[2]] else y for
				 x, y in enumerate(np.random.choice([0., -1.], size=self.data_loader.n_entity, p=[1 - self.neg_weight, self.neg_weight]))]
			neg_sample_t = [1. if x in self.data_loader.train_hr_t[pos_triple[0]][pos_triple[2]] else y for
				 x, y in enumerate(np.random.choice([0., -1.], size=self.data_loader.n_entity, p=[1 - self.neg_weight, self.neg_weight]))]	
			return np.asarray(pos_triple, dtype=np.int_), np.asarray(neg_sample_h, dtype=np.float32), np.asarray(neg_sample_t, dtype=np.float32)		
		else:
			raise Exception("Invalid dataset type, option('train', 'valid', 'test')")

class ProjE_R_Dataset(Dataset):

	def __init__(self, data_loader, dataset='train', neg_weight=0.5):

		self.data_loader = data_loader
		self.dataset = dataset
		self.neg_weight = neg_weight

	def __len__(self):
		if self.dataset == 'train':
			return self.data_loader.train_triple.shape[0]
		elif self.dataset == 'valid':
			return self.data_loader.valid_triple.shape[0]
		elif self.dataset == 'test':
			return self.data_loader.test_triple.shape[0]
		else:
			raise Exception("Invalid dataset type, option('train', 'valid', 'test')")

	def __getitem__(self, idx):
		if self.dataset == 'valid':
			pos_triple = self.data_loader.valid_triple[idx]
			return pos_triple
		
		elif self.dataset == 'test':
			pos_triple = self.data_loader.test_triple[idx]
			return pos_triple			
		
		elif self.dataset == 'train':
			pos_triple = self.data_loader.train_triple[idx]	
			neg_sample_r = [1. if x in self.data_loader.train_ht_r[pos_triple[0]][pos_triple[1]] else y for
				 x, y in enumerate(np.random.choice([0., -1.], size=self.data_loader.n_relation, p=[1 - self.neg_weight, self.neg_weight]))]
			return np.asarray(pos_triple, dtype=np.int_), np.asarray(neg_sample_r, dtype=np.float32)
		else:
			raise Exception("Invalid dataset type, option('train', 'valid', 'test')")

class TAProjE_Dataset(Dataset):

	def __init__(self, data_loader, dataset='train', max_len=20, neg_weight=0.1):

		self.data_loader = data_loader
		self.dataset = dataset
		self.max_len = max_len
		self.neg_weight = neg_weight
		self.description_list = self.data_loader.get_description_list(self.max_len)

	def __len__(self):
		if self.dataset == 'train':
			return self.data_loader.train_triple.shape[0]
		elif self.dataset == 'valid':
			return self.data_loader.valid_triple.shape[0]
		elif self.dataset == 'test':
			return self.data_loader.test_triple.shape[0]
		else:
			raise Exception("Invalid dataset type, option('train', 'valid', 'test')")

	def __getitem__(self, idx):
		if self.dataset == 'valid':
			pos_triple = self.data_loader.valid_triple[idx]
			hd = self.description_list[pos_triple[0]]
			td = self.description_list[pos_triple[1]]			
			return np.asarray(pos_triple, dtype=np.int_), np.asarray(hd, dtype=np.int_), np.asarray(td, dtype=np.int_)
		
		elif self.dataset == 'test':
			pos_triple = self.data_loader.test_triple[idx]
			hd = self.description_list[pos_triple[0]]
			td = self.description_list[pos_triple[1]]			
			return np.asarray(pos_triple, dtype=np.int_), np.asarray(hd, dtype=np.int_), np.asarray(td, dtype=np.int_)			
		
		elif self.dataset == 'train':
			pos_triple = self.data_loader.train_triple[idx]	
			hd = self.description_list[pos_triple[0]]
			td = self.description_list[pos_triple[1]]
			neg_sample_h = [1. if x in self.data_loader.train_tr_h[pos_triple[1]][pos_triple[2]] else y for
				 x, y in enumerate(np.random.choice([0., -1.], size=self.data_loader.n_entity, p=[1 - self.neg_weight, self.neg_weight]))]
			neg_sample_t = [1. if x in self.data_loader.train_hr_t[pos_triple[0]][pos_triple[2]] else y for
				 x, y in enumerate(np.random.choice([0., -1.], size=self.data_loader.n_entity, p=[1 - self.neg_weight, self.neg_weight]))]	
			return np.asarray(pos_triple, dtype=np.int_), np.asarray(hd, dtype=np.int_), np.asarray(td, dtype=np.int_), np.asarray(neg_sample_h, dtype=np.float32), np.asarray(neg_sample_t, dtype=np.float32)
		else:
			raise Exception("Invalid dataset type, option('train', 'valid', 'test')")

class TAProjE_R_Dataset(Dataset):

	def __init__(self, data_loader, dataset='train', max_len=20, neg_weight=0.5):

		self.data_loader = data_loader
		self.dataset = dataset
		self.max_len = max_len
		self.neg_weight = neg_weight
		self.description_list = self.data_loader.get_description_list(self.max_len)

	def __len__(self):
		if self.dataset == 'train':
			return self.data_loader.train_triple.shape[0]
		elif self.dataset == 'valid':
			return self.data_loader.valid_triple.shape[0]
		elif self.dataset == 'test':
			return self.data_loader.test_triple.shape[0]
		else:
			raise Exception("Invalid dataset type, option('train', 'valid', 'test')")

	def __getitem__(self, idx):
		if self.dataset == 'valid':
			pos_triple = self.data_loader.valid_triple[idx]
			hd = self.description_list[pos_triple[0]]
			td = self.description_list[pos_triple[1]]			
			return np.asarray(pos_triple, dtype=np.int_), np.asarray(hd, dtype=np.int_), np.asarray(td, dtype=np.int_)
		
		elif self.dataset == 'test':
			pos_triple = self.data_loader.test_triple[idx]
			hd = self.description_list[pos_triple[0]]
			td = self.description_list[pos_triple[1]]			
			return np.asarray(pos_triple, dtype=np.int_), np.asarray(hd, dtype=np.int_), np.asarray(td, dtype=np.int_)			
		
		elif self.dataset == 'train':
			pos_triple = self.data_loader.train_triple[idx]	
			hd = self.description_list[pos_triple[0]]
			td = self.description_list[pos_triple[1]]
			neg_sample_r = [1. if x in self.data_loader.train_ht_r[pos_triple[0]][pos_triple[1]] else y for
				 x, y in enumerate(np.random.choice([0., -1.], size=self.data_loader.n_relation, p=[1 - self.neg_weight, self.neg_weight]))]
			return np.asarray(pos_triple, dtype=np.int_), np.asarray(hd, dtype=np.int_), np.asarray(td, dtype=np.int_), np.asarray(neg_sample_r, dtype=np.float32)
		else:
			raise Exception("Invalid dataset type, option('train', 'valid', 'test')")

if __name__=='__main__':
	data_loader = DataLoader('data/FB15K_D/')
	# data_loader.get_pretrained_word_vec(200, 60)
	data_loader.get_description_list()