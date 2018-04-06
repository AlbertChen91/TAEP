import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.utils import data
from data_loader import DataLoader, ProjE_R_Dataset, TAProjE_R_Dataset
from model import ProjE_R, TAProjE_R
torch.backends.cudnn.benchmark=True

parser = argparse.ArgumentParser(description='TAProjE_R')
parser.add_argument('--data', dest='data_dir', type=str, help="Data folder", default='./data/FB15K_D/')
parser.add_argument('--model_', dest='model_', type=str, help="Model selected", default='SSP')
parser.add_argument('--lr', dest='lr', type=float, help="Learning rate", default=0.1)
parser.add_argument("--dim", dest='dim', type=int, help="Embedding dimension", default=100)
parser.add_argument("--max_len", dest="max_len", type=int, help="Max description length", default=20)
parser.add_argument('--encoder', dest='encoder', type=str, help="Encoder type", default='nbow')
parser.add_argument('--combine_methods', dest='combine_methods', type=str, help="Text embedding and entity embedding combination methods", default='gate')
parser.add_argument('--diagonal', dest='diagonal', action="store_true", help="If dimensional attentive combination parameters use diagonal matrix or not", default=False)
parser.add_argument("--batch", dest='batch', type=int, help="Batch size", default=10)
parser.add_argument("--test_batch", dest='test_batch', type=int, help="Batch size", default=100)
parser.add_argument("--neg_weight", dest='neg_weight', type=float, help="Negative sample rate", default=0.5)
parser.add_argument("--worker", dest='n_worker', type=int, help="Evaluation worker", default=3)
parser.add_argument("--save_dir", dest='save_dir', type=str, help="Saved model path", default='./')
parser.add_argument("--resume", dest='resume', type=str, help="Resume from model file", default="")
parser.add_argument('--pretrain', dest='pretrain', action="store_true", help="If use pretrained embedding or not", default=False)
parser.add_argument("--subset", dest='subset', type=float, help="The rate of used subset from training set, default whole training set", default=1.0)
parser.add_argument("--eval_start", dest='eval_start', type=int, help="Epoch when evaluation start", default=90)
parser.add_argument("--eval_per", dest='eval_per', type=int, help="Evaluation per x iteration", default=1)
parser.add_argument("--save_m", dest='save_m', type=int, help="Number of saved models", default=1)
parser.add_argument("--epochs", dest='epochs', type=int, help="Epochs to run", default=100)
parser.add_argument("--optimizer", dest='optimizer', type=str, help="Optimizer", default='adam')
parser.add_argument('--reg_weight', dest='reg_weight', type=float, help="The weight of L1 regularization", default=1e-5)
parser.add_argument('--dropout', dest='dropout', type=float, help="The dropout rate", default=0.5)
parser.add_argument('--clip', dest='clip', type=float, help="The gradient clip max norm", default=0.0)
parser.add_argument("--tolerance", dest="tolerance", type=int, help="Early stopping tolerance", default=3)
parser.add_argument("--early_stop_epsilon", dest="early_stop_epsilon", type=float, help="Early stopping by mean rank evaluation changes", default=0.0)

args = parser.parse_args()
use_cuda = torch.cuda.is_available()

def use_optimizer(model, lr, weight_decay=0, lr_decay=0, momentum=0, rho=0.9, method='sgd'):
	if method=='sgd':
		return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
	elif method=='adagrad':
		return optim.Adagrad(model.parameters(), lr=lr, lr_decay=lr_decay, weight_decay=weight_decay)
	elif method=='adadelta':
		return optim.Adadelta(model.parameters(), rho=rho, lr=lr, weight_decay=weight_decay)
	elif method=='adam':
		return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
	else:
		raise Exception("Invalid method, option('sgd', 'adagrad', 'adadelta', 'adam')")

def save_checkpoint(state, epoch, filename='_epoch.mod.tar'):
	remove_model = args.save_dir+str(epoch-args.save_m-1)+'_epoch.mod.tar'
	if os.path.exists(remove_model):
		 os.remove(remove_model)
	torch.save(state, filename)	

def evaluation(testing_data, r_pred, ht_r):
	assert len(testing_data) == len(r_pred)

	mean_rank_r = list()
	filtered_mean_rank_r = list()

	for i in range(len(testing_data)):
		h = testing_data[i, 0]
		t = testing_data[i, 1]
		r = testing_data[i, 2]
		# mean rank

		mr = 1
		for val in r_pred[i]:
			if val == r:
				mean_rank_r.append(mr)
				break
			mr += 1

		# filtered mean rank
		fmr = 1
		for val in r_pred[i]:
			if val == r:
				filtered_mean_rank_r.append(fmr)
				break
			if h in ht_r and t in ht_r[h] and val in ht_r[h][t]:
				continue
			else:
				fmr += 1

	return mean_rank_r, filtered_mean_rank_r

def test_ProjE_R(model, data_loader, test_type='test'): 
	acc_mean_rank_r = list()
	acc_filtered_mean_rank_r = list()
	testloader = data.DataLoader(ProjE_R_Dataset(data_loader, dataset=test_type, neg_weight=args.neg_weight), batch_size=args.test_batch, shuffle=False, num_workers=args.n_worker)
	for batch_idx, pos_triple in enumerate(testloader):	
		triple = Variable(pos_triple)
		if use_cuda:
			triple = triple.cuda()
		r_pred = model.pred(triple)
		r_pred = r_pred.data.cpu().numpy()
		mr_r, fmr_r = evaluation(pos_triple.numpy(), r_pred, data_loader.ht_r)
		acc_mean_rank_r += mr_r
		acc_filtered_mean_rank_r += fmr_r
		print 'batch %d tested' % batch_idx

	mean_rank_r = np.mean(acc_mean_rank_r)
	filtered_mean_rank_r = np.mean(acc_filtered_mean_rank_r)
	hit_1_r = np.mean(np.asarray(acc_mean_rank_r, dtype=np.int32) <= 1)
	filtered_hit_1_r = np.mean(np.asarray(acc_filtered_mean_rank_r, dtype=np.int32) <= 1)		

	return mean_rank_r, filtered_mean_rank_r, hit_1_r, filtered_hit_1_r

def train_epoch_ProjE_R(model, data_loader, optimizer, epoch):
	trainloader = data.DataLoader(ProjE_R_Dataset(data_loader, dataset='train', neg_weight=args.neg_weight), batch_size=args.batch, shuffle=True, num_workers=args.n_worker)
	batchs = len(trainloader)
	losses = []
	for batch_idx, (triple, neg_sample_r) in enumerate(trainloader):
		triple, neg_sample_r = Variable(triple), Variable(neg_sample_r)
		if use_cuda:
			triple, neg_sample_r = triple.cuda(), neg_sample_r.cuda()
		loss = model(triple, neg_sample_r)
		loss_ = loss.data[0]
		losses.append(loss_)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		print '[Epoch %d/%d ] | Iter %d/%d | Loss %3f' % (epoch, args.epochs, batch_idx, batchs, loss_)
	return np.mean(np.array(losses))

def train_ProjE_R():
	data_loader = DataLoader(args.data_dir, args.subset)
	model = ProjE_R(args.dim, data_loader.n_entity, data_loader.n_relation, args.dropout, args.reg_weight)
	if use_cuda:
		model = model.cuda()
	optimizer = use_optimizer(model, lr=args.lr, method=args.optimizer)
	start_epoch = 0
	best_filtered_hit_1 = 0
	tolerance = 0
	stopped = False
	ave_losses = []
	if not os.path.exists(args.save_dir):
		os.makedirs(args.save_dir)		
	if args.resume != '':
		if os.path.isfile(args.resume):
			print "=> loading checkpoint '{}'".format(args.resume)
			checkpoint = torch.load(args.resume)
			start_epoch = checkpoint['epoch']
			ave_losses = checkpoint['losses']
			model.load_state_dict(checkpoint['state_dict'])
			optimizer.load_state_dict(checkpoint['optimizer'])
			print model.state_dict().keys()
			print "=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch'])
		else:
			print "=> no checkpoint found at '{}'".format(args.resume)
	for epoch in range(start_epoch, args.epochs):
		loss = train_epoch_ProjE_R(model, data_loader, optimizer, epoch)
		print 'Epoch %d train over, average loss: %3f' % (epoch, loss)
		ave_losses.append(loss)
		if epoch%args.eval_per==0 or epoch==args.epochs-1:		
			print 'Waiting for model validation...'
			mean_rank, filtered_mean_rank, hit_1, filtered_hit_1 = test_ProjE_R(model, data_loader, 'valid')
			print "[Validation] epoch %d mean rank: %.3f filtered mean rank: %.3f hit@1: %.3f filtered hit@1: %.3f" % (
				epoch, mean_rank, filtered_mean_rank, hit_1, filtered_hit_1)	
			with open(args.save_dir+'validation_log.txt', 'ab') as f:
				f.write("[Validation] epoch %d mean rank: %.3f filtered mean rank: %.3f hit@1: %.3f filtered hit@1: %.3f loss: %.3f\n" % (
					epoch, mean_rank, filtered_mean_rank, hit_1, filtered_hit_1, loss))
			# Using validation set for earlystopping.
			if filtered_hit_1 - best_filtered_hit_1 <= args.early_stop_epsilon:
				tolerance += 1
				if tolerance >= args.tolerance:
					stopped = True
			else:
				tolerance = 0
				best_filtered_hit_1 = filtered_hit_1  		
			print 'Waiting for model testing...'
			mean_rank, filtered_mean_rank, hit_1, filtered_hit_1 = test_ProjE_R(model, data_loader, 'test')
			print "[Testing] epoch %d mean rank: %.3f filtered mean rank: %.3f hit@1: %.3f filtered hit@1: %.3f" % (
				epoch, mean_rank, filtered_mean_rank, hit_1, filtered_hit_1)	
			with open(args.save_dir+'testing_log.txt', 'ab') as f:
				f.write("[Testing] epoch %d mean rank: %.3f filtered mean rank: %.3f hit@1: %.3f filtered hit@1: %.3f loss: %.3f\n" % (
					epoch, mean_rank, filtered_mean_rank, hit_1, filtered_hit_1, loss))
		save_checkpoint({
			'epoch': epoch + 1,
			'losses': ave_losses,
			'state_dict': model.state_dict(),
			'optimizer' : optimizer.state_dict()
		}, epoch, args.save_dir+str(epoch)+'_epoch.mod.tar')
		
		if stopped:
			break

def test_TAProjE_R(model, data_loader, test_type='test'): 
	acc_mean_rank_r = list()
	acc_filtered_mean_rank_r = list()
	testloader = data.DataLoader(TAProjE_R_Dataset(data_loader, dataset=test_type, max_len=args.max_len, neg_weight=args.neg_weight), batch_size=args.test_batch, shuffle=False, num_workers=args.n_worker)
	for batch_idx, (pos_triple, hd, td) in enumerate(testloader):	
		triple, hd, td = Variable(pos_triple), Variable(hd), Variable(td)
		if use_cuda:
			triple, hd, td = triple.cuda(), hd.cuda(), td.cuda()
		r_pred = model.pred(triple, hd, td)
		r_pred = r_pred.data.cpu().numpy()
		mr_r, fmr_r = evaluation(pos_triple.numpy(), r_pred, data_loader.ht_r)
		acc_mean_rank_r += mr_r
		acc_filtered_mean_rank_r += fmr_r
		print 'batch %d tested' % batch_idx

	mean_rank_r = np.mean(acc_mean_rank_r)
	filtered_mean_rank_r = np.mean(acc_filtered_mean_rank_r)
	hit_1_r = np.mean(np.asarray(acc_mean_rank_r, dtype=np.int32) <= 1)
	filtered_hit_1_r = np.mean(np.asarray(acc_filtered_mean_rank_r, dtype=np.int32) <= 1)		

	return mean_rank_r, filtered_mean_rank_r, hit_1_r, filtered_hit_1_r	

def train_epoch_TAProjE_R(model, data_loader, optimizer, epoch):
	trainloader = data.DataLoader(TAProjE_R_Dataset(data_loader, dataset='train', max_len=args.max_len, neg_weight=args.neg_weight), batch_size=args.batch, shuffle=True, num_workers=args.n_worker)
	batchs = len(trainloader)
	losses = []
	for batch_idx, (triple, hd, td, neg_sample_r) in enumerate(trainloader):
		triple, hd, td, neg_sample_r = Variable(triple), Variable(hd), Variable(td), Variable(
			neg_sample_r)
		if use_cuda:
			triple, hd, td, neg_sample_r = triple.cuda(), hd.cuda(), td.cuda(
				), neg_sample_r.cuda()
		loss = model(triple, hd, td, neg_sample_r)
		loss_ = loss.data[0]
		losses.append(loss_)
		optimizer.zero_grad()
		loss.backward()
		if args.clip!=0:
			nn.utils.clip_grad_norm(model.parameters(), args.clip)
		optimizer.step()
		print '[Epoch %d/%d ] | Iter %d/%d | Loss %3f' % (epoch, args.epochs, batch_idx, batchs, loss_)
	return np.mean(np.array(losses))

def train_TAProjE_R():
	data_loader = DataLoader(args.data_dir, args.subset)
	model = TAProjE_R(args.dim, data_loader.n_entity, data_loader.n_relation, data_loader.n_word, args.dropout, args.reg_weight, args.encoder, args.combine_methods, args.diagonal, args.pretrain)
	if use_cuda:
		model = model.cuda()
	optimizer = use_optimizer(model, lr=args.lr, method=args.optimizer)
	start_epoch = 0
	best_filtered_hit_1 = 0
	tolerance = 0
	stopped = False
	ave_losses = []
	# lowest_loss = 10000
	if not os.path.exists(args.save_dir):
		os.makedirs(args.save_dir)		
	if args.resume != '':
		if os.path.isfile(args.resume):
			print "=> loading checkpoint '{}'".format(args.resume)
			checkpoint = torch.load(args.resume)
			start_epoch = checkpoint['epoch']
			ave_losses = checkpoint['losses']
			model.load_state_dict(checkpoint['state_dict'])
			optimizer.load_state_dict(checkpoint['optimizer'])
			print model.state_dict().keys()
			print "=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch'])
		else:
			print "=> no checkpoint found at '{}'".format(args.resume)
	for epoch in range(start_epoch, args.epochs):
		loss = train_epoch_TAProjE_R(model, data_loader, optimizer, epoch)
		print 'Epoch %d train over, average loss: %3f' % (epoch, loss)
		ave_losses.append(loss)
		if epoch%args.eval_per==0 or epoch==args.epochs-1:
			print 'Waiting for model validation...'
			mean_rank, filtered_mean_rank, hit_1, filtered_hit_1 = test_TAProjE_R(model, data_loader, 'valid')
			print "[Validation] epoch %d mean rank: %.3f filtered mean rank: %.3f hit@1: %.3f filtered hit@1: %.3f" % (
				epoch, mean_rank, filtered_mean_rank, hit_1, filtered_hit_1)	
			with open(args.save_dir+'validation_log.txt', 'ab') as f:
				f.write("[Validation] epoch %d mean rank: %.3f filtered mean rank: %.3f hit@1: %.3f filtered hit@1: %.3f loss: %.3f\n" % (
					epoch, mean_rank, filtered_mean_rank, hit_1, filtered_hit_1, loss))
			# Using validation set for earlystopping.
			if filtered_hit_1 - best_filtered_hit_1 <= args.early_stop_epsilon:
				tolerance += 1
				if tolerance >= args.tolerance:
					stopped = True
			else:
				tolerance = 0
				best_filtered_hit_1 = filtered_hit_1  		
			print 'Waiting for model testing...'
			mean_rank, filtered_mean_rank, hit_1, filtered_hit_1 = test_TAProjE_R(model, data_loader, 'test')
			print "[Testing] epoch %d mean rank: %.3f filtered mean rank: %.3f hit@1: %.3f filtered hit@1: %.3f" % (
				epoch, mean_rank, filtered_mean_rank, hit_1, filtered_hit_1)	
			with open(args.save_dir+'testing_log.txt', 'ab') as f:
				f.write("[Testing] epoch %d mean rank: %.3f filtered mean rank: %.3f hit@1: %.3f filtered hit@1: %.3f loss: %.3f\n" % (
					epoch, mean_rank, filtered_mean_rank, hit_1, filtered_hit_1, loss))
		save_checkpoint({
			'epoch': epoch + 1,
			'losses': ave_losses,
			'state_dict': model.state_dict(),
			'optimizer' : optimizer.state_dict()
		}, epoch, args.save_dir+str(epoch)+'_epoch.mod.tar')
		if stopped:
			break

def get_model_param():
	data_loader = DataLoader(args.data_dir)
	model = ProjE_R(args.dim, data_loader.n_entity, data_loader.n_relation, args.dropout, args.reg_weight)
	if use_cuda:
		model = model.cuda()
	optimizer = use_optimizer(model, lr=args.lr, method=args.optimizer)
	if not os.path.exists(args.save_dir):
		os.makedirs(args.save_dir)		
	if args.resume != '':
		if os.path.isfile(args.resume):
			print "=> loading checkpoint '{}'".format(args.resume)
			checkpoint = torch.load(args.resume)
			start_epoch = checkpoint['epoch']
			ave_losses = checkpoint['losses']
			model.load_state_dict(checkpoint['state_dict'])
			optimizer.load_state_dict(checkpoint['optimizer'])
			print model.state_dict().keys()
			print "=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch'])
		else:
			print "=> no checkpoint found at '{}'".format(args.resume)
	np.save('pretrain_embedding/entity_embedding', model.entity_embedding.weight.data.cpu().numpy())
	np.save('pretrain_embedding/relation_embedding', model.relation_embedding.weight.data.cpu().numpy())
	np.save('pretrain_embedding/weight_h', model.weight_h.data.cpu().numpy())
	np.save('pretrain_embedding/weight_hr', model.weight_hr.data.cpu().numpy())
	np.save('pretrain_embedding/weight_t', model.weight_t.data.cpu().numpy())
	np.save('pretrain_embedding/weight_tr', model.weight_tr.data.cpu().numpy())	
	np.save('pretrain_embedding/weight_bias_h', model.weight_bias_h.data.cpu().numpy())
	np.save('pretrain_embedding/weight_bias_t', model.weight_bias_t.data.cpu().numpy())
	print 'OK'
	# data_loader = DataLoader(args.data_dir)
	# model = ProjE_D(args.dim, data_loader.n_entity, data_loader.n_relation, data_loader.n_word, args.dropout, args.reg_weight)
	# if use_cuda:
	# 	model = model.cuda()
	# optimizer = use_optimizer(model, lr=args.lr, method=args.optimizer)
	# # lowest_loss = 10000
	# if not os.path.exists(args.save_dir):
	# 	os.makedirs(args.save_dir)		
	# if args.resume != '':
	# 	if os.path.isfile(args.resume):
	# 		print "=> loading checkpoint '{}'".format(args.resume)
	# 		checkpoint = torch.load(args.resume)
	# 		start_epoch = checkpoint['epoch']
	# 		ave_losses = checkpoint['losses']
	# 		model.load_state_dict(checkpoint['state_dict'])
	# 		optimizer.load_state_dict(checkpoint['optimizer'])
	# 		print model.state_dict().keys()
	# 		print "=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch'])
	# 	else:
	# 		print "=> no checkpoint found at '{}'".format(args.resume)	
	# print model.gate.weight.data.cpu().numpy()[2]


def main():
	print args
	# get_model_param()
	if args.model_=='ProjE_R':
		train_ProjE_R()
	elif args.model_=='TAProjE_R':
		train_TAProjE_R()

if __name__ == '__main__':
	main()


